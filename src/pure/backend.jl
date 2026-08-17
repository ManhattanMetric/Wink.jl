# The pure-Julia backend: Wink's DEFAULT local model route. No C, no vendored
# binaries, no servers — GGUF reader, SPM/BPE tokenizers, and generic-array
# forward passes (gemma-3, gemma-4 MoE, OLMoE), all Julia.
#
# GPU story (portable to any GPUArrays backend, vendor package user-loaded):
# `local_model!(path; array = MtlArray)` keeps TWO views of the same weights —
# the host model (mmap-backed, zero-copy) for token-by-token generation,
# where launch latency makes the CPU the fast device, and a device twin for
# prefill, where batch, plus the fused MoE/attention kernels, put the GPU
# ~4× ahead. The KV cache lives on the host; large prefills ship the prefix
# up, run chunked on the device, and ship the result back.
#
# The KV strategy is the one proven on the llama.cpp backend — the cache
# holds the canonical history render (add_assistant = false), each call
# appends the canonical delta, and the generation scratch is rolled back —
# except that rollback here is one integer assignment: `cache.n = base`.
#
# Tool calls are constrained at the logits (constrain.jl): free sampling
# until the model emits <|tool_call>, then propose-and-verify against the
# compiled tool matchers until the call closes — a malformed call is
# unrepresentable, exactly the lazy-GBNF semantic of the llama.cpp backend,
# minus llama.cpp. Tools require a template family with a canonical call
# syntax (gemma-4); other families run text-only.

mutable struct PureModel
    path::String
    engine::Module                    # Gemma3 | Gemma4 | OLMoE
    model::Any                        # host model: mmap-backed, generation
    gpu_model::Any                    # device twin for prefill, or nothing
    tokmod::Module                    # SPMTokenizer | BPETokenizer
    tok::Any
    family::ChatTemplate
    cache::Any
    kv_text::String
    n_ctx::Int
    eog::Vector{Int}
end

const PURE_MODEL = Ref{Union{Nothing, PureModel}}(nothing)

const PURE_ENGINES = Dict{String, Module}()   # filled at __init__ (world age)
function _pure_engine(arch::AbstractString)
    isempty(PURE_ENGINES) && merge!(PURE_ENGINES, Dict(
        "gemma3" => Gemma3, "gemma4" => Gemma4, "olmoe" => OLMoE))
    get(PURE_ENGINES, arch, nothing)
end

function _load_pure!(path::AbstractString; n_ctx::Integer = 16_384,
        array = nothing)
    p = expanduser(path)
    isfile(p) || error("model not found: $p")
    f = GGUF.GGUFFile(p)
    arch = String(GGUF.metadata(f, "general.architecture"))
    engine = _pure_engine(arch)
    engine === nothing &&
        error("pure backend has no $arch forward pass (have: " *
              join(sort(collect(keys(PURE_ENGINES))), ", ") * ")")
    tokmodel = String(GGUF.metadata(f, "tokenizer.ggml.model", "llama"))
    # "gemma4" is SPM with gemma-4's vocab conventions — the pure SPM
    # tokenizer handles it (oracle-verified on the 26B QAT file)
    tokmod = tokmodel in ("llama", "gemma4") ? SPMTokenizer :
             tokmodel == "gpt2" ? BPETokenizer :
             error("pure backend has no \"$tokmodel\" tokenizer (llama/gpt2)")
    tok = tokmod.Tokenizer(f)
    m = engine.load_model(f)
    gpu = array === nothing ? nothing : Adapt.adapt(array, m)
    family = template_family(get(f.meta, "tokenizer.chat_template", nothing))
    family isa ChatMLTemplate &&
        @warn "unrecognized chat template family; ChatML fallback will degrade output"
    eog = unique(filter(>=(0), [tok.eos,
        get(tok.id_of, "<eos>", -1), get(tok.id_of, "<end_of_turn>", -1),
        get(tok.id_of, "<eot>", -1), get(tok.id_of, "<|endoftext|>", -1)]))
    PURE_MODEL[] = PureModel(p, engine, m, gpu, tokmod, tok, family,
        engine.KVCache(m; capacity = Int(n_ctx)), "", Int(n_ctx), eog)
    CONFIG.chat_model = "pure:" * basename(p)
    budget = _coordinate_budget!(n_ctx)
    printstyled(CONFIG.status_io,
        "  [Wink] pure-Julia model loaded: ", basename(p),
        " ($arch, n_ctx = $n_ctx, context_budget = $budget, ",
        "template: $(nameof(typeof(family)))",
        gpu === nothing ? "" : ", GPU prefill", ")\n";
        color = :light_black)
    return CONFIG.chat_model
end

# ---- hybrid prefill -----------------------------------------------------------

const GPU_PREFILL_MIN = 256   # below this, upload+download beats nothing
const GPU_PREFILL_CHUNK = 512

# copy the first n cache positions between host and device caches; the
# prefix is contiguous in each (hd, nkv, capacity) array, so offset copyto!
# moves it as one linear block per layer
function _copy_cache!(dst, src, n::Int)
    n == 0 && return dst
    for il in eachindex(src.K)
        len = size(src.K[il], 1) * size(src.K[il], 2) * n
        copyto!(dst.K[il], 1, src.K[il], 1, len)
        copyto!(dst.V[il], 1, src.V[il], 1, len)
    end
    return dst
end

# advance the model over toks, returning the final logits column as an
# n_vocab × 1 HOST matrix; big batches detour through the GPU twin
function _prefill!(pm::PureModel, toks::Vector{Int})
    if pm.gpu_model !== nothing && length(toks) >= GPU_PREFILL_MIN
        gc = pm.engine.KVCache(pm.gpu_model;
            capacity = pm.cache.n + length(toks))
        _copy_cache!(gc, pm.cache, pm.cache.n)
        gc.n = pm.cache.n
        local logits
        for lo in 1:GPU_PREFILL_CHUNK:length(toks)
            logits = pm.engine.step!(pm.gpu_model, gc,
                toks[lo:min(lo + GPU_PREFILL_CHUNK - 1, end)])
            yield()               # keep the thinking spinner alive
        end
        _copy_cache!(pm.cache, gc, gc.n)
        pm.cache.n = gc.n
        return reshape(Array(logits[:, end]), :, 1)
    end
    logits = pm.engine.step!(pm.model, pm.cache, toks)
    return reshape(collect(Float32, view(logits, :, size(logits, 2))), :, 1)
end

# ---- sampling -----------------------------------------------------------------

# temperature + top-k sampling over one logits column (host-side; we own
# the sampler loop, which is what makes constrained tool calls possible)
function _pure_sample(logits::AbstractVector{Float32}; temp::Float32 = 0.7f0,
        top_k::Int = 40)
    temp <= 0 && return Int(argmax(logits)) - 1
    idx = partialsortperm(logits, 1:min(top_k, length(logits)); rev = true)
    w = exp.((logits[idx] .- logits[idx[1]]) ./ temp)
    w ./= sum(w)
    r = rand(Float32)
    acc = 0.0f0
    for (i, p) in zip(idx, w)
        acc += p
        acc >= r && return Int(i) - 1
    end
    return Int(idx[end]) - 1
end

# ---- the seam implementation --------------------------------------------------

function _pure_aitools(pm::PureModel, history, tools)
    t0 = time()
    with_tools = !isempty(tools) && pm.family isa Gemma4Template
    isempty(tools) || with_tools || debug_status(CONFIG.status_io,
        "$(nameof(typeof(pm.family))) has no canonical tool-call syntax; " *
        "$(length(tools)) tools not offered to the model")
    msgs = [_local_msg(m) for m in history]
    schemas = with_tools ? [_tool_schema(t) for t in tools] : Dict{String, Any}[]
    base = render_chat(pm.family, msgs; tools = schemas, add_assistant = false)
    full = render_chat(pm.family, msgs; tools = schemas, add_assistant = true)
    tokd(txt, add) = pm.tokmod.tokenize(pm.tok, String(txt);
        add_special = add, parse_special = true)
    delta = if !isempty(pm.kv_text) && startswith(base, pm.kv_text)
        tokd(SubString(base, ncodeunits(pm.kv_text) + 1), false)
    else
        pm.cache.n = 0          # splice (compaction, reset): rebuild the cache
        tokd(base, true)
    end
    suffix = tokd(SubString(full, ncodeunits(base) + 1), false)
    n0 = pm.cache.n
    needed = n0 + length(delta) + length(suffix) + 16
    needed <= pm.n_ctx ||
        error("context window full ($needed tokens needed, n_ctx = " *
              "$(pm.n_ctx)): run :compact or Wink.reset!(), or reload with " *
              "local_model!(path; n_ctx = ...)")
    col = _prefill!(pm, vcat(delta, suffix))
    pm.kv_text = base
    base_n = n0 + length(delta)
    suffix_base = pm.cache.n

    matchers = with_tools ? compile_matchers(schemas) : ToolMatcher[]
    banned = with_tools ? filter(>=(0),
        [get(pm.tok.id_of, "<|tool_response>", -1)]) : Int[]

    function generate_once!(col)
        out = IOBuffer()
        n_gen = 0
        tail = ""
        call_bytes = UInt8[]
        in_call = false
        while n_gen < CONFIG.max_tokens && pm.cache.n < pm.n_ctx - 4
            logits = vec(col)
            for b in banned
                logits[b + 1] = -Inf32
            end
            local next, s
            if in_call
                # propose-and-verify: erase invalid candidates and redraw
                # (each rejection erases one token, so this terminates; the
                # isinf guard catches the theoretical everything-masked case)
                while true
                    isinf(maximum(logits)) && return String(take!(out)), n_gen
                    next = _pure_sample(logits)
                    s = pm.tokmod.piece(pm.tok, next; special = true)
                    st = isempty(s) ? :invalid :
                         call_state(matchers, vcat(call_bytes, codeunits(s)))
                    st === :invalid || break
                    logits[next + 1] = -Inf32
                end
                append!(call_bytes, codeunits(s))
            else
                next = _pure_sample(logits)
                next in pm.eog && break
                s = pm.tokmod.piece(pm.tok, next; special = true)
            end
            print(out, s)
            n_gen += 1
            tail = last(tail * s, 24)
            if !in_call && with_tools && occursin(CALL_TRIGGER, tail)
                in_call = true
                gen = String(take!(out))
                print(out, gen)              # peek without consuming
                trig = findlast(CALL_TRIGGER, gen)
                call_bytes = Vector{UInt8}(codeunits(gen)[(last(trig) + 1):end])
                if call_state(matchers, call_bytes) === :invalid
                    in_call = false          # unsalvageable region: free text
                    call_bytes = UInt8[]
                end
            end
            in_call && call_state(matchers, call_bytes) === :complete && break
            step = pm.engine.step!(pm.model, pm.cache, [next])
            col = reshape(collect(Float32, view(step, :, 1)), :, 1)
            yield()                          # keep the thinking spinner alive
        end
        return String(take!(out)), n_gen
    end

    text, n_gen = generate_once!(col)
    if parse_tool_call(text) === nothing &&
       isempty(strip(g4_strip_thinking(text)))
        # an empty round (immediate EOG, or pure channel noise): roll the
        # scratch back and redraw once rather than ending the turn silently
        pm.cache.n = suffix_base
        text, n_gen = generate_once!(col)
    end

    total = pm.cache.n
    pm.cache.n = base_n          # roll the scratch back to the canonical render
    call = parse_tool_call(text)
    call === nothing &&
        return PT.AIToolRequest(;
            content = String(strip(replace(g4_strip_thinking(text),
                CALL_CLOSE => "", "<|tool_response>" => ""))),
            tool_calls = PT.ToolMessage[], tokens = (total, n_gen),
            elapsed = time() - t0)
    idx = findfirst(CALL_TRIGGER, text)
    preface = g4_strip_thinking(SubString(text, 1, first(idx) - 1))
    args = Dict{Symbol, Any}(Symbol(k) => v for (k, v) in call.args)
    tc = PT.ToolMessage(;
        tool_call_id = "local-" * string(LOCAL_CALL_COUNTER[] += 1),
        raw = JSON3.write(call.args), name = call.name, args)
    return PT.AIToolRequest(; content = isempty(preface) ? nothing : String(preface),
        tool_calls = [tc], tokens = (total, n_gen), elapsed = time() - t0)
end
