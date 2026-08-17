# The local backend: a GGUF model loaded INTO the same Julia process as the
# session it pair-programs in. Activated by `local_model!`, it takes over
# `_aitools_call`'s default route (explicit schemas — tests — still go to
# PromptingTools), so the whole agent loop, compaction included, runs
# against local weights with no server, no HTTP, no C, and no vendored
# binaries — GGUF reader, SPM/BPE tokenizers, and generic-array forward
# passes (gemma-3, gemma-4 MoE, OLMoE), all Julia.
#
# GPU story (portable to any GPUArrays backend, vendor package user-loaded):
# `local_model!(path; array = MtlArray)` keeps TWO views of the same weights —
# the host model (mmap-backed, zero-copy) for token-by-token generation,
# where launch latency makes the CPU the fast device, and a device twin for
# prefill, where batch, plus the fused MoE/attention kernels, put the GPU
# ~4× ahead. The KV cache lives on the host; large prefills ship the prefix
# up, run chunked on the device, and ship the result back.
#
# Conversation state: the KV cache holds exactly the CANONICAL rendering of
# the history (render_chat with add_assistant = false). Each call appends
# the canonical delta (splices — compaction — break the prefix and trigger
# a full re-decode), then decodes the generation-prompt suffix and the
# sampled tokens as a scratch region that is rolled back after generation —
# rollback here is one integer assignment: `cache.n = base`.
#
# Tool calls are constrained at the logits (constrain.jl): free sampling
# until the model emits <|tool_call>, then propose-and-verify against the
# compiled tool matchers until the call closes — a malformed call is
# unrepresentable. Tools require a template family with a canonical call
# syntax (gemma-4); other families run text-only.

mutable struct LocalModel
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

const LOCAL_MODEL = Ref{Union{Nothing, LocalModel}}(nothing)
const LOCAL_CALL_COUNTER = Ref(0)

const LOCAL_ENGINES = Dict{String, Module}()   # filled lazily (world age)
function _local_engine(arch::AbstractString)
    isempty(LOCAL_ENGINES) && merge!(LOCAL_ENGINES, Dict(
        "gemma3" => Gemma3, "gemma4" => Gemma4, "olmoe" => OLMoE))
    get(LOCAL_ENGINES, arch, nothing)
end

# The global context_budget default assumes frontier-sized windows; a local
# model's real ceiling is n_ctx. Lower the budget (never raise it — explicit
# user settings win) so the fold/distill/mine ladder fires well before the
# window fills.
function _coordinate_budget!(n_ctx::Integer)
    auto = (3 * Int(n_ctx)) ÷ 4
    if CONFIG.context_budget == 0
        @warn "auto-compaction is disabled (context_budget = 0); the local " *
              "model's $(n_ctx)-token context can overflow mid-conversation"
    elseif CONFIG.context_budget > auto
        CONFIG.context_budget = auto
    end
    return CONFIG.context_budget
end

"""
    local_model!(path::AbstractString; n_ctx = 16_384, array = nothing)

Load a GGUF model INTO this Julia process and route all of Wink's chat
through it — no server, no HTTP, no C: the GGUF reader (F32/F16/BF16 +
q4_0/q8_0/q6_K zero-copy over the mmap), SPM and GPT-2 BPE tokenizers, and
generic-array forward passes for gemma-3, gemma-4 (MoE), and OLMoE are all
Julia, oracle-validated against llama.cpp.

Tool calls are constrained at the logits (a malformed call is
unrepresentable) on template families with a canonical call syntax
(gemma-4); other families run text-only.

`array = MtlArray` (with Metal.jl loaded — any GPUArrays vendor package
works) keeps a device twin of the weights for prefill, where batch puts
the GPU ~4× ahead, while generation stays on the host mmap-backed model,
where launch latency makes the CPU the fast device.

**Choosing a model:** Wink is a tool-driven agent — every future development
direction assumes tool calling. Select models on their tool-calling ability
first; a model that chats beautifully but cannot drive tools is decorative
here. (This is guidance, not enforced in code.)

Loading coordinates the compaction ladder with the model's real ceiling:
`CONFIG.context_budget` is lowered to 75% of `n_ctx` when it sits above
that. `local_model!(nothing)` unloads and returns routing to the
configured provider.
"""
function local_model!(path::AbstractString; n_ctx::Integer = 16_384,
        array = nothing)
    local_model!(nothing)
    p = expanduser(path)
    isfile(p) || error("model not found: $p")
    f = GGUF.GGUFFile(p)
    arch = String(GGUF.metadata(f, "general.architecture"))
    engine = _local_engine(arch)
    engine === nothing &&
        error("local backend has no $arch forward pass (have: " *
              join(sort(collect(keys(LOCAL_ENGINES))), ", ") * ")")
    tokmodel = String(GGUF.metadata(f, "tokenizer.ggml.model", "llama"))
    # "gemma4" is SPM with gemma-4's vocab conventions — the pure SPM
    # tokenizer handles it (oracle-verified on the 26B QAT file)
    tokmod = tokmodel in ("llama", "gemma4") ? SPMTokenizer :
             tokmodel == "gpt2" ? BPETokenizer :
             error("local backend has no \"$tokmodel\" tokenizer (llama/gpt2)")
    tok = tokmod.Tokenizer(f)
    m = engine.load_model(f)
    gpu = array === nothing ? nothing : Adapt.adapt(array, m)
    family = template_family(get(f.meta, "tokenizer.chat_template", nothing))
    family isa ChatMLTemplate &&
        @warn "unrecognized chat template family; ChatML fallback will degrade output"
    eog = unique(filter(>=(0), [tok.eos,
        get(tok.id_of, "<eos>", -1), get(tok.id_of, "<end_of_turn>", -1),
        get(tok.id_of, "<eot>", -1), get(tok.id_of, "<|endoftext|>", -1)]))
    LOCAL_MODEL[] = LocalModel(p, engine, m, gpu, tokmod, tok, family,
        engine.KVCache(m; capacity = Int(n_ctx)), "", Int(n_ctx), eog)
    CONFIG.chat_model = "local:" * basename(p)
    budget = _coordinate_budget!(n_ctx)
    printstyled(CONFIG.status_io,
        "  [Wink] local model loaded: ", basename(p),
        " ($arch, n_ctx = $n_ctx, context_budget = $budget, ",
        "template: $(nameof(typeof(family)))",
        gpu === nothing ? "" : ", GPU prefill", ")\n";
        color = :light_black)
    return CONFIG.chat_model
end

local_model!(::Nothing) = (LOCAL_MODEL[] = nothing)   # GC handles the memory

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
function _prefill!(lm::LocalModel, toks::Vector{Int})
    if lm.gpu_model !== nothing && length(toks) >= GPU_PREFILL_MIN
        gc = lm.engine.KVCache(lm.gpu_model;
            capacity = lm.cache.n + length(toks))
        _copy_cache!(gc, lm.cache, lm.cache.n)
        gc.n = lm.cache.n
        local logits
        for lo in 1:GPU_PREFILL_CHUNK:length(toks)
            logits = lm.engine.step!(lm.gpu_model, gc,
                toks[lo:min(lo + GPU_PREFILL_CHUNK - 1, end)])
            yield()               # keep the thinking spinner alive
        end
        _copy_cache!(lm.cache, gc, gc.n)
        lm.cache.n = gc.n
        return reshape(Array(logits[:, end]), :, 1)
    end
    logits = lm.engine.step!(lm.model, lm.cache, toks)
    return reshape(collect(Float32, view(logits, :, size(logits, 2))), :, 1)
end

# ---- sampling -----------------------------------------------------------------

# temperature + top-k sampling over one logits column (host-side; we own
# the sampler loop, which is what makes constrained tool calls possible).
# Single pass with a small insertion-sorted top-k — partialsortperm would
# allocate a full index permutation of the 262k vocab per token.
function _local_sample(logits::AbstractVector{Float32}; temp::Float32 = 0.7f0,
        top_k::Int = 40)
    temp <= 0 && return Int(argmax(logits)) - 1
    k = min(top_k, length(logits))
    idx = Vector{Int}(undef, k)
    val = fill(-Inf32, k)
    @inbounds for i in eachindex(logits)
        x = logits[i]
        x <= val[k] && continue
        j = k
        while j > 1 && val[j - 1] < x
            val[j] = val[j - 1]
            idx[j] = idx[j - 1]
            j -= 1
        end
        val[j] = x
        idx[j] = i
    end
    w = exp.((val .- val[1]) ./ temp)
    w ./= sum(w)
    r = rand(Float32)
    acc = 0.0f0
    @inbounds for j in 1:k
        acc += w[j]
        acc >= r && return idx[j] - 1
    end
    return idx[k] - 1
end

# ---- Wink-native history → renderer messages ----------------------------------

_string_keys(d::AbstractDict) = Dict{String, Any}(String(k) => v for (k, v) in d)

_local_msg(m::PT.SystemMessage) =
    Dict{String, Any}("role" => "system", "content" => m.content)
_local_msg(m::PT.UserMessage) =
    Dict{String, Any}("role" => "user", "content" => m.content)
_local_msg(m::PT.AIMessage) =
    Dict{String, Any}("role" => "assistant", "content" => something(m.content, ""))
# A call-bearing turn must render with empty content: under the canonical
# template, content alongside tool calls CLOSES the model turn, and after the
# responses no generation prompt follows — the model would have nowhere to
# continue. The preface text instead rides the `reasoning` field, which the
# renderer emits as the thought channel before the calls — the canonical home
# for pre-call narration, and the template's own reasoning guard drops it
# automatically once the conversation moves past the next user turn.
_local_msg(m::PT.AIToolRequest) =
    Dict{String, Any}("role" => "assistant",
        "content" => isempty(m.tool_calls) ? something(m.content, "") : "",
        "reasoning" => isempty(m.tool_calls) ? "" : something(m.content, ""),
        "tool_calls" => [Dict{String, Any}(
            "id" => c.tool_call_id,
            "function" => Dict{String, Any}(
                "name" => something(c.name, "unknown"),
                "arguments" => _string_keys(something(c.args, Dict{Symbol, Any}()))))
                         for c in m.tool_calls])
_local_msg(m::PT.ToolMessage) =
    Dict{String, Any}("role" => "tool", "tool_call_id" => m.tool_call_id,
        "name" => something(m.name, "unknown"),
        "content" => string(something(m.content, "")))
_local_msg(m) = Dict{String, Any}("role" => "user", "content" => string(m.content))

_tool_schema(t) = Dict{String, Any}("function" => Dict{String, Any}(
    "name" => t.name,
    "description" => something(t.description, ""),
    "parameters" => t.parameters))

# ---- the seam implementation --------------------------------------------------

function _local_aitools(lm::LocalModel, history, tools)
    t0 = time()
    with_tools = !isempty(tools) && lm.family isa Gemma4Template
    isempty(tools) || with_tools || debug_status(CONFIG.status_io,
        "$(nameof(typeof(lm.family))) has no canonical tool-call syntax; " *
        "$(length(tools)) tools not offered to the model")
    msgs = [_local_msg(m) for m in history]
    schemas = with_tools ? [_tool_schema(t) for t in tools] : Dict{String, Any}[]
    base = render_chat(lm.family, msgs; tools = schemas, add_assistant = false)
    full = render_chat(lm.family, msgs; tools = schemas, add_assistant = true)
    tokd(txt, add) = lm.tokmod.tokenize(lm.tok, String(txt);
        add_special = add, parse_special = true)
    delta = if !isempty(lm.kv_text) && startswith(base, lm.kv_text)
        tokd(SubString(base, ncodeunits(lm.kv_text) + 1), false)
    else
        lm.cache.n = 0          # splice (compaction, reset): rebuild the cache
        tokd(base, true)
    end
    suffix = tokd(SubString(full, ncodeunits(base) + 1), false)
    n0 = lm.cache.n
    needed = n0 + length(delta) + length(suffix) + 16
    needed <= lm.n_ctx ||
        error("context window full ($needed tokens needed, n_ctx = " *
              "$(lm.n_ctx)): run :compact or Wink.reset!(), or reload with " *
              "local_model!(path; n_ctx = ...)")
    col = _prefill!(lm, vcat(delta, suffix))
    lm.kv_text = base
    base_n = n0 + length(delta)
    suffix_base = lm.cache.n

    matchers = with_tools ? compile_matchers(schemas) : ToolMatcher[]
    banned = with_tools ? filter(>=(0),
        [get(lm.tok.id_of, "<|tool_response>", -1)]) : Int[]

    function generate_once!(col)
        out = IOBuffer()
        n_gen = 0
        tail = ""
        call_bytes = UInt8[]
        in_call = false
        while n_gen < CONFIG.max_tokens && lm.cache.n < lm.n_ctx - 4
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
                    next = _local_sample(logits)
                    s = lm.tokmod.piece(lm.tok, next; special = true)
                    st = isempty(s) ? :invalid :
                         call_state(matchers, vcat(call_bytes, codeunits(s)))
                    st === :invalid || break
                    logits[next + 1] = -Inf32
                end
                append!(call_bytes, codeunits(s))
            else
                next = _local_sample(logits)
                next in lm.eog && break
                s = lm.tokmod.piece(lm.tok, next; special = true)
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
            step = lm.engine.step!(lm.model, lm.cache, [next])
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
        lm.cache.n = suffix_base
        text, n_gen = generate_once!(col)
    end

    total = lm.cache.n
    lm.cache.n = base_n          # roll the scratch back to the canonical render
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
