# The pure-Julia backend: Wink's DEFAULT local model route. No C, no vendored
# binaries, no servers — GGUF reader, SPM tokenizer, and the generic-array
# forward pass, all Julia, portable to any GPUArrays backend via the `array`
# keyword (e.g. `local_model!(path; array = MtlArray)` with Metal.jl loaded).
#
# The KV strategy is the one proven on the llama.cpp backend — the cache
# holds the canonical history render (add_assistant = false), each call
# appends the canonical delta, and the generation scratch is rolled back —
# except that rollback here is one integer assignment: `cache.n = base`.
#
# v1 scope: TEXT-ONLY. Grammar-constrained tool calling was a llama.cpp
# sampler; its pure replacement (a GBNF-constrained sampler over our logits)
# is part of qualifying this backend, after which the llama.cpp backend is
# removed entirely.

mutable struct PureModel
    path::String
    model::Gemma3.Model
    tok::SPMTokenizer.Tokenizer
    family::ChatTemplate
    cache::Gemma3.KVCache
    kv_text::String
    n_ctx::Int
    eog::Vector{Int}
end

const PURE_MODEL = Ref{Union{Nothing, PureModel}}(nothing)

function _load_pure!(path::AbstractString; n_ctx::Integer = 16_384,
        array = nothing)
    p = expanduser(path)
    isfile(p) || error("model not found: $p")
    f = GGUF.GGUFFile(p)
    tok = SPMTokenizer.Tokenizer(f)
    m = Gemma3.load_model(f)
    array === nothing || (m = Adapt.adapt(array, m))
    family = template_family(get(f.meta, "tokenizer.chat_template", nothing))
    family isa ChatMLTemplate &&
        @warn "unrecognized chat template family; ChatML fallback will degrade output"
    eog = unique(filter(>=(0), [tok.eos,
        get(tok.id_of, "<eos>", -1), get(tok.id_of, "<end_of_turn>", -1),
        get(tok.id_of, "<eot>", -1)]))
    PURE_MODEL[] = PureModel(p, m, tok, family,
        Gemma3.KVCache(m; capacity = Int(n_ctx)), "", Int(n_ctx), eog)
    CONFIG.chat_model = "pure:" * basename(p)
    budget = _coordinate_budget!(n_ctx)
    printstyled(CONFIG.status_io,
        "  [Wink] pure-Julia model loaded: ", basename(p),
        " (n_ctx = $n_ctx, context_budget = $budget, ",
        "template: $(nameof(typeof(family))), text-only backend)\n";
        color = :light_black)
    return CONFIG.chat_model
end

# temperature + top-k sampling over one logits column (on the CPU: one small
# copy per token, negligible at these scales)
function _pure_sample(logits::Vector{Float32}; temp::Float32 = 0.7f0,
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

function _pure_aitools(pm::PureModel, history, tools)
    t0 = time()
    isempty(tools) || debug_status(CONFIG.status_io,
        "pure backend is text-only for now; $(length(tools)) tools not " *
        "offered to the model")
    msgs = [_local_msg(m) for m in history]
    base = render_chat(pm.family, msgs; add_assistant = false)
    full = render_chat(pm.family, msgs; add_assistant = true)
    tokd(txt, add) = SPMTokenizer.tokenize(pm.tok, String(txt);
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
    logits = Gemma3.step!(pm.model, pm.cache, vcat(delta, suffix))
    pm.kv_text = base
    base_n = n0 + length(delta)
    out = IOBuffer()
    n_gen = 0
    col = Array(view(logits, :, size(logits, 2)))
    while n_gen < CONFIG.max_tokens && pm.cache.n < pm.n_ctx - 4
        next = _pure_sample(col)
        next in pm.eog && break
        print(out, SPMTokenizer.piece(pm.tok, next; special = false))
        n_gen += 1
        logits = Gemma3.step!(pm.model, pm.cache, [next])
        col = Array(view(logits, :, 1))
        yield()                  # keep the thinking spinner alive
    end
    total = pm.cache.n
    pm.cache.n = base_n          # roll the scratch back to the canonical render
    return PT.AIToolRequest(; content = String(strip(String(take!(out)))),
        tool_calls = PT.ToolMessage[], tokens = (total, n_gen),
        elapsed = time() - t0)
end
