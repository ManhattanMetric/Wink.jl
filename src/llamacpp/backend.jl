# The in-process llama.cpp backend: the model loaded into the same Julia
# process as the session it pair-programs in. Activated by `local_model!`,
# it takes over `_aitools_call`'s default route (explicit schemas — tests —
# still go to PromptingTools), so the whole agent loop, compaction included,
# runs against local weights with no server and no HTTP.
#
# Conversation state: the KV cache holds exactly the CANONICAL rendering of
# the history (render_chat with add_assistant = false). Each call appends the
# canonical delta (splices — compaction — break the prefix and trigger a full
# re-decode), then decodes the generation-prompt suffix and the sampled
# tokens as a scratch region that is rolled back (llama_memory_seq_rm) after
# generation. The cache therefore always re-aligns with what the next render
# will produce, keeping multi-turn decoding incremental.
#
# Tool calls are grammar-constrained: when tools are present, a lazy GBNF
# grammar (tool_call_grammar) leaves the model free until it emits
# "<|tool_call>", then clamps everything to a well-formed call — the
# malformed-call failure class is unrepresentable here.

mutable struct LocalModel
    path::String
    model::Ptr{L.llama_model}
    vocab::Ptr{L.llama_vocab}
    ctx::Ptr{L.llama_context}
    family::ChatTemplate
    n_ctx::Int
    kv_text::String
    kv_tokens::Int
end

const LOCAL_MODEL = Ref{Union{Nothing, LocalModel}}(nothing)
const LOCAL_CALL_COUNTER = Ref(0)
const LOCAL_SEED = Ref{UInt32}(0x57494e4b)
const LOCAL_ATEXIT = Ref(false)

# ---- low-level helpers (proven in the spike) ----------------------------------

function _poke!(r::Ref{T}, name::Symbol, val) where {T}
    i = Base.fieldindex(T, name)
    FT = fieldtype(T, i)
    GC.@preserve r begin
        p = Base.unsafe_convert(Ptr{T}, r)
        unsafe_store!(Ptr{FT}(Ptr{UInt8}(p) + fieldoffset(T, i)), convert(FT, val))
    end
    return r
end

function _lc_tokenize(vocab, text::AbstractString; add_special::Bool)
    buf = Vector{Int32}(undef, ncodeunits(text) + 32)
    n = L.llama_tokenize(vocab, text, ncodeunits(text), buf, length(buf),
        add_special, true)
    if n < 0
        resize!(buf, -n)
        n = L.llama_tokenize(vocab, text, ncodeunits(text), buf, length(buf),
            add_special, true)
    end
    n < 0 && error("llama_tokenize failed")
    return resize!(buf, n)
end

function _lc_piece(vocab, tok::Integer)
    buf = Vector{UInt8}(undef, 256)
    len = L.llama_token_to_piece(vocab, tok, buf, length(buf), 0, true)
    return String(buf[1:max(len, 0)])
end

# Refuse to slam into the context wall at the C level: a friendly error the
# user can act on beats "failed to find a memory slot".
function _room_check(lm, needed::Integer)
    needed <= lm.n_ctx - 32 && return nothing
    error("context window full ($needed tokens needed, n_ctx = $(lm.n_ctx)): " *
          "run :compact or Wink.reset!(), or reload with a larger context " *
          "via local_model!(path; n_ctx = ...). If this recurs, check that " *
          "CONFIG.context_budget sits well below n_ctx so compaction fires first.")
end

# Decode in n_batch-sized chunks (llama_decode rejects oversized batches).
function _lc_decode!(ctx, toks::Vector{Int32}; n_batch::Integer = 512)
    i = 1
    while i <= length(toks)
        chunk = toks[i:min(i + n_batch - 1, length(toks))]
        ret = GC.@preserve chunk L.llama_decode(ctx,
            L.llama_batch_get_one(pointer(chunk), length(chunk)))
        ret == 0 || error("llama_decode failed with $ret (context full? " *
                          "lower context_budget or raise n_ctx in local_model!)")
        i += length(chunk)
        # blocking ccalls starve Julia's cooperative scheduler: without this,
        # with_thinking's spinner task never gets a frame. Every prompt chunk
        # and every generated token passes through here, so this one yield
        # keeps the UI alive at both cadences.
        yield()
    end
    return length(toks)
end

# ---- logging ------------------------------------------------------------------
#
# llama.cpp logs straight to stderr by default — model-load inventories, Metal
# kernel compiles, and per-token grammar traces that are pure trace-level
# noise. We install a filtering callback: warnings and errors always show
# (out-of-memory matters), info shows under CONFIG.debug, and the per-token
# DEBUG firehose only with ENV["WINK_LLAMA_TRACE"]. CONT lines inherit the
# level of the message they continue.

const LLAMA_LOG_THRESHOLD = Ref{UInt32}(UInt32(L.GGML_LOG_LEVEL_WARN))
const LLAMA_LOG_LAST = Ref{UInt32}(0)

function _llama_log_cb(level::UInt32, text::Ptr{Cchar}, ::Ptr{Cvoid})::Cvoid
    lvl = level == UInt32(L.GGML_LOG_LEVEL_CONT) ? LLAMA_LOG_LAST[] : level
    LLAMA_LOG_LAST[] = lvl
    lvl >= LLAMA_LOG_THRESHOLD[] && print(stderr, unsafe_string(Ptr{UInt8}(text)))
    return nothing
end

function _install_log_filter!()
    LLAMA_LOG_THRESHOLD[] = haskey(ENV, "WINK_LLAMA_TRACE") ?
                            UInt32(L.GGML_LOG_LEVEL_DEBUG) :
                            CONFIG.debug ? UInt32(L.GGML_LOG_LEVEL_INFO) :
                            UInt32(L.GGML_LOG_LEVEL_WARN)
    cb = @cfunction(_llama_log_cb, Cvoid, (UInt32, Ptr{Cchar}, Ptr{Cvoid}))
    L.llama_log_set(cb, C_NULL)
    return nothing
end

# ---- lifecycle ----------------------------------------------------------------

_default_libllama() = normpath(joinpath(@__DIR__, "..", "..", "spike", "vendor",
    "llama-b10405", "libllama.dylib"))

# The global context_budget default assumes frontier-sized windows; a local
# model's real ceiling is n_ctx. Lower the budget (never raise it — explicit
# user settings win) so the fold/distill/mine ladder fires well before
# llama_decode hits the wall.
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
    local_model!(path::AbstractString; backend = :pure, n_ctx, ...)

Load a GGUF model INTO this Julia process and route all of Wink's chat
through it — no server, no HTTP.

`backend = :pure` (the default) is the pure-Julia backend: GGUF reader,
SentencePiece tokenizer, and generic-array forward pass, portable to any
GPUArrays backend via `array` (e.g. `array = MtlArray` with Metal.jl
loaded). Currently text-only. Defaults to `n_ctx = 16_384`.

`backend = :llamacpp` is PRE-DEPRECATED: it remains only until the pure
backend qualifies as the local path, and will then be REMOVED — do not
build on it. It requires a local llama.cpp dylib (`ENV["WINK_LIBLLAMA"]`
or the vendored release under `spike/vendor/`) and supports
grammar-constrained tool calls. Defaults to `n_ctx = 32_768`;
`n_gpu_layers` applies here only.

Loading coordinates the compaction ladder with the model's real ceiling:
`CONFIG.context_budget` is lowered to 75% of `n_ctx` when it sits above that.
`local_model!(nothing)` unloads any local backend and returns routing to the
configured provider.
"""
function local_model!(path::AbstractString; backend::Symbol = :pure,
        n_ctx::Integer = backend === :pure ? 16_384 : 32_768,
        n_gpu_layers::Integer = 99, array = nothing)
    local_model!(nothing)
    backend === :pure && return _load_pure!(path; n_ctx, array)
    backend === :llamacpp || error("unknown backend $backend (:pure or :llamacpp)")
    @warn "backend = :llamacpp is pre-deprecated: it will be removed once " *
          "the pure-Julia backend qualifies as the local path; do not build " *
          "on it" maxlog = 1
    return _load_llamacpp!(path; n_ctx, n_gpu_layers)
end

function _load_llamacpp!(path::AbstractString; n_ctx::Integer = 32_768,
        n_gpu_layers::Integer = 99)
    if isempty(LibLlama.libllama)
        LibLlama.set_lib!(get(ENV, "WINK_LIBLLAMA", _default_libllama()))
    end
    isfile(LibLlama.libllama) ||
        error("libllama not found at $(LibLlama.libllama) — set " *
              "ENV[\"WINK_LIBLLAMA\"] to a current llama.cpp dylib")
    p = expanduser(path)
    isfile(p) || error("model not found: $p")

    _install_log_filter!()
    L.llama_backend_init()
    mp = Ref(L.llama_model_default_params())
    _poke!(mp, :n_gpu_layers, Int32(n_gpu_layers))
    model = L.llama_model_load_from_file(p, mp[])
    model == C_NULL && error("failed to load $p")
    vocab = L.llama_model_get_vocab(model)
    cp = Ref(L.llama_context_default_params())
    _poke!(cp, :n_ctx, UInt32(n_ctx))
    _poke!(cp, :n_batch, UInt32(512))
    ctx = L.llama_init_from_model(model, cp[])
    ctx == C_NULL && (L.llama_model_free(model); error("failed to create context"))

    tmpl_ptr = L.llama_model_chat_template(model, C_NULL)
    template = tmpl_ptr == C_NULL ? nothing : unsafe_string(tmpl_ptr)
    family = template_family(template; native_probe = () -> begin
        role, content = "user", "probe"
        buf = Vector{UInt8}(undef, 512)
        GC.@preserve role content begin
            cmsg = [L.llama_chat_message(Ptr{Cchar}(pointer(role)),
                Ptr{Cchar}(pointer(content)))]
            L.llama_chat_apply_template(template, cmsg, 1, true, buf, length(buf)) >= 0
        end
    end)
    family isa Gemma4Template ||
        @warn "template family $(nameof(typeof(family))): tool calling is only " *
              "implemented for the Gemma-4 family; other models run text-only"

    LOCAL_MODEL[] = LocalModel(p, model, vocab, ctx, family, Int(n_ctx), "", 0)
    _register_local_atexit!()
    CONFIG.chat_model = "local:" * basename(p)
    budget = _coordinate_budget!(n_ctx)
    printstyled(CONFIG.status_io,
        "  [Wink] local model loaded in-process: ", basename(p),
        " (n_ctx = $n_ctx, context_budget = $budget, ",
        "template: $(nameof(typeof(family))))\n";
        color = :light_black)
    return CONFIG.chat_model
end

function local_model!(::Nothing)
    PURE_MODEL[] = nothing        # GC handles the pure backend's memory
    lm = LOCAL_MODEL[]
    lm === nothing && return nothing
    LOCAL_MODEL[] = nothing
    L.llama_free(lm.ctx)
    L.llama_model_free(lm.model)
    return nothing
end

# Free the Metal contexts before C++ static destructors run, or ggml's device
# teardown asserts on the way out.
function _register_local_atexit!()
    LOCAL_ATEXIT[] && return nothing
    atexit(() -> (local_model!(nothing); local_embed_model!(nothing)))
    LOCAL_ATEXIT[] = true
    return nothing
end

# ---- in-process embeddings ----------------------------------------------------
#
# reindex!/search_docs route their embedding traffic here when an embedder is
# loaded, completing the no-server story: chat AND retrieval in one process.

mutable struct LocalEmbedder
    path::String
    model::Ptr{L.llama_model}
    vocab::Ptr{L.llama_vocab}
    ctx::Ptr{L.llama_context}
    n_embd::Int
    n_batch::Int
    use_encode::Bool
end

const LOCAL_EMBED = Ref{Union{Nothing, LocalEmbedder}}(nothing)

"""
    local_embed_model!(path::AbstractString; n_ctx = 2_048, n_gpu_layers = 99)

Load an embedding GGUF (e.g. nomic-embed-text) INTO this Julia process and
route `reindex!`/`search_docs` embedding traffic through it — no embedding
server needed. Pooling comes from the model's own metadata. Sets
`CONFIG.embed_model` to a `local:` name so the doc index is cached per model.

`local_embed_model!(nothing)` unloads (dropping the in-memory doc index) and
clears `embed_model`, degrading doc search to keyword matching until an
embedding provider is configured again.
"""
function local_embed_model!(path::AbstractString; n_ctx::Integer = 2_048,
        n_gpu_layers::Integer = 99)
    local_embed_model!(nothing)
    if isempty(LibLlama.libllama)
        LibLlama.set_lib!(get(ENV, "WINK_LIBLLAMA", _default_libllama()))
    end
    isfile(LibLlama.libllama) ||
        error("libllama not found at $(LibLlama.libllama) — set " *
              "ENV[\"WINK_LIBLLAMA\"] to a current llama.cpp dylib")
    p = expanduser(path)
    isfile(p) || error("model not found: $p")

    _install_log_filter!()
    L.llama_backend_init()
    mp = Ref(L.llama_model_default_params())
    _poke!(mp, :n_gpu_layers, Int32(n_gpu_layers))
    model = L.llama_model_load_from_file(p, mp[])
    model == C_NULL && error("failed to load $p")
    vocab = L.llama_model_get_vocab(model)
    cp = Ref(L.llama_context_default_params())
    _poke!(cp, :n_ctx, UInt32(n_ctx))
    _poke!(cp, :n_batch, UInt32(n_ctx))
    # the encoder path asserts n_ubatch >= n_tokens: the whole sequence must
    # fit one PHYSICAL micro-batch, not just the logical batch
    _poke!(cp, :n_ubatch, UInt32(n_ctx))
    _poke!(cp, :embeddings, true)
    ctx = L.llama_init_from_model(model, cp[])
    ctx == C_NULL && (L.llama_model_free(model); error("failed to create context"))

    use_encode = L.llama_model_has_encoder(model) && !L.llama_model_has_decoder(model)
    LOCAL_EMBED[] = LocalEmbedder(p, model, vocab, ctx,
        Int(L.llama_model_n_embd(model)), Int(n_ctx), use_encode)
    _register_local_atexit!()
    CONFIG.embed_model = "local:" * basename(p)
    printstyled(CONFIG.status_io,
        "  [Wink] local embedder loaded in-process: ", basename(p),
        " ($(L.llama_model_n_embd(model)) dims)\n"; color = :light_black)
    return CONFIG.embed_model
end

function local_embed_model!(::Nothing)
    le = LOCAL_EMBED[]
    le === nothing && return nothing
    LOCAL_EMBED[] = nothing
    L.llama_free(le.ctx)
    L.llama_model_free(le.model)
    DOC_INDEX[] = nothing
    startswith(CONFIG.embed_model, "local:") && (CONFIG.embed_model = "")
    return nothing
end

"""
    _local_embed_texts(texts; io = CONFIG.status_io) -> Matrix{Float32}

Embed each text through the in-process embedder; unit-norm columns, matching
what the RAG index expects. Each text is embedded as its own sequence with a
cleared cache; over-length texts are token-truncated to the batch size.
"""
function _local_embed_texts(texts::Vector{String}; io::IO = CONFIG.status_io)
    le = LOCAL_EMBED[]
    le === nothing && error("no local embedder loaded (local_embed_model!)")
    mem = L.llama_get_memory(le.ctx)
    cols = Matrix{Float32}(undef, le.n_embd, length(texts))
    for (i, text) in enumerate(texts)
        toks = _lc_tokenize(le.vocab, text; add_special = true)
        length(toks) > le.n_batch && resize!(toks, le.n_batch)
        L.llama_memory_clear(mem, true)
        ret = GC.@preserve toks begin
            batch = L.llama_batch_get_one(pointer(toks), length(toks))
            le.use_encode ? L.llama_encode(le.ctx, batch) :
            L.llama_decode(le.ctx, batch)
        end
        ret == 0 || error("embedding decode failed with $ret")
        ptr = L.llama_get_embeddings_seq(le.ctx, 0)
        ptr == C_NULL && (ptr = L.llama_get_embeddings_ith(le.ctx, Int32(-1)))
        ptr == C_NULL && error("model returned no embeddings")
        v = unsafe_wrap(Array, ptr, le.n_embd)
        n = LinearAlgebra.norm(v)
        cols[:, i] .= n > 0 ? v ./ n : v
        i % 50 == 0 && status(io, "embedded $i/$(length(texts)) docstrings")
        yield()
    end
    return cols
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

# Tokens the model must never emit in Wink's flow. `<|tool_response>` is an
# EOG token (llama.cpp stops generation so a harness can supply results), but
# Wink always injects tool responses itself — when the model reflexively
# opens the next call→response cycle with it, generation dies with an empty
# reply. Ban it at the logits.
function _banned_tokens(vocab)
    banned = L.llama_logit_bias[]
    for marker in ("<|tool_response>",)
        toks = _lc_tokenize(vocab, marker; add_special = false)
        length(toks) == 1 &&
            push!(banned, L.llama_logit_bias(toks[1], -Inf32))
    end
    return banned
end

function _local_chain(vocab; grammar = nothing)
    chain = L.llama_sampler_chain_init(L.llama_sampler_chain_default_params())
    banned = _banned_tokens(vocab)
    isempty(banned) || GC.@preserve banned L.llama_sampler_chain_add(chain,
        L.llama_sampler_init_logit_bias(L.llama_vocab_n_tokens(vocab),
            length(banned), pointer(banned)))
    if grammar !== nothing
        trigger = "<|tool_call>"
        triggers = [trigger]
        ptrs = [Ptr{Cchar}(pointer(t)) for t in triggers]
        g = GC.@preserve triggers ptrs L.llama_sampler_init_grammar_lazy(
            vocab, grammar, "root", ptrs, length(ptrs), C_NULL, 0)
        g == C_NULL && error("grammar sampler rejected the tool grammar")
        L.llama_sampler_chain_add(chain, g)   # mask first, then sample
    end
    L.llama_sampler_chain_add(chain, L.llama_sampler_init_top_k(Int32(40)))
    L.llama_sampler_chain_add(chain, L.llama_sampler_init_top_p(0.95f0, 1))
    L.llama_sampler_chain_add(chain, L.llama_sampler_init_temp(0.7f0))
    L.llama_sampler_chain_add(chain,
        L.llama_sampler_init_dist(LOCAL_SEED[] += UInt32(1)))
    return chain
end

function _local_aitools(lm::LocalModel, history, tools)
    t0 = time()
    msgs = [_local_msg(m) for m in history]
    schemas = [_tool_schema(t) for t in tools]
    base = render_chat(lm.family, msgs; tools = schemas, add_assistant = false)
    full = render_chat(lm.family, msgs; tools = schemas, add_assistant = true)
    mem = L.llama_get_memory(lm.ctx)

    scratch = 0
    text = ""
    n_gen = 0
    try
        # canonical prefix: append the delta, or re-decode on a splice
        if !isempty(lm.kv_text) && startswith(base, lm.kv_text)
            delta = SubString(base, ncodeunits(lm.kv_text) + 1)
            toks = isempty(delta) ? Int32[] :
                   _lc_tokenize(lm.vocab, delta; add_special = false)
        else
            L.llama_memory_clear(mem, true)
            lm.kv_text = ""
            lm.kv_tokens = 0
            toks = _lc_tokenize(lm.vocab, base; add_special = true)
        end
        _room_check(lm, lm.kv_tokens + length(toks))
        _lc_decode!(lm.ctx, toks)
        lm.kv_tokens += length(toks)
        lm.kv_text = base

        # scratch region: generation-prompt suffix, then sampled tokens
        suffix = SubString(full, ncodeunits(base) + 1)
        stoks = isempty(suffix) ? Int32[] :
                _lc_tokenize(lm.vocab, suffix; add_special = false)
        _room_check(lm, lm.kv_tokens + length(stoks) + 16)
        scratch += _lc_decode!(lm.ctx, stoks)

        gen_grammar = isempty(schemas) ? nothing : tool_call_grammar(schemas)
        suffix_base = scratch
        function generate_once!()
            chain = _local_chain(lm.vocab; grammar = gen_grammar)
            out = IOBuffer()
            next = Int32[0]
            tail = ""
            call_opened = false
            g = 0
            try
                # cap generation to the room the window actually has left
                while g < CONFIG.max_tokens && lm.kv_tokens + scratch < lm.n_ctx - 8
                    tok = L.llama_sampler_sample(chain, lm.ctx, Int32(-1))
                    L.llama_vocab_is_eog(lm.vocab, tok) && break
                    s = _lc_piece(lm.vocab, tok)
                    print(out, s)
                    tail = last(tail * s, 24)
                    occursin("<|tool_call>", tail) && (call_opened = true)
                    g += 1
                    next[1] = tok
                    scratch += _lc_decode!(lm.ctx, next)
                    # a call is complete only when one was actually OPENED —
                    # the model can emit a stray close token with the grammar
                    # dormant, and breaking on it would swallow the reply
                    call_opened && endswith(tail, "<tool_call|>") && break
                end
            finally
                L.llama_sampler_free(chain)
            end
            return String(take!(out)), g
        end
        text, n_gen = generate_once!()
        if parse_tool_call(text) === nothing &&
           isempty(strip(g4_strip_thinking(text)))
            # an empty round (immediate EOG, or pure channel noise): rewind
            # whatever the dud generation decoded and redraw once with a
            # fresh sampler seed rather than ending the turn with silence
            if scratch > suffix_base
                L.llama_memory_seq_rm(mem, 0,
                    L.llama_pos(lm.kv_tokens + suffix_base), L.llama_pos(-1))
                scratch = suffix_base
            end
            text, n_gen = generate_once!()
        end
    catch
        # a failed decode leaves the cache in an unknown state: start clean
        L.llama_memory_clear(mem, true)
        lm.kv_text = ""
        lm.kv_tokens = 0
        rethrow()
    finally
        # roll the scratch region back so the cache matches the canonical
        # render the next call will produce
        scratch > 0 && LOCAL_MODEL[] === lm &&
            L.llama_memory_seq_rm(mem, 0, lm.kv_tokens, -1)
    end

    total = lm.kv_tokens + scratch
    call = parse_tool_call(text)
    # The model may open/close thought channels (it does after tool
    # responses); strip them like the renderer strips them from history, so
    # stored content matches its re-rendered form — and drop any stray tool
    # markers a dormant-grammar wobble left behind.
    call === nothing &&
        return PT.AIToolRequest(;
            content = String(strip(replace(g4_strip_thinking(text),
                "<tool_call|>" => "", "<|tool_response>" => ""))),
            tool_calls = PT.ToolMessage[], tokens = (total, n_gen),
            elapsed = time() - t0)
    idx = findfirst("<|tool_call>", text)
    preface = g4_strip_thinking(SubString(text, 1, first(idx) - 1))
    args = Dict{Symbol, Any}(Symbol(k) => v for (k, v) in call.args)
    tc = PT.ToolMessage(;
        tool_call_id = "local-" * string(LOCAL_CALL_COUNTER[] += 1),
        raw = JSON3.write(call.args), name = call.name, args)
    return PT.AIToolRequest(; content = isempty(preface) ? nothing : String(preface),
        tool_calls = [tc], tokens = (total, n_gen), elapsed = time() - t0)
end
