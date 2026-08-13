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
    end
    return length(toks)
end

# ---- lifecycle ----------------------------------------------------------------

_default_libllama() = normpath(joinpath(@__DIR__, "..", "..", "spike", "vendor",
    "llama-b10405", "libllama.dylib"))

"""
    local_model!(path::AbstractString; n_ctx = 16_384, n_gpu_layers = 99)

Load a GGUF model INTO this Julia process (llama.cpp via LibLlama) and route
all of Wink's chat through it — no server, no HTTP. The model's chat template
is read from its own metadata and rendered by the matching `render_chat`
family; tool calls are grammar-constrained to be well-formed by construction.
The `libllama` library resolves from `ENV["WINK_LIBLLAMA"]` or the vendored
release under `spike/vendor/`.

`local_model!(nothing)` unloads and returns routing to the configured
provider.
"""
function local_model!(path::AbstractString; n_ctx::Integer = 16_384,
        n_gpu_layers::Integer = 99)
    local_model!(nothing)
    if isempty(LibLlama.libllama)
        LibLlama.set_lib!(get(ENV, "WINK_LIBLLAMA", _default_libllama()))
    end
    isfile(LibLlama.libllama) ||
        error("libllama not found at $(LibLlama.libllama) — set " *
              "ENV[\"WINK_LIBLLAMA\"] to a current llama.cpp dylib")
    p = expanduser(path)
    isfile(p) || error("model not found: $p")

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
    if !LOCAL_ATEXIT[]
        # free the Metal context before C++ static destructors run, or ggml's
        # device teardown asserts on the way out
        atexit(() -> local_model!(nothing))
        LOCAL_ATEXIT[] = true
    end
    CONFIG.chat_model = "local:" * basename(p)
    printstyled(CONFIG.status_io,
        "  [Wink] local model loaded in-process: ", basename(p),
        " (n_ctx = $n_ctx, template: $(nameof(typeof(family))))\n";
        color = :light_black)
    return CONFIG.chat_model
end

function local_model!(::Nothing)
    lm = LOCAL_MODEL[]
    lm === nothing && return nothing
    LOCAL_MODEL[] = nothing
    L.llama_free(lm.ctx)
    L.llama_model_free(lm.model)
    return nothing
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
# continue. With empty content the turn stays open across the tool cycle.
# The preface text is still stored on the AIToolRequest for transcripts.
_local_msg(m::PT.AIToolRequest) =
    Dict{String, Any}("role" => "assistant",
        "content" => isempty(m.tool_calls) ? something(m.content, "") : "",
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

function _local_chain(vocab; grammar = nothing)
    chain = L.llama_sampler_chain_init(L.llama_sampler_chain_default_params())
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
        _lc_decode!(lm.ctx, toks)
        lm.kv_tokens += length(toks)
        lm.kv_text = base

        # scratch region: generation-prompt suffix, then sampled tokens
        suffix = SubString(full, ncodeunits(base) + 1)
        stoks = isempty(suffix) ? Int32[] :
                _lc_tokenize(lm.vocab, suffix; add_special = false)
        scratch += _lc_decode!(lm.ctx, stoks)

        chain = _local_chain(lm.vocab;
            grammar = isempty(schemas) ? nothing : tool_call_grammar(schemas))
        out = IOBuffer()
        next = Int32[0]
        tail = ""
        try
            while n_gen < CONFIG.max_tokens
                tok = L.llama_sampler_sample(chain, lm.ctx, Int32(-1))
                L.llama_vocab_is_eog(lm.vocab, tok) && break
                s = _lc_piece(lm.vocab, tok)
                print(out, s)
                tail = last(tail * s, 16)
                n_gen += 1
                next[1] = tok
                scratch += _lc_decode!(lm.ctx, next)
                endswith(tail, "<tool_call|>") && break   # call complete
            end
        finally
            L.llama_sampler_free(chain)
        end
        text = String(take!(out))
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
    # stored content matches its re-rendered form.
    call === nothing &&
        return PT.AIToolRequest(; content = String(g4_strip_thinking(text)),
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
