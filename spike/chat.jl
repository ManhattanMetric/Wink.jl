# Chat spike: the model's own chat template plus a real sampler chain, with
# KV-cache continuation across turns — the conversation mechanics the
# LlamaCppBackend needs, proven in-process.
#
# Template: read from GGUF metadata (llama_model_chat_template) and rendered
# by llama.cpp's C-side matcher (llama_chat_apply_template), which recognizes
# common template families from their Jinja source. If it doesn't know this
# model's family it returns -1 and we fall back to ChatML — reported loudly,
# since backend correctness depends on which path ran.
#
# Continuation: llama.cpp's simple-chat pattern. After each assistant turn,
# re-render the conversation WITHOUT the assistant prompt and remember the
# rendered length; the next turn renders the full conversation and decodes
# only the bytes past that mark, so the KV cache is never re-fed. The sampled
# end-of-generation token is decoded into the cache before breaking, keeping
# it aligned with the template's rendered close marker (modulo whitespace).
#
# Run: julia --project=spike spike/chat.jl [path-to-gguf]

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "render.jl"))

const MODEL = get(ARGS, 1, expanduser(
    "~/.lmstudio/models/lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF/" *
    "NVIDIA-Nemotron-3-Nano-4B-Q8_0.gguf"))

L.llama_backend_init()
print("loading $(basename(MODEL)) … ")
model = load_model(MODEL)
vocab = L.llama_model_get_vocab(model)
ctx = new_context(model; n_ctx = 8192)
println("ok")

# ---- template rendering -------------------------------------------------------

tmpl_ptr = L.llama_model_chat_template(model, C_NULL)
const TEMPLATE = tmpl_ptr == C_NULL ? nothing : unsafe_string(tmpl_ptr)

# Probe: can llama.cpp's C matcher render this template at all?
function native_probe()
    TEMPLATE === nothing && return false
    role, content = "user", "probe"
    buf = Vector{UInt8}(undef, 512)
    GC.@preserve role content begin
        cmsg = [L.llama_chat_message(Ptr{Cchar}(pointer(role)),
            Ptr{Cchar}(pointer(content)))]
        return L.llama_chat_apply_template(TEMPLATE, cmsg, 1, true,
            buf, length(buf)) >= 0
    end
end

const FAMILY = template_family(TEMPLATE; native_probe)
println("template: ", TEMPLATE === nothing ? "(none in metadata)" :
        "$(length(TEMPLATE)) chars in metadata", " — family: ",
    nameof(typeof(FAMILY)),
    FAMILY isa ChatMLTemplate ? " (LAST-RESORT FALLBACK — output will be degraded)" : "")

render(msgs; add_assistant) = render_chat(FAMILY, msgs; add_assistant)

# ---- sampler chain ------------------------------------------------------------

chain = L.llama_sampler_chain_init(L.llama_sampler_chain_default_params())
L.llama_sampler_chain_add(chain, L.llama_sampler_init_top_k(Int32(40)))
L.llama_sampler_chain_add(chain, L.llama_sampler_init_top_p(0.95f0, 1))
L.llama_sampler_chain_add(chain, L.llama_sampler_init_temp(0.7f0))
L.llama_sampler_chain_add(chain, L.llama_sampler_init_dist(UInt32(42)))

function generate!(ctx, vocab, chain; max_tokens = 512)
    out = IOBuffer()
    next = Int32[0]
    n = 0
    t0 = time()
    while n < max_tokens
        tok = L.llama_sampler_sample(chain, ctx, Int32(-1))
        next[1] = tok
        if L.llama_vocab_is_eog(vocab, tok)
            decode!(ctx, next)   # keep the close marker in the KV cache
            break
        end
        print(out, piece(vocab, tok))
        n += 1
        decode!(ctx, next)
    end
    return (text = String(take!(out)), tokens = n, dt = time() - t0)
end

# ---- two turns ----------------------------------------------------------------

msgs = [(role = "system",
         content = "You are Wink, a terse assistant living inside a Julia REPL."),
        (role = "user",
         content = "In one sentence: what makes multiple dispatch powerful?")]

prompt = render(msgs; add_assistant = true)
toks = tokenize(vocab, prompt; add_special = true)
println("\n[turn 1] prompt: $(length(toks)) tokens")
decode!(ctx, toks)
r1 = generate!(ctx, vocab, chain)
println("assistant: ", strip(r1.text))
println("[turn 1] $(r1.tokens) tokens in $(round(r1.dt; digits = 2))s → ",
    round(r1.tokens / r1.dt; digits = 1), " tok/s")

push!(msgs, (role = "assistant", content = strip(r1.text)))
prev_len = ncodeunits(render(msgs; add_assistant = false))

push!(msgs, (role = "user",
    content = "Show it in two lines of Julia code, nothing else."))
full = render(msgs; add_assistant = true)
delta = String(codeunits(full)[(prev_len + 1):end])
toks2 = tokenize(vocab, delta; add_special = false)
println("\n[turn 2] delta: $(length(toks2)) tokens (full conversation would be ",
    length(tokenize(vocab, full; add_special = true)), ") — KV cache carried over")
decode!(ctx, toks2)
r2 = generate!(ctx, vocab, chain)
println("assistant: ", strip(r2.text))
println("[turn 2] $(r2.tokens) tokens in $(round(r2.dt; digits = 2))s → ",
    round(r2.tokens / r2.dt; digits = 1), " tok/s")

L.llama_sampler_free(chain)
L.llama_free(ctx)
L.llama_model_free(model)
L.llama_backend_free()
println("\nclean shutdown ok")
