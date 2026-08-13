# The in-process spike loop again, this time through the Clang.jl-generated
# bindings (LibLlama.jl) instead of hand-written ccalls. Same model, same
# greedy-in-Julia sampling; agreement with inprocess.jl's output validates the
# generated layer end to end.
#
# Run: julia --project=spike spike/inprocess_gen.jl [path-to-gguf]

include(joinpath(@__DIR__, "LibLlama.jl"))
import .LibLlama as L

const MODEL = get(ARGS, 1, expanduser(
    "~/.lmstudio/models/lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF/" *
    "NVIDIA-Nemotron-3-Nano-4B-Q8_0.gguf"))
isfile(MODEL) || error("model not found at $MODEL")

# Overwrite one field of an isbits struct held in a Ref (the C-style
# "defaults, then tweak" idiom without a 37-argument constructor).
function poke!(r::Ref{T}, name::Symbol, val) where {T}
    i = Base.fieldindex(T, name)
    FT = fieldtype(T, i)
    GC.@preserve r begin
        p = Base.unsafe_convert(Ptr{T}, r)
        unsafe_store!(Ptr{FT}(Ptr{UInt8}(p) + fieldoffset(T, i)), convert(FT, val))
    end
    return r
end

L.llama_backend_init()

mp = Ref(L.llama_model_default_params())
poke!(mp, :n_gpu_layers, Int32(99))

print("loading $(basename(MODEL)) … ")
t0 = time()
model = L.llama_model_load_from_file(MODEL, mp[])
model == C_NULL && error("model load failed")
println("done ($(round(time() - t0; digits = 1))s)")

vocab = L.llama_model_get_vocab(model)
n_vocab = L.llama_vocab_n_tokens(vocab)

cp = Ref(L.llama_context_default_params())
poke!(cp, :n_ctx, UInt32(4096))
poke!(cp, :n_batch, UInt32(512))
ctx = L.llama_init_from_model(model, cp[])
ctx == C_NULL && error("context init failed")
L.llama_n_ctx(ctx) == 4096 || error("layout mismatch on readback")
println("context ok (n_ctx = 4096, n_vocab = $n_vocab)")

# The model ships its chat template in GGUF metadata — the piece the backend
# will use to format conversations natively.
tmpl = L.llama_model_chat_template(model, C_NULL)
println("chat template in metadata: ", tmpl == C_NULL ? "(none)" :
        "yes ($(length(unsafe_string(tmpl))) chars)")

piece(tok) = begin
    buf = Vector{UInt8}(undef, 256)
    len = L.llama_token_to_piece(vocab, tok, buf, 256, 0, true)
    String(buf[1:max(len, 0)])
end

decode!(tokens::Vector{Int32}) = GC.@preserve tokens begin
    L.llama_decode(ctx, L.llama_batch_get_one(pointer(tokens), length(tokens))) == 0 ||
        error("llama_decode failed")
end

prompt = "The capital of France is"
toks = Vector{Int32}(undef, 512)
n = L.llama_tokenize(vocab, prompt, ncodeunits(prompt), toks, length(toks), true, true)
n > 0 || error("tokenize failed")
resize!(toks, n)
decode!(toks)

out = IOBuffer()
next = Int32[0]
n_gen = 0
t0 = time()
for _ in 1:32
    logits = L.llama_get_logits_ith(ctx, Int32(-1))
    v = unsafe_wrap(Array, logits, Int(n_vocab))
    tok = Int32(argmax(v) - 1)
    L.llama_vocab_is_eog(vocab, tok) && break
    print(out, piece(tok))
    global n_gen += 1
    next[1] = tok
    decode!(next)
end
dt = time() - t0

println("completion: ", repr(String(take!(out))))
println("generated $n_gen tokens in $(round(dt; digits = 2))s → ",
    round(n_gen / dt; digits = 1), " tok/s (generated bindings)")

L.llama_free(ctx)
L.llama_model_free(model)
L.llama_backend_free()
println("clean shutdown ok")
