# The oracle run: pure-Julia Gemma-3 vs llama.cpp on the SAME GGUF file,
# compared position-by-position at the logits. The in-process llama.cpp from
# the local-inference work is the ground truth; llama.cpp also lends its
# tokenizer (pure-Julia SentencePiece is the next spike).
#
# Run: julia --project=spike pure/run.jl

include(joinpath(@__DIR__, "..", "spike", "LibLlama.jl"))
import .LibLlama as L
include(joinpath(@__DIR__, "gguf.jl"))
using .GGUF
include(joinpath(@__DIR__, "gemma3.jl"))
using .Gemma3
using LinearAlgebra

const MODEL = joinpath(@__DIR__, "models", "gemma-3-270m-it-F16.gguf")
const PROMPT = "The capital of France is"

# ---- llama.cpp side: tokenizer + per-position oracle logits -------------------

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
lmodel = L.llama_model_load_from_file(MODEL, L.llama_model_default_params())
lmodel == C_NULL && error("llama.cpp failed to load $MODEL")
vocab = L.llama_model_get_vocab(lmodel)
n_vocab = Int(L.llama_vocab_n_tokens(vocab))
cp = Ref(L.llama_context_default_params())
poke!(cp, :n_ctx, UInt32(512))
ctx = L.llama_init_from_model(lmodel, cp[])

buf = Vector{Int32}(undef, 512)
n = L.llama_tokenize(vocab, PROMPT, ncodeunits(PROMPT), buf, length(buf), true, true)
toks = Int.(buf[1:n])
piece(t) = (b = Vector{UInt8}(undef, 64);
    len = L.llama_token_to_piece(vocab, Int32(t), b, 64, 0, true);
    String(b[1:max(len, 0)]))
println("prompt: ", repr(PROMPT), " → ", length(toks), " tokens ", toks)

function oracle_logits(toks)
    out = Matrix{Float32}(undef, n_vocab, length(toks))
    one = Int32[0]
    for (i, t) in enumerate(toks)
        one[1] = t
        GC.@preserve one begin
            L.llama_decode(ctx, L.llama_batch_get_one(pointer(one), 1)) == 0 ||
                error("oracle decode failed")
        end
        out[:, i] = unsafe_wrap(Array, L.llama_get_logits_ith(ctx, Int32(-1)), n_vocab)
    end
    return out
end
ora = oracle_logits(toks)

# ---- pure-Julia side ----------------------------------------------------------

println("loading weights through the pure reader …")
t0 = time()
m = Gemma3.load_model(GGUFFile(MODEL))
println("loaded in ", round(time() - t0; digits = 1), "s")
t0 = time()
ours = Gemma3.forward(m, toks)
println("pure forward over $(length(toks)) positions: ",
    round(time() - t0; digits = 2), "s")

# ---- comparison ---------------------------------------------------------------

agree = 0
for i in eachindex(toks)
    a = argmax(view(ours, :, i))
    b = argmax(view(ora, :, i))
    ok = a == b
    global agree += ok
    println("pos $i: pure→", rpad(repr(piece(a - 1)), 14),
        " llama.cpp→", rpad(repr(piece(b - 1)), 14), ok ? " MATCH" : "  ✗")
end
cs = dot(ours[:, end], ora[:, end]) / (norm(ours[:, end]) * norm(ora[:, end]))
println("top-1 agreement: $agree/$(length(toks))   ",
    "final-position logit cosine: ", round(cs; digits = 6),
    "   max|Δ|: ", round(maximum(abs.(ours[:, end] .- ora[:, end])); digits = 3))

# ---- generation face-off ------------------------------------------------------

eog = [t for t in 0:(n_vocab - 1) if L.llama_vocab_is_eog(vocab, Int32(t))]
gen = Gemma3.generate(m, toks; max_tokens = 16, eog)
println("pure Julia continuation:  ", repr(join(piece.(gen))))

# llama.cpp greedy continuation from the same state (ctx already holds prompt)
lgen = Int[]
one = Int32[0]
for _ in 1:16
    lg = unsafe_wrap(Array, L.llama_get_logits_ith(ctx, Int32(-1)), n_vocab)
    nt = argmax(lg) - 1
    L.llama_vocab_is_eog(vocab, Int32(nt)) && break
    push!(lgen, nt)
    one[1] = nt
    GC.@preserve one L.llama_decode(ctx, L.llama_batch_get_one(pointer(one), 1))
end
println("llama.cpp continuation:   ", repr(join(piece.(lgen))))
println(gen == lgen ? "IDENTICAL GREEDY CONTINUATIONS" :
        "continuations diverge at token $(findfirst(i -> i > length(lgen) || gen[i] != lgen[i], eachindex(gen)))")

L.llama_free(ctx)
L.llama_model_free(lmodel)
L.llama_backend_free()
