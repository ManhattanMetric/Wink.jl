# The summit test: the 26B-A4B daily driver, pure Julia vs llama.cpp on the
# QAT q4_0 file that has been on this disk since the local-inference branch.
# Tokenizer is the pure SPM one (gemma-4 is SentencePiece); llama.cpp is
# logits oracle only. Both engines mmap the same file, so RAM cost is shared.
#
# Run: julia -t 8 --project=spike pure/test_gemma4.jl

include(joinpath(@__DIR__, "..", "spike", "LibLlama.jl"))
import .LibLlama as L
include(joinpath(@__DIR__, "quant.jl"))
using .Quant
include(joinpath(@__DIR__, "gguf.jl"))
using .GGUF
include(joinpath(@__DIR__, "tokenizer.jl"))
using .SPMTokenizer
include(joinpath(@__DIR__, "gemma4.jl"))
using .Gemma4
using LinearAlgebra, Test

const MODEL = expanduser(
    "~/.lmstudio/models/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf")
const PROMPT = "The capital of France is"

f = GGUFFile(MODEL)
tok = Tokenizer(f)
toks = SPMTokenizer.tokenize(tok, PROMPT; add_special = true, parse_special = true)
println("prompt: ", repr(PROMPT), " → ", length(toks), " tokens")

println("loading the 26B through the pure reader (zero-copy over mmap) …")
t0 = time()
m = Gemma4.load_model(f)
println("loaded in ", round(time() - t0; digits = 1), "s — ",
    length(m.layers), " layers, ", length(m.layers[1].gate_up_exps),
    " experts/layer, embd ", typeof(m.embd).name.name,
    ", experts ", typeof(m.layers[1].gate_up_exps[1]).name.name)

# ---- oracle -------------------------------------------------------------------

L.llama_backend_init()
lmodel = L.llama_model_load_from_file(MODEL, L.llama_model_default_params())
lmodel == C_NULL && error("oracle failed to load the 26B")
vocab = L.llama_model_get_vocab(lmodel)
n_vocab = Int(L.llama_vocab_n_tokens(vocab))
ctx = L.llama_init_from_model(lmodel, L.llama_context_default_params())
ora = Matrix{Float32}(undef, n_vocab, length(toks))
one_ = Int32[0]
for (i, t) in enumerate(toks)
    one_[1] = t
    GC.@preserve one_ L.llama_decode(ctx, L.llama_batch_get_one(pointer(one_), 1))
    ora[:, i] = unsafe_wrap(Array, L.llama_get_logits_ith(ctx, Int32(-1)), n_vocab)
end

# ---- ours ---------------------------------------------------------------------

t0 = time()
ours = Gemma4.forward(m, toks)
println("pure 26B forward (", Threads.nthreads(), " threads): ",
    round(time() - t0; digits = 1), "s for ", length(toks), " positions")

piece(t) = SPMTokenizer.piece(tok, t)
@testset "gemma-4 26B vs oracle" begin
    agree = 0
    for i in eachindex(toks)
        a, b = argmax(view(ours, :, i)), argmax(view(ora, :, i))
        agree += a == b
        println("pos $i: pure→", rpad(repr(piece(a - 1)), 16),
            " oracle→", rpad(repr(piece(b - 1)), 16), a == b ? " MATCH" : "  ✗")
    end
    @test agree >= length(toks) - 1
    cs = dot(ours[:, end], ora[:, end]) / (norm(ours[:, end]) * norm(ora[:, end]))
    println("final-position cosine: ", round(cs; digits = 6))
    @test cs > 0.995
end

eog = [t for t in 0:(n_vocab - 1) if L.llama_vocab_is_eog(vocab, Int32(t))]
t0 = time()
gen = Gemma4.generate(m, toks; max_tokens = 12, eog)
dt = time() - t0
lgen = Int[]
for _ in 1:12
    lg = unsafe_wrap(Array, L.llama_get_logits_ith(ctx, Int32(-1)), n_vocab)
    nt = argmax(lg) - 1
    L.llama_vocab_is_eog(vocab, Int32(nt)) && break
    push!(lgen, nt)
    one_[1] = nt
    GC.@preserve one_ L.llama_decode(ctx, L.llama_batch_get_one(pointer(one_), 1))
end
println("pure 26B:  ", repr(join(piece.(gen))), "  (",
    round(length(gen) / dt; digits = 2), " tok/s CPU)")
println("llama.cpp: ", repr(join(piece.(lgen))))
n_common = something(findfirst(i -> i > length(lgen) || gen[i] != lgen[i],
    eachindex(gen)), length(gen) + 1) - 1
println("continuations agree for ", n_common, "/", length(gen), " tokens")

L.llama_free(ctx)
L.llama_model_free(lmodel)
