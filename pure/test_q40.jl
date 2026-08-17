# Quantized inference through DISPATCH: the same Gemma3.forward that runs
# F16 weights runs the QAT Q4_0 file, because the quantized tensors load as
# Q4_0Matrix/Q8_0Matrix — AbstractMatrix{Float32}s whose mul! dequantizes
# inside the dot product. Full-precision weights never materialize.
#
# Oracle expectations differ from the F16 tests: llama.cpp REQUANTIZES the
# activations to q8_0 for its integer q4_0 dot products, while we compute
# the exact dequantized product in f32 — the two are deliberately not
# bit-identical, so agreement is judged at top-1 and cosine, and greedy
# continuations may legitimately diverge after several tokens.
#
# Run: julia -t auto --project=spike pure/test_q40.jl

include(joinpath(@__DIR__, "..", "spike", "LibLlama.jl"))
import .LibLlama as L
include(joinpath(@__DIR__, "quant.jl"))
using .Quant
include(joinpath(@__DIR__, "gguf.jl"))
using .GGUF
include(joinpath(@__DIR__, "tokenizer.jl"))
using .SPMTokenizer
include(joinpath(@__DIR__, "gemma3.jl"))
using .Gemma3
using LinearAlgebra, Test

const MODEL = joinpath(@__DIR__, "models", "gemma-3-270m-it-qat-Q4_0.gguf")
const PROMPT = "The capital of France is"

f = GGUFFile(MODEL)
tok = Tokenizer(f)
toks = SPMTokenizer.tokenize(tok, PROMPT; add_special = true, parse_special = true)

println("loading QUANTIZED weights (no dequant-to-array anywhere) …")
m = Gemma3.load_model(f)
println("weight types: wq → ", typeof(m.layers[1].wq), ", embd → ", typeof(m.embd))
@test m.layers[1].wq isa Quant.Q4_0Matrix
@test m.embd isa Quant.Q8_0Matrix

L.llama_backend_init()
lmodel = L.llama_model_load_from_file(MODEL, L.llama_model_default_params())
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

t0 = time()
ours = Gemma3.forward(m, toks)
println("quantized pure forward (", Threads.nthreads(), " threads): ",
    round(time() - t0; digits = 2), "s for ", length(toks), " positions")

piece(t) = SPMTokenizer.piece(tok, t)
@testset "q4_0 vs oracle" begin
    agree = 0
    for i in eachindex(toks)
        a, b = argmax(view(ours, :, i)), argmax(view(ora, :, i))
        agree += a == b
        println("pos $i: pure→", rpad(repr(piece(a - 1)), 14),
            " oracle→", rpad(repr(piece(b - 1)), 14), a == b ? " MATCH" : "  ✗")
    end
    @test agree >= length(toks) - 1     # requantization slack, judged honestly
    cs = dot(ours[:, end], ora[:, end]) / (norm(ours[:, end]) * norm(ora[:, end]))
    println("final-position cosine: ", round(cs; digits = 6))
    @test cs > 0.999
end

eog = [t for t in 0:(n_vocab - 1) if L.llama_vocab_is_eog(vocab, Int32(t))]
t0 = time()
gen = Gemma3.generate(m, toks; max_tokens = 24, eog)
dt = time() - t0
lgen = Int[]
for _ in 1:24
    lg = unsafe_wrap(Array, L.llama_get_logits_ith(ctx, Int32(-1)), n_vocab)
    nt = argmax(lg) - 1
    L.llama_vocab_is_eog(vocab, Int32(nt)) && break
    push!(lgen, nt)
    one_[1] = nt
    GC.@preserve one_ L.llama_decode(ctx, L.llama_batch_get_one(pointer(one_), 1))
end
println("pure q4:   ", repr(join(piece.(gen))), "  (",
    round(length(gen) / dt; digits = 1), " tok/s CPU)")
println("llama.cpp: ", repr(join(piece.(lgen))))
n_common = something(findfirst(i -> i > length(lgen) || gen[i] != lgen[i],
    eachindex(gen)), length(gen) + 1) - 1
println("continuations agree for ", n_common, "/", length(gen),
    " tokens (divergence past that is requantization noise, not error)")

L.llama_free(ctx)
L.llama_model_free(lmodel)
