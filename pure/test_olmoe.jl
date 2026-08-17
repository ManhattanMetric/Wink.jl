# MoE oracle validation: pure-Julia OLMoE vs llama.cpp on the same GGUF,
# position by position at the logits. The tokenizer is borrowed from
# llama.cpp for this spike (OLMoE is GPT-2 BPE; the pure BPE tokenizer is
# queued work).
#
# Run: julia --project=spike pure/test_olmoe.jl

include(joinpath(@__DIR__, "..", "spike", "LibLlama.jl"))
import .LibLlama as L
include(joinpath(@__DIR__, "quant.jl"))
include(joinpath(@__DIR__, "gguf.jl"))
using .GGUF
include(joinpath(@__DIR__, "olmoe.jl"))
using .OLMoE
using LinearAlgebra, Test

const MODEL = joinpath(@__DIR__, "models", "OLMoE-1B-7B-0924-Instruct-f16.gguf")
const PROMPT = "The capital of France is"

L.llama_backend_init()
lmodel = L.llama_model_load_from_file(MODEL, L.llama_model_default_params())
lmodel == C_NULL && error("oracle failed to load")
vocab = L.llama_model_get_vocab(lmodel)
n_vocab = Int(L.llama_vocab_n_tokens(vocab))
ctx = L.llama_init_from_model(lmodel, L.llama_context_default_params())

buf = Vector{Int32}(undef, 128)
n = L.llama_tokenize(vocab, PROMPT, ncodeunits(PROMPT), buf, length(buf), true, true)
toks = Int.(buf[1:n])
piece(t) = (b = Vector{UInt8}(undef, 64);
    len = L.llama_token_to_piece(vocab, Int32(t), b, 64, 0, true);
    String(b[1:max(len, 0)]))
println("prompt: ", repr(PROMPT), " → ", length(toks), " tokens")

ora = Matrix{Float32}(undef, n_vocab, length(toks))
one_ = Int32[0]
for (i, t) in enumerate(toks)
    one_[1] = t
    GC.@preserve one_ L.llama_decode(ctx, L.llama_batch_get_one(pointer(one_), 1))
    ora[:, i] = unsafe_wrap(Array, L.llama_get_logits_ith(ctx, Int32(-1)), n_vocab)
end

println("loading 13.8GB of experts through the pure reader …")
t0 = time()
m = OLMoE.load_model(GGUFFile(MODEL))
println("loaded in ", round(time() - t0; digits = 1), "s")
t0 = time()
ours = OLMoE.forward(m, toks)
println("pure MoE forward over $(length(toks)) positions: ",
    round(time() - t0; digits = 2), "s")

@testset "MoE vs oracle" begin
    agree = 0
    for i in eachindex(toks)
        a, b = argmax(view(ours, :, i)), argmax(view(ora, :, i))
        agree += a == b
        println("pos $i: pure→", rpad(repr(piece(a - 1)), 14),
            " oracle→", rpad(repr(piece(b - 1)), 14), a == b ? " MATCH" : "  ✗")
    end
    @test agree == length(toks)
    cs = dot(ours[:, end], ora[:, end]) / (norm(ours[:, end]) * norm(ora[:, end]))
    println("final-position cosine: ", round(cs; digits = 6),
        "  max|Δ|: ", round(maximum(abs.(ours[:, end] .- ora[:, end])); digits = 3))
    @test cs > 0.999
end

eog = [t for t in 0:(n_vocab - 1) if L.llama_vocab_is_eog(vocab, Int32(t))]
t0 = time()
gen = OLMoE.generate(m, toks; max_tokens = 16, eog)
dt = time() - t0
lgen = Int[]
for _ in 1:16
    lg = unsafe_wrap(Array, L.llama_get_logits_ith(ctx, Int32(-1)), n_vocab)
    nt = argmax(lg) - 1
    L.llama_vocab_is_eog(vocab, Int32(nt)) && break
    push!(lgen, nt)
    one_[1] = nt
    GC.@preserve one_ L.llama_decode(ctx, L.llama_batch_get_one(pointer(one_), 1))
end
println("pure:      ", repr(join(piece.(gen))), "  (",
    round(length(gen) / dt; digits = 1), " tok/s CPU)")
println("llama.cpp: ", repr(join(piece.(lgen))))
println(gen == lgen ? "IDENTICAL MoE GREEDY CONTINUATIONS" : "DIVERGED")

L.llama_free(ctx)
L.llama_model_free(lmodel)
