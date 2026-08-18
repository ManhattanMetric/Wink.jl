# Quantized OLMoE vs llama.cpp: the fast shape for the fine-tuning
# direction — 1B active params at q4_0. Expert stacks are zero-copy
# Q4_0Stack views; the pure BPE tokenizer feeds both engines. Both engines
# requantize activations (llama.cpp q8_0, our SDOT path likewise), so the
# bar is per-position top-1 + cosine + a shared greedy prefix.
#
# Run: julia -t 8 --project=spike pure/test_olmoe_q40.jl

include(joinpath(@__DIR__, "..", "spike", "LibLlama.jl"))
import .LibLlama as L
include(joinpath(@__DIR__, "quant.jl"))
using .Quant
include(joinpath(@__DIR__, "gguf.jl"))
using .GGUF
include(joinpath(@__DIR__, "..", "src", "bpe.jl"))
using .BPETokenizer
include(joinpath(@__DIR__, "olmoe.jl"))
using .OLMoE
using LinearAlgebra, Test

const MODEL = joinpath(@__DIR__, "models", "OLMoE-1B-7B-0924-Instruct-Q4_0.gguf")
const PROMPT = "The capital of France is"

f = GGUFFile(MODEL)
tok = Tokenizer(f)
toks = BPETokenizer.tokenize(tok, PROMPT; add_special = true, parse_special = true)
println("prompt: ", repr(PROMPT), " → ", length(toks), " tokens")

m = OLMoE.load_model(f)
println("experts: ", typeof(m.layers[1].gate_exps).name.name,
    "  attn: ", typeof(m.layers[1].wq).name.name,
    "  embd: ", typeof(m.embd).name.name)

L.llama_backend_init()
lmodel = L.llama_model_load_from_file(MODEL, L.llama_model_default_params())
lmodel == C_NULL && error("oracle failed to load")
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
ours = OLMoE.forward(m, toks)
println("pure forward (", Threads.nthreads(), " threads): ",
    round(time() - t0; digits = 1), "s for ", length(toks), " positions")

piece(t) = BPETokenizer.piece(tok, t)
@testset "OLMoE q4_0 vs oracle" begin
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
gen = OLMoE.generate(m, toks; max_tokens = 24, eog)
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
println("pure:      ", repr(join(piece.(gen))), "  (",
    round(length(gen) / dt; digits = 2), " tok/s CPU)")
println("llama.cpp: ", repr(join(piece.(lgen))))
n_common = something(findfirst(i -> i > length(lgen) || gen[i] != lgen[i],
    eachindex(gen)), length(gen) + 1) - 1
println("continuations agree for ", n_common, "/", length(gen),
    " tokens (divergence past that is requantization noise)")
@test n_common >= 2

L.llama_free(ctx)
L.llama_model_free(lmodel)
