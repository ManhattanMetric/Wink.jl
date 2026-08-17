# Sliding-window + KV-cache validation against the llama.cpp oracle on a
# prompt LONGER than the 512-token window — the first input where local
# attention actually differs from global, so any window off-by-one shows up
# as logit disagreement. Also checks cache self-consistency: one big prefill
# vs token-by-token stepping must agree exactly.
#
# Run: julia --project=spike pure/test_swa.jl

include(joinpath(@__DIR__, "..", "spike", "LibLlama.jl"))
import .LibLlama as L
include(joinpath(@__DIR__, "quant.jl"))
include(joinpath(@__DIR__, "gguf.jl"))
using .GGUF
include(joinpath(@__DIR__, "tokenizer.jl"))
using .SPMTokenizer
include(joinpath(@__DIR__, "gemma3.jl"))
using .Gemma3
using LinearAlgebra, Test

const MODEL = joinpath(@__DIR__, "models", "gemma-3-270m-it-F16.gguf")

para = "The sliding window means local layers can only look back so far. " *
       "Julia makes the mask a slice of the cache, nothing more. "
text = repeat(para, 30) * "In conclusion, the capital of France is"

f = GGUFFile(MODEL)
tok = Tokenizer(f)
toks = tokenize(tok, text; add_special = true, parse_special = true)
println("prompt: ", length(toks), " tokens (window is 512 — local ≠ global here)")
@assert length(toks) > 600

m = Gemma3.load_model(f)

# ---- oracle -------------------------------------------------------------------

L.llama_backend_init()
lmodel = L.llama_model_load_from_file(MODEL, L.llama_model_default_params())
vocab = L.llama_model_get_vocab(lmodel)
n_vocab = Int(L.llama_vocab_n_tokens(vocab))
cp = Ref(L.llama_context_default_params())
let p = Base.unsafe_convert(Ptr{L.llama_context_params}, cp)
    GC.@preserve cp begin
        unsafe_store!(Ptr{UInt32}(Ptr{UInt8}(p) + fieldoffset(
            L.llama_context_params, Base.fieldindex(L.llama_context_params, :n_ctx))),
            UInt32(2048))
    end
end
ctx = L.llama_init_from_model(lmodel, cp[])
t32 = Int32.(toks)
GC.@preserve t32 begin
    L.llama_decode(ctx, L.llama_batch_get_one(pointer(t32), length(t32))) == 0 ||
        error("oracle decode failed")
end
ora = copy(unsafe_wrap(Array, L.llama_get_logits_ith(ctx, Int32(-1)), n_vocab))

# ---- ours: big prefill --------------------------------------------------------

t0 = time()
c1 = Gemma3.KVCache(m; capacity = length(toks) + 8)
ours = Gemma3.step!(m, c1, toks)[:, end]
println("pure prefill of $(length(toks)) tokens: ", round(time() - t0; digits = 1), "s")

# ---- ours: token-by-token (cache self-consistency) ----------------------------

c2 = Gemma3.KVCache(m; capacity = length(toks) + 8)
local stepped
for t in toks
    global stepped = Gemma3.step!(m, c2, [t])
end
stepped = stepped[:, 1]

piece_(id) = SPMTokenizer.piece(tok, id)
@testset "sliding window + cache" begin
    @test argmax(ours) == argmax(ora)
    cs = dot(ours, ora) / (norm(ours) * norm(ora))
    println("top-1: pure→", repr(piece_(argmax(ours) - 1)),
        "  oracle→", repr(piece_(argmax(ora) - 1)),
        "  cosine: ", round(cs; digits = 6))
    @test cs > 0.999
    # prefill and one-at-a-time stepping are bit-for-bit questions of order;
    # demand near-identity
    @test maximum(abs.(ours .- stepped)) < 1e-3
    @test argmax(ours) == argmax(stepped)
end

# greedy continuation face-off on the long prompt
gen = Gemma3.generate(m, toks; max_tokens = 12, eog = [tok.eos, 1])
lgen = Int[]
one_ = Int32[0]
for _ in 1:12
    lg = unsafe_wrap(Array, L.llama_get_logits_ith(ctx, Int32(-1)), n_vocab)
    nt = argmax(lg) - 1
    L.llama_vocab_is_eog(vocab, Int32(nt)) && break
    push!(lgen, nt)
    one_[1] = nt
    GC.@preserve one_ L.llama_decode(ctx, L.llama_batch_get_one(pointer(one_), 1))
end
println("pure:      ", repr(join(piece_.(gen))))
println("llama.cpp: ", repr(join(piece_.(lgen))))
println(gen == lgen ? "IDENTICAL past-window continuations" : "DIVERGED")

L.llama_free(ctx)
L.llama_model_free(lmodel)
