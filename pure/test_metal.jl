# The portability payoff: the SAME forward pass on Metal, reached by nothing
# but `Adapt.adapt(MtlArray, model)`. Correctness vs the CPU run (which is
# itself oracle-validated), then a timing face-off. On CUDA/AMD/oneAPI boxes
# the identical test would run with their array type — that's the point.
#
# Run: julia --project=spike pure/test_metal.jl

include(joinpath(@__DIR__, "gguf.jl"))
using .GGUF
include(joinpath(@__DIR__, "tokenizer.jl"))
using .SPMTokenizer
include(joinpath(@__DIR__, "gemma3.jl"))
using .Gemma3
using Metal, Adapt, LinearAlgebra, Test

const MODEL = joinpath(@__DIR__, "models", "gemma-3-270m-it-F16.gguf")

f = GGUFFile(MODEL)
tok = Tokenizer(f)
m = Gemma3.load_model(f)
prompt = "<start_of_turn>user\nIn one short sentence, what is the Julia language known for?<end_of_turn>\n<start_of_turn>model\n"
ids = tokenize(tok, prompt; add_special = true, parse_special = true)

t0 = time()
mg = Adapt.adapt(MtlArray, m)
println("model moved to Metal in ", round(time() - t0; digits = 1), "s (",
    typeof(mg.embd), ")")

@testset "Metal ≡ CPU" begin
    lc = Gemma3.forward(m, ids)[:, end]
    lg = Array(Gemma3.forward(mg, ids))[:, end]
    @test argmax(lc) == argmax(lg)
    cs = dot(lc, lg) / (norm(lc) * norm(lg))
    println("final-position: cpu→", repr(SPMTokenizer.piece(tok, argmax(lc) - 1)),
        "  metal→", repr(SPMTokenizer.piece(tok, argmax(lg) - 1)),
        "  cosine: ", round(cs; digits = 6))
    @test cs > 0.9999

    gc_ = Gemma3.generate(m, ids; max_tokens = 24, eog = [tok.eos, 1])
    gg = Gemma3.generate(mg, ids; max_tokens = 24, eog = [tok.eos, 1])
    println("cpu:   ", repr(detokenize(tok, gc_)))
    println("metal: ", repr(detokenize(tok, gg)))
    @test gc_ == gg
end

# timing (after warmup above)
for (name, mm) in (("cpu", m), ("metal", mg))
    t0 = time()
    g = Gemma3.generate(mm, ids; max_tokens = 32, eog = Int[])
    dt = time() - t0
    println(rpad(name, 6), length(g), " tokens in ", round(dt; digits = 2),
        "s → ", round(length(g) / dt; digits = 1), " tok/s")
end
