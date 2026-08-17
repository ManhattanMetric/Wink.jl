# The 26B on the 40-core GPU. The CPU pure path is already oracle-exact vs
# llama.cpp (test_gemma4.jl), so it serves as the reference here — no
# llama.cpp needed. Weights upload as quantized bytes (13.4GB, never
# dequantized at rest); the KA kernels in quant.jl dequantize per-thread.
#
# Run: julia -t 8 --project=spike pure/test_gemma4_metal.jl

include(joinpath(@__DIR__, "quant.jl"))
using .Quant
include(joinpath(@__DIR__, "gguf.jl"))
using .GGUF
include(joinpath(@__DIR__, "tokenizer.jl"))
using .SPMTokenizer
include(joinpath(@__DIR__, "gemma4.jl"))
using .Gemma4
using Metal, Adapt, LinearAlgebra, Test

const MODEL = expanduser(
    "~/.lmstudio/models/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf")
const PROMPT = "The capital of France is"

f = GGUFFile(MODEL)
tok = Tokenizer(f)
toks = SPMTokenizer.tokenize(tok, PROMPT; add_special = true, parse_special = true)

m = load_model(f)
t0 = time()
cpu = Gemma4.forward(m, toks)
println("CPU forward:   ", round(time() - t0; digits = 1), "s (",
    Threads.nthreads(), " threads)")

t0 = time()
gm = adapt(MtlArray, m)
println("upload to Metal: ", round(time() - t0; digits = 1), "s")

t0 = time()
gpu = Array(Gemma4.forward(gm, toks))
println("Metal forward: ", round(time() - t0; digits = 1), "s (incl. compile)")
t0 = time()
Array(Gemma4.forward(gm, toks))
println("Metal forward (warm): ", round(time() - t0; digits = 1), "s")

piece(t) = SPMTokenizer.piece(tok, t)
@testset "gemma-4 26B Metal vs CPU" begin
    for i in eachindex(toks)
        a, b = argmax(view(gpu, :, i)), argmax(view(cpu, :, i))
        println("pos $i: metal→", rpad(repr(piece(a - 1)), 16),
            " cpu→", rpad(repr(piece(b - 1)), 16), a == b ? " MATCH" : "  ✗")
        @test a == b
    end
    cs = dot(gpu[:, end], cpu[:, end]) / (norm(gpu[:, end]) * norm(cpu[:, end]))
    println("final-position cosine: ", round(cs; digits = 6))
    # the CPU path requantizes activations to int8 (SDOT); the GPU kernels
    # compute exact f32 dequant — they now differ by the requantization band
    @test cs > 0.995
end

# generation face-off: same tokens expected, and the GPU should finally
# out-run the CPU at 26B scale (the 270m was launch-latency-bound)
N = 24
t0 = time()
ggen = Gemma4.generate(gm, toks; max_tokens = N)
gdt = time() - t0
t0 = time()
cgen = Gemma4.generate(m, toks; max_tokens = N)
cdt = time() - t0
println("metal: ", repr(join(piece.(ggen))), "  (",
    round(length(ggen) / gdt; digits = 2), " tok/s)")
println("cpu:   ", repr(join(piece.(cgen))), "  (",
    round(length(cgen) / cdt; digits = 2), " tok/s)")
n = something(findfirst(i -> i > length(cgen) || ggen[i] != cgen[i],
    eachindex(ggen)), length(ggen) + 1) - 1
println("generations agree for ", n, "/", length(ggen),
    " tokens (divergence past that is requantization noise)")
@test n >= 2
