# Prefill scaling on the 26B: CPU vs Metal, 512-token chunks (the realistic
# shape for Wink's long system prompt). With the fused MoE and attention
# kernels the GPU wins prefill ~4×; generation (T=1) remains CPU territory —
# launch latency dominates single-token steps.
#
# Run: julia -t 8 --project=spike pure/bench_gemma4_prefill.jl

include(joinpath(@__DIR__, "quant.jl"))
using .Quant
include(joinpath(@__DIR__, "gguf.jl"))
using .GGUF
include(joinpath(@__DIR__, "tokenizer.jl"))
using .SPMTokenizer
include(joinpath(@__DIR__, "gemma4.jl"))
using .Gemma4
using Metal, Adapt

const MODEL = expanduser(
    "~/.lmstudio/models/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf")
f = GGUFFile(MODEL)
tok = Tokenizer(f)
text = repeat("Julia is a high-level dynamic language for technical computing. ", 120)
toks = SPMTokenizer.tokenize(tok, text; add_special = true, parse_special = true)
println("prefill benchmark: ", length(toks), " tokens, 512-token chunks")

function prefill(m, toks; chunk = 512)
    c = Gemma4.KVCache(m; capacity = length(toks))
    local logits
    for lo in 1:chunk:length(toks)
        logits = Gemma4.step!(m, c, toks[lo:min(lo + chunk - 1, length(toks))])
    end
    return logits
end

m = load_model(f)
t0 = time(); lc = prefill(m, toks); cdt = time() - t0
println("CPU:   ", round(cdt; digits = 1), "s = ",
    round(length(toks) / cdt; digits = 1), " tok/s prefill")

gm = adapt(MtlArray, m)
prefill(gm, toks[1:32])   # compile warmup
t0 = time(); lg = Array(prefill(gm, toks)); gdt = time() - t0
println("Metal: ", round(gdt; digits = 1), "s = ",
    round(length(toks) / gdt; digits = 1), " tok/s prefill")
println("top-1 agree: ", argmax(lg[:, end]) == argmax(Array(lc)[:, end]))
