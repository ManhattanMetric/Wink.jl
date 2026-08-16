# Chat generation with ZERO llama.cpp: GGUF reading, tokenization, the
# forward pass, sampling, and detokenization are all pure Julia. This is the
# distributable artifact in miniature — stdlib only, no vendored binaries,
# no ccalls. (llama.cpp survives in this directory solely as the dev-time
# oracle behind run.jl and test_tokenizer.jl.)
#
# Run: julia pure/chat_pure.jl [prompt]

include(joinpath(@__DIR__, "gguf.jl"))
using .GGUF
include(joinpath(@__DIR__, "tokenizer.jl"))
using .SPMTokenizer
include(joinpath(@__DIR__, "gemma3.jl"))
using .Gemma3

const MODEL = joinpath(@__DIR__, "models", "gemma-3-270m-it-F16.gguf")
const QUESTION = get(ARGS, 1, "In one short sentence, what is the Julia language known for?")

t0 = time()
f = GGUFFile(MODEL)
tok = Tokenizer(f)
m = Gemma3.load_model(f)
println("model + tokenizer loaded in ", round(time() - t0; digits = 1),
    "s — pure Julia, no llama.cpp")

# gemma-3's chat format, straight from its documented convention
prompt = "<start_of_turn>user\n" * QUESTION * "<end_of_turn>\n<start_of_turn>model\n"
ids = tokenize(tok, prompt; add_special = true, parse_special = true)
println("prompt: ", length(ids), " tokens")

t0 = time()
gen = Gemma3.generate(m, ids; max_tokens = 48, eog = [tok.eos, 1])
dt = time() - t0
println("\nassistant: ", strip(detokenize(tok, gen)))
println("\n", length(gen), " tokens in ", round(dt; digits = 1), "s → ",
    round(length(gen) / dt; digits = 1), " tok/s (CPU, KV-cached)")
