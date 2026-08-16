# Oracle battery: the pure SPM tokenizer vs llama.cpp's, string for string.
# llama.cpp appears here ONLY as the dev-time oracle; the pure path no longer
# needs it at runtime.
#
# Run: julia --project=spike pure/test_tokenizer.jl

include(joinpath(@__DIR__, "..", "spike", "LibLlama.jl"))
import .LibLlama as L
include(joinpath(@__DIR__, "gguf.jl"))
using .GGUF
include(joinpath(@__DIR__, "tokenizer.jl"))
using .SPMTokenizer
using Test

const MODEL = joinpath(@__DIR__, "models", "gemma-3-270m-it-F16.gguf")

L.llama_backend_init()
mp = Ref(L.llama_model_default_params())
let p = Base.unsafe_convert(Ptr{L.llama_model_params}, mp)
    GC.@preserve mp unsafe_store!(
        Ptr{Bool}(Ptr{UInt8}(p) + fieldoffset(L.llama_model_params,
            Base.fieldindex(L.llama_model_params, :vocab_only))), true)
end
lmodel = L.llama_model_load_from_file(MODEL, mp[])
vocab = L.llama_model_get_vocab(lmodel)

function oracle(text; add_special, parse_special)
    buf = Vector{Int32}(undef, 4 * ncodeunits(text) + 32)
    n = L.llama_tokenize(vocab, text, ncodeunits(text), buf, length(buf),
        add_special, parse_special)
    n < 0 && error("oracle tokenize failed on $(repr(text))")
    return Int.(buf[1:n])
end

tok = Tokenizer(GGUFFile(MODEL))

const BATTERY = [
    "The capital of France is",
    "Hello, world!",
    "  leading and trailing  ",
    "a\nb\n\nc\n\n\nd",
    "f(x) = x^2 .+ γ  # comment ∘ ∀ε>0",
    "julia> sort([3,1,2]; rev=true)",
    "3.14159 + 2im ≈ π",
    "tabs\tand\ttabs",
    "🦋 café naïve — em-dash… “quotes”",
    "日本語のテキストと한국어 텍스트",
    "Русский текст здесь",
    "<start_of_turn>user\nHello there<end_of_turn>\n<start_of_turn>model\n",
    "text<eos>after",
    "no specials here <notatoken> honest",
    "module BlogDomain\nusing Dates\nstruct Post\n    id::Int\nend\nend",
    "x = \"string with ▁ actual underscore-block\"",
    "CREATE TABLE posts (id INTEGER PRIMARY KEY, title TEXT);",
    "",
    " ",
    "<bos><eos>",
    repeat("word ", 50),
]

@testset "pure SPM vs llama.cpp oracle" begin
    for text in BATTERY, ps in (true, false), as in (true, false)
        ours = tokenize(tok, text; add_special = as, parse_special = ps)
        theirs = oracle(text; add_special = as, parse_special = ps)
        @test ours == theirs
        ours == theirs || println("MISMATCH ", repr(text), " ps=$ps as=$as\n  ours:   ",
            ours, "\n  oracle: ", theirs)
    end

    # detokenization matches llama.cpp piece rendering across the id space we use
    ids = tokenize(tok, BATTERY[12]; add_special = true, parse_special = true)
    lpiece(id) = (b = Vector{UInt8}(undef, 128);
        n = L.llama_token_to_piece(vocab, Int32(id), b, 128, 0, true);
        String(b[1:max(n, 0)]))
    @test detokenize(tok, ids; special = true) == join(lpiece.(ids))

    # round-trip of ordinary text
    plain = "The quick brown fox jumps over the lazy dog."
    @test detokenize(tok, tokenize(tok, plain)) == plain
end

L.llama_model_free(lmodel)
println("tokenizer battery complete")
