# Unit tests for the tool-call grammar compiler and parser.
#
# Run: julia --project=spike spike/test_grammar.jl

using Test

include(joinpath(@__DIR__, "render.jl"))
include(joinpath(@__DIR__, "grammar.jl"))

D(pairs...) = Dict{String, Any}(pairs...)

const TOOLS = [
    D("function" => D("name" => "search_docs",
        "description" => "Semantic doc search.",
        "parameters" => D("type" => "object",
            "properties" => D(
                "query" => D("type" => "string"),
                "limit" => D("type" => "integer")),
            "required" => ["query"]))),
    D("function" => D("name" => "get_doc",
        "description" => "Fetch a docstring.",
        "parameters" => D("type" => "object",
            "properties" => D("name" => D("type" => "string")),
            "required" => ["name"]))),
]

@testset "grammar compiler" begin
    g = tool_call_grammar(TOOLS)
    @test occursin("root ::= \"<|tool_call>call:\" call \"<tool_call|>\"", g)
    @test occursin("call ::= tool-search-docs | tool-get-doc", g)
    # args in dictsort order: limit before query
    @test occursin("tool-search-docs ::= \"search_docs{\" \"limit:\" integer \",\" \"query:\" string \"}\"", g)
    @test occursin("tool-get-doc ::= \"get_doc{\" \"name:\" string \"}\"", g)
    # the quote token survives GBNF escaping
    @test occursin("string ::= \"<|\\\"|>\" strchar* \"<|\\\"|>\"", g)
    @test_throws Exception tool_call_grammar([])
end

@testset "call parser" begin
    # exactly what render_chat emits for a historical call — round-trip
    rendered = "<|tool_call>call:search_docs{limit:3,query:<|\"|>multiple dispatch<|\"|>}<tool_call|>"
    c = parse_tool_call(rendered)
    @test c.name == "search_docs"
    @test c.args == Dict("limit" => 3, "query" => "multiple dispatch")

    # surrounded by prose (the lazy-mode shape)
    c = parse_tool_call("Let me look that up.\n" * rendered * "\n")
    @test c.name == "search_docs"

    # nested and typed values
    c = parse_tool_call("<|tool_call>call:f{a:{b:<|\"|>x<|\"|>,c:[1,2.5,true,null]},d:false}<tool_call|>")
    @test c.args["a"]["b"] == "x"
    @test c.args["a"]["c"] == Any[1, 2.5, true, nothing]
    @test c.args["d"] === false

    # strings containing grammar-ish characters
    c = parse_tool_call("<|tool_call>call:g{code:<|\"|>f(x) = {x, [1]}\n<|\"|>}<tool_call|>")
    @test c.args["code"] == "f(x) = {x, [1]}\n"

    @test parse_tool_call("no call here") === nothing
end

println("grammar tests passed")
