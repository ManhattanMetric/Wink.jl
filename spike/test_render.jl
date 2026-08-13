# Validate the Julia Gemma-4 renderer byte-for-byte against Python jinja2
# rendering the actual template (render_groundtruth.py). Falls back to
# structural asserts when no jinja2-equipped python is available.
#
# Run: julia --project=spike spike/test_render.jl

using Test
import JSON3

include(joinpath(@__DIR__, "render.jl"))

D(pairs...) = Dict{String, Any}(pairs...)
msg(role, content) = D("role" => role, "content" => content)

const SEARCH_TOOL = D("function" => D(
    "name" => "search_docs",
    "description" => "Semantic search over the session's documentation.",
    "parameters" => D(
        "type" => "object",
        "properties" => D(
            "query" => D("type" => "string",
                "description" => "What to look for."),
            "limit" => D("type" => "integer",
                "description" => "Maximum results.")),
        "required" => ["query"])))

const CALL_MSG = D("role" => "assistant", "content" => "",
    "tool_calls" => [D("id" => "c1", "function" => D(
        "name" => "search_docs",
        "arguments" => D("query" => "multiple dispatch", "limit" => 3)))])

const CASES = [
    # 1: plain system + user, generation prompt, thinking off
    D("messages" => [msg("system", "You are terse."), msg("user", "Hi there")]),
    # 2: multi-turn with a thought span to strip from history
    D("messages" => [msg("system", "You are terse."), msg("user", "Q1"),
        msg("assistant",
            "<|channel>thought\npondering…\n<channel|>A1 final."),
        msg("user", "Q2")]),
    # 3: tools declared alongside a system prompt
    D("messages" => [msg("system", "Use tools wisely."), msg("user", "Find dispatch docs")],
        "tools" => [SEARCH_TOOL]),
    # 4: tool call + response consumed, generation prompt — turn stays OPEN
    D("messages" => [msg("system", "s"), msg("user", "u"), CALL_MSG,
        D("role" => "tool", "tool_call_id" => "c1",
            "content" => "3 results found")],
        "tools" => [SEARCH_TOOL]),
    # 5: tool call with NO response yet — trailing <|tool_response> primer
    D("messages" => [msg("system", "s"), msg("user", "u"), CALL_MSG],
        "tools" => [SEARCH_TOOL]),
    # 6: thinking enabled
    D("messages" => [msg("system", "s"), msg("user", "u")],
        "enable_thinking" => true),
    # 7: no system, no tools — no system block at all
    D("messages" => [msg("user", "just this")]),
    # 8: no generation prompt (the prev-length continuation render)
    D("messages" => [msg("system", "s"), msg("user", "u"),
        msg("assistant", "done.")],
        "add_generation_prompt" => false),
]

julia_render(case) = render_chat(
    Gemma4Template(enable_thinking = get(case, "enable_thinking", false)),
    case["messages"];
    tools = get(case, "tools", []),
    add_assistant = get(case, "add_generation_prompt", true))

function groundtruth()
    candidates = filter(!isempty, [get(ENV, "WINK_JINJA_PYTHON", ""),
        "/Users/jballanc/.claude/jobs/4ac5c760/tmp/jinja-venv/bin/python",
        "python3"])
    spec = JSON3.write(D("template" => joinpath(@__DIR__, "gemma4_template.jinja"),
        "cases" => CASES))
    for py in candidates
        out = try
            read(pipeline(IOBuffer(spec),
                `$py $(joinpath(@__DIR__, "render_groundtruth.py"))`), String)
        catch
            continue
        end
        return collect(String.(JSON3.read(out)))
    end
    return nothing
end

@testset "gemma4 renderer" begin
    expected = groundtruth()
    if expected === nothing
        @warn "no jinja2-equipped python found; structural asserts only " *
              "(set WINK_JINJA_PYTHON for byte-exact validation)"
    else
        @testset "byte-exact vs jinja2 (case $i)" for i in eachindex(CASES)
            @test julia_render(CASES[i]) == expected[i]
        end
    end

    # structural asserts (always run)
    r = julia_render(CASES[1])
    @test startswith(r, "<|turn>system\nYou are terse.<turn|>\n<|turn>user\nHi there<turn|>\n")
    @test endswith(r, "<|turn>model\n<|channel>thought\n<channel|>")

    r = julia_render(CASES[2])
    @test occursin("A1 final.", r)
    @test !occursin("pondering", r)               # history thought spans stripped

    r = julia_render(CASES[3])
    @test occursin("<|tool>declaration:search_docs{description:<|\"|>", r)
    @test occursin("required:[<|\"|>query<|\"|>]", r)
    @test occursin("type:<|\"|>OBJECT<|\"|>", r)

    r = julia_render(CASES[4])
    @test occursin("<|tool_call>call:search_docs{limit:3,query:<|\"|>multiple dispatch<|\"|>}<tool_call|>", r)
    @test occursin("<|tool_response>response:search_docs{value:<|\"|>3 results found<|\"|>}<tool_response|>", r)
    @test !endswith(r, "<turn|>\n")               # model turn left open

    r = julia_render(CASES[5])
    @test endswith(r, "<tool_call|><|tool_response>")   # primed for results

    r = julia_render(CASES[6])
    @test occursin("<|turn>system\n<|think|>\n", r)
    @test endswith(r, "<|turn>model\n")           # thinking: no pre-closed channel

    r = julia_render(CASES[7])
    @test startswith(r, "<|turn>user\n")          # no system block

    r = julia_render(CASES[8])
    @test endswith(r, "done.<turn|>\n")           # no generation prompt
end
