# Offline tests for the in-process backend's conversion layers. The live
# generation path needs a GGUF on disk and is exercised manually (see
# spike/); everything here runs without a model.
@testset "local backend conversions" begin
    # inactive by default: the PT route is untouched
    @test Wink.LOCAL_MODEL[] === nothing

    # Wink-native history → renderer messages
    m = Wink._local_msg(PT.SystemMessage("sys"))
    @test m["role"] == "system" && m["content"] == "sys"
    m = Wink._local_msg(PT.UserMessage("hi"))
    @test m["role"] == "user"
    m = Wink._local_msg(PT.AIMessage("text"))
    @test m["role"] == "assistant" && m["content"] == "text"

    req = PT.AIToolRequest(; content = "checking",
        tool_calls = [PT.ToolMessage(; tool_call_id = "local-1", raw = "{}",
            name = "get_doc", args = Dict{Symbol, Any}(:name => "sort"))])
    m = Wink._local_msg(req)
    @test m["role"] == "assistant"
    @test m["tool_calls"][1]["function"]["name"] == "get_doc"
    @test m["tool_calls"][1]["function"]["arguments"] == Dict("name" => "sort")
    # call-bearing turns: preface rides the reasoning field (rendered as the
    # thought channel), never content (which would close the turn dead)
    @test m["content"] == ""
    @test m["reasoning"] == "checking"
    plain = Wink._local_msg(PT.AIToolRequest(; content = "just text"))
    @test plain["content"] == "just text"
    @test plain["reasoning"] == ""

    m = Wink._local_msg(PT.ToolMessage(; content = "the docstring",
        tool_call_id = "local-1", raw = "{}", name = "get_doc"))
    @test m["role"] == "tool" && m["content"] == "the docstring"

    # real registry tools → schemas → grammar, end to end offline
    tools = collect(values(Wink.build_tool_map()))
    schemas = [Wink._tool_schema(t) for t in tools]
    @test any(s -> s["function"]["name"] == "eval_code", schemas)
    @test all(s -> haskey(s["function"], "parameters"), schemas)
    g = Wink.tool_call_grammar(schemas)
    @test occursin("tool-eval-code ::=", g)
    @test occursin("tool-get-doc ::= \"get_doc{\" \"name:\" string \"}\"", g)
    @test occursin("root ::= \"<|tool_call>call:\"", g)

    # a native tool exchange renders through the Gemma-4 family and the
    # emitted call parses back to the same name and args
    hist = PT.AbstractMessage[
        PT.SystemMessage("sys"), PT.UserMessage("docs for sort?"), req,
        PT.ToolMessage(; content = "sorts things", tool_call_id = "local-1",
            raw = "{}", name = "get_doc")]
    msgs = [Wink._local_msg(x) for x in hist]
    r = Wink.render_chat(Wink.Gemma4Template(), msgs;
        tools = schemas, add_assistant = true)
    @test occursin("<|tool_call>call:get_doc{name:<|\"|>sort<|\"|>}<tool_call|>", r)
    # the preface renders as the thought channel before the call
    @test occursin("<|channel>thought\nchecking\n<channel|><|tool_call>", r)
    @test occursin("<|tool_response>response:get_doc{value:<|\"|>sorts things<|\"|>}", r)
    c = Wink.parse_tool_call(r)
    @test c.name == "get_doc" && c.args == Dict("name" => "sort")
    # model turn stays open after tool responses — the agentic-loop shape
    @test !endswith(r, "<turn|>\n")

    # generation-prompt suffix is a pure suffix of the full render (the
    # invariant the backend's KV scratch/rollback design rests on)
    base = Wink.render_chat(Wink.Gemma4Template(), msgs;
        tools = schemas, add_assistant = false)
    @test startswith(r, base)
end

@testset "local budget coordination" begin
    old_budget = Wink.CONFIG.context_budget
    try
        # the frontier-sized default lowers to 75% of the model's window
        Wink.CONFIG.context_budget = 100_000
        @test Wink._coordinate_budget!(32_768) == 24_576
        @test Wink.CONFIG.context_budget == 24_576

        # an explicit lower setting is respected
        Wink.CONFIG.context_budget = 8_000
        @test Wink._coordinate_budget!(32_768) == 8_000

        # disabled stays disabled (warned, not overridden)
        Wink.CONFIG.context_budget = 0
        @test (@test_logs (:warn, r"auto-compaction is disabled") Wink._coordinate_budget!(16_384)) == 0
    finally
        Wink.CONFIG.context_budget = old_budget
    end

    # the wall pre-check produces an actionable error, not a C-level one
    lm_like = (; n_ctx = 1_024, )
    @test Wink._room_check((; n_ctx = 16_384), 9_000) === nothing
    err = try
        Wink._room_check(lm_like, 9_000)
        nothing
    catch e
        sprint(showerror, e)
    end
    @test occursin("context window full", err)
    @test occursin(":compact", err)
    @test occursin("n_ctx", err)
end
