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
