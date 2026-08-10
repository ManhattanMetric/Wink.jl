@testset "compact fold" begin
    long = repeat("x", 300)
    block(name, body) = "<tool_result name=\"$name\">\n$body\n</tool_result>"

    # a stale re-derivable result folds to an elision that keeps the tool name
    out, n = Wink.fold_message(block("get_source", long))
    @test n == 1
    @test occursin("elided", out)
    @test occursin("get_source", out)
    @test !occursin(long, out)

    # mutating tools' results are historical facts — never folded
    for name in ("eval_code", "run_shell", "edit_file")
        out, n = Wink.fold_message(block(name, long))
        @test n == 0
        @test occursin(long, out)
    end

    # short bodies are not worth an elision line
    out, n = Wink.fold_message(block("get_doc", "tiny"))
    @test n == 0
    @test occursin("tiny", out)

    # blocks inside one message fold independently
    out, n = Wink.fold_message(block("get_doc", long) * "\n" * block("eval_code", long))
    @test n == 1
    @test occursin("elided", out)
    @test occursin(long, out)

    # history level: the most recent keep_recent result messages are protected
    hist = PT.AbstractMessage[
        PT.SystemMessage("sys"),
        PT.UserMessage("question"),
        PT.AIMessage("→ get_doc(...)"),
        PT.UserMessage(block("get_doc", long)),
        PT.AIMessage("→ get_source(...)"),
        PT.UserMessage(block("get_source", long)),
        PT.AIMessage("→ list_methods(...)"),
        PT.UserMessage(block("list_methods", long)),
    ]
    @test Wink.fold_history!(hist; keep_recent = 2) == 1
    @test occursin("elided", hist[4].content)
    @test occursin(long, hist[6].content)
    @test occursin(long, hist[8].content)
    # a second pass finds nothing new
    @test Wink.fold_history!(hist; keep_recent = 2) == 0

    # global-chat wrapper is a no-op without a conversation
    Wink.reset!()
    @test Wink.compact!() == 0
end

@testset "compact trigger" begin
    # The echo replays a get_doc call whose usage reports a prompt far over
    # budget, so every round triggers maybe_compact!; once more results exist
    # than keep_recent, the oldest gets folded mid-turn.
    response = Dict{Symbol, Any}(
        :content => Dict{Symbol, Any}[Dict{Symbol, Any}(:type => "tool_use",
            :id => "toolu_1", :name => "get_doc",
            :input => Dict{Symbol, Any}(:name => "sort"))],
        :stop_reason => "tool_use",
        :usage => Dict{Symbol, Any}(:input_tokens => 200_000, :output_tokens => 3))
    schema = PT.TestEchoAnthropicSchema(; response, status = 200)
    old_budget = Wink.CONFIG.context_budget
    old_rounds = Wink.CONFIG.max_rounds
    chat = Wink.new_chat()
    try
        Wink.CONFIG.context_budget = 1_000
        Wink.CONFIG.max_rounds = 4
        Wink.run_turn!(chat, "docs for sort?"; schema, io = devnull)
    finally
        Wink.CONFIG.context_budget = old_budget
        Wink.CONFIG.max_rounds = old_rounds
    end
    results = [m for m in chat.history if Wink.is_tool_result_msg(m)]
    @test length(results) == 4
    @test occursin("elided", results[1].content)      # stale → folded
    @test !occursin("elided", results[end].content)   # recent → intact

    # budget 0 disables the trigger entirely
    schema2 = PT.TestEchoAnthropicSchema(; response, status = 200)
    chat2 = Wink.new_chat()
    try
        Wink.CONFIG.context_budget = 0
        Wink.CONFIG.max_rounds = 4
        Wink.run_turn!(chat2, "docs for sort?"; schema = schema2, io = devnull)
    finally
        Wink.CONFIG.context_budget = old_budget
        Wink.CONFIG.max_rounds = old_rounds
    end
    @test !any(m -> Wink.is_tool_result_msg(m) && occursin("elided", m.content),
        chat2.history)
end
