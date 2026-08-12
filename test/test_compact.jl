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

@testset "compact distill" begin
    # a plausible flattened history: system, then turns of user/ai pairs
    mkhist(n) = begin
        h = PT.AbstractMessage[PT.SystemMessage("sys"), PT.UserMessage("build a blog")]
        while length(h) < n
            push!(h, PT.AIMessage("→ eval_code(...)"))
            push!(h, PT.UserMessage("<tool_result name=\"eval_code\">\nok\n</tool_result>"))
        end
        h
    end

    # span selection: cut lands so the first kept message is an AIMessage
    h = mkhist(14)
    r = Wink.distill_span(h; keep_recent = 6)
    @test r == 2:8
    @test h[first(r) - 1] isa PT.SystemMessage
    @test h[last(r) + 1] isa PT.AIMessage

    # too-small histories refuse to distill (this is also the cooldown)
    @test Wink.distill_span(mkhist(8); keep_recent = 6) === nothing
    @test Wink.distill_span(PT.AbstractMessage[PT.SystemMessage("sys")]) === nothing

    # rendering labels roles
    txt = Wink.render_span(h, 2:3)
    @test occursin("[user]\nbuild a blog", txt)
    @test occursin("[assistant]", txt)

    # a successful distillation splices the brief in place of the span
    brief_schema = PT.TestEchoAnthropicSchema(;
        response = anthropic_text_response(
            "Goal: a blog. Done: nothing durable. Decisions: plain repo over " *
            "package. Open: scaffold layout."),
        status = 200)
    chat = Wink.Chat(mkhist(14), Wink.build_tool_map(), 0, 0)
    @test Wink.distill_history!(chat; schema = brief_schema, io = devnull)
    @test length(chat.history) == 14 - 7 + 1
    @test chat.history[1] isa PT.SystemMessage
    @test occursin("session brief", chat.history[2].content)
    @test occursin("plain repo over package", chat.history[2].content)
    @test chat.history[3] isa PT.AIMessage      # alternation survives the splice
    @test chat.tokens_in > 0                    # the distill call is tallied

    # an empty brief leaves the history untouched
    empty_schema = PT.TestEchoAnthropicSchema(;
        response = anthropic_text_response(""), status = 200)
    chat2 = Wink.Chat(mkhist(14), Wink.build_tool_map(), 0, 0)
    @test !Wink.distill_history!(chat2; schema = empty_schema, io = devnull)
    @test length(chat2.history) == 14

    # escalation policy: a productive fold defers distillation to a later round
    over = PT.AIMessage(; content = "x", tokens = (200_000, 1))
    long = repeat("x", 300)
    h3 = mkhist(14)
    h3[4] = PT.UserMessage("<tool_result name=\"get_doc\">\n$long\n</tool_result>")
    chat3 = Wink.Chat(h3, Wink.build_tool_map(), 0, 0)
    old_budget = Wink.CONFIG.context_budget
    try
        Wink.CONFIG.context_budget = 1_000
        Wink.maybe_compact!(chat3, over, devnull; schema = brief_schema)
        @test occursin("elided", chat3.history[4].content)      # folded...
        @test length(chat3.history) == 14                       # ...not distilled
        Wink.maybe_compact!(chat3, over, devnull; schema = brief_schema)
        @test any(m -> m isa PT.UserMessage && occursin("session brief", m.content),
            chat3.history)                                      # now distilled
        @test length(chat3.history) < 14
    finally
        Wink.CONFIG.context_budget = old_budget
    end
end

@testset "compact mine" begin
    mkhist(n) = begin
        h = PT.AbstractMessage[PT.SystemMessage("sys"), PT.UserMessage("build a blog")]
        while length(h) < n
            push!(h, PT.AIMessage("→ eval_code(...)"))
            push!(h, PT.UserMessage("<tool_result name=\"eval_code\">\nok\n</tool_result>"))
        end
        h
    end
    propose_response(name, def, why) = Dict{Symbol, Any}(
        :content => Dict{Symbol, Any}[Dict{Symbol, Any}(:type => "tool_use",
            :id => "toolu_9", :name => "propose_abstraction",
            :input => Dict{Symbol, Any}(:name => name, :definition => def,
                :rationale => why))],
        :stop_reason => "tool_use",
        :usage => Dict{Symbol, Any}(:input_tokens => 5, :output_tokens => 2))

    good_def = """
    \"\"\"
        wink_mined_double(x)

    Double `x` (a test-mined abstraction).
    \"\"\"
    wink_mined_double(x) = 2x
    """
    schema = PT.TestEchoAnthropicSchema(;
        response = propose_response("wink_mined_double", good_def,
            "spelled out twice in the span"),
        status = 200)

    old_confirm = Wink.CONFIG.confirm
    old_auto = Wink.CONFIG.autoeval
    seen = Ref{Any}(nothing)
    chat = Wink.Chat(mkhist(14), Wink.build_tool_map(), 0, 0)
    try
        Wink.CONFIG.autoeval = false
        Wink.CONFIG.confirm = (k, t) -> (seen[] = (k, t); true)
        @test Wink.mine_abstractions!(chat; schema, io = devnull) ==
              ["wink_mined_double"]
        @test Wink.eval_in_main("wink_mined_double(3)").value_repr == "6"
        @test seen[][1] === :abstract
        @test occursin("spelled out twice", seen[][2])   # rationale shown
        @test occursin("wink_mined_double(x) = 2x", seen[][2])
        @test chat.tokens_in > 0                         # the mining call is tallied

        # an already-defined name is never re-proposed to the user
        seen[] = nothing
        @test Wink.mine_abstractions!(chat; schema, io = devnull) == String[]
        @test seen[] === nothing

        # declined proposals are not installed
        Wink.CONFIG.confirm = (k, t) -> false
        declined = PT.TestEchoAnthropicSchema(;
            response = propose_response("wink_mined_never",
                "wink_mined_never() = 1", "x"), status = 200)
        @test Wink.mine_abstractions!(chat; schema = declined, io = devnull) ==
              String[]
        @test !isdefined(Main, :wink_mined_never)

        # definitions that fail to evaluate are skipped, not installed
        Wink.CONFIG.confirm = (k, t) -> true
        broken = PT.TestEchoAnthropicSchema(;
            response = propose_response("wink_mined_broken", "function ((", "x"),
            status = 200)
        @test Wink.mine_abstractions!(chat; schema = broken, io = devnull) ==
              String[]
        @test !isdefined(Main, :wink_mined_broken)

        # a plain-text reply ("none") proposes nothing
        none = PT.TestEchoAnthropicSchema(;
            response = anthropic_text_response("none"), status = 200)
        @test Wink.mine_abstractions!(chat; schema = none, io = devnull) == String[]

        # too-small spans never spend a model call
        small = Wink.Chat(PT.AbstractMessage[PT.SystemMessage("s"),
                PT.UserMessage("q")], Wink.build_tool_map(), 0, 0)
        @test Wink.mine_abstractions!(small; schema, io = devnull) == String[]
    finally
        Wink.CONFIG.confirm = old_confirm
        Wink.CONFIG.autoeval = old_auto
    end

    # promoted names are handed to the distill pass, which sees them in its input
    brief_schema = PT.TestEchoAnthropicSchema(;
        response = anthropic_text_response(
            "Goal: g. Done: d. Decisions: -. Open: -."),
        status = 200)
    chat2 = Wink.Chat(mkhist(14), Wink.build_tool_map(), 0, 0)
    @test Wink.distill_history!(chat2; schema = brief_schema, io = devnull,
        promoted = ["wink_mined_double"])
    @test occursin("wink_mined_double", string(brief_schema.inputs))

    # full ladder via maybe_compact!: nothing to fold → the span is mined and
    # the definition lands in Main; the echo replays the same tool call to the
    # distill pass too, which therefore yields no brief text and leaves the
    # history intact — mining and distillation fail independently
    over = PT.AIMessage(; content = "x", tokens = (200_000, 1))
    ladder = PT.TestEchoAnthropicSchema(;
        response = propose_response("wink_mined_ladder",
            "\"\"\"\n    wink_mined_ladder()\n\nTest.\n\"\"\"\nwink_mined_ladder() = :ok",
            "recurred"),
        status = 200)
    chat3 = Wink.Chat(mkhist(14), Wink.build_tool_map(), 0, 0)
    old_budget = Wink.CONFIG.context_budget
    try
        Wink.CONFIG.autoeval = true
        Wink.CONFIG.context_budget = 1_000
        Wink.maybe_compact!(chat3, over, devnull; schema = ladder)
        @test Wink.eval_in_main("wink_mined_ladder()").value_repr == ":ok"
        @test length(chat3.history) == 14
    finally
        Wink.CONFIG.autoeval = old_auto
        Wink.CONFIG.context_budget = old_budget
    end
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
