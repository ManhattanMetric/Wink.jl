# Agent-loop tests against PromptingTools' TestEcho mock schemas — no network.
# The echo schema replays the same canned response on every call and records
# the *rendered* payload in `.inputs`, which lets us regression-test the
# flattening workaround for PT's Anthropic renderer dropping tool messages.

# NOTE: the content vector must be eltype Dict{Symbol,Any} (like real parsed
# JSON). With an overly-concrete eltype, PT's tools_array comprehension infers
# `Union{}` for empty results and its AIToolRequest construction TypeErrors.
anthropic_text_response(text) = Dict{Symbol, Any}(
    :content => Dict{Symbol, Any}[Dict{Symbol, Any}(:type => "text", :text => text)],
    :stop_reason => "end_turn",
    :usage => Dict{Symbol, Any}(:input_tokens => 10, :output_tokens => 4))

anthropic_tool_response(name, input) = Dict{Symbol, Any}(
    :content => Dict{Symbol, Any}[Dict{Symbol, Any}(:type => "tool_use",
        :id => "toolu_1", :name => name, :input => input)],
    :stop_reason => "tool_use",
    :usage => Dict{Symbol, Any}(:input_tokens => 7, :output_tokens => 3))

@testset "agent" begin
    # --- plain text turn ---
    schema = PT.TestEchoAnthropicSchema(;
        response = anthropic_text_response("All good."), status = 200)
    chat = Wink.new_chat()
    ans = Wink.run_turn!(chat, "hello?"; schema, io = devnull)
    @test ans == "All good."
    @test chat.history[end] isa PT.AIMessage
    @test chat.tokens_in == 10
    @test chat.tokens_out == 4

    # --- tool-call turns ---
    # The echo always answers with the same tool_use, so the loop runs to the
    # round cap, executes the (real) tool each round, then forces a final call.
    schema2 = PT.TestEchoAnthropicSchema(;
        response = anthropic_tool_response("get_doc",
            Dict{Symbol, Any}(:name => "TestPkg.greet")),
        status = 200)
    old_rounds = Wink.CONFIG.max_rounds
    chat2 = Wink.new_chat()
    try
        Wink.CONFIG.max_rounds = 2
        ans2 = Wink.run_turn!(chat2, "docs for greet?"; schema = schema2, io = devnull)
        @test ans2 == ""   # forced-final echo is another tool_use; content is empty
    finally
        Wink.CONFIG.max_rounds = old_rounds
    end
    # the history is native: the assistant's request and each result are
    # first-class tool messages
    @test any(m -> m isa PT.AIToolRequest && !isempty(m.tool_calls), chat2.history)
    tool_results = [m for m in chat2.history if m isa PT.ToolMessage]
    @test !isempty(tool_results)
    @test all(m -> m.name == "get_doc", tool_results)
    # ...and the real get_doc tool actually ran
    @test any(m -> occursin("friendly", string(m.content)), tool_results)
    # round-cap wrap-up message was appended
    @test any(m -> m isa PT.UserMessage && occursin("Tool budget exhausted", m.content),
        chat2.history)
    # REGRESSION (renderer gap): for an Anthropic schema the RENDERED payload
    # is the flattened projection — the tool result appears as ordinary text
    # there even though the stored history is native
    @test occursin("tool_result", string(schema2.inputs))

    # --- gated tool: denial feeds DECLINED back into the conversation ---
    schema3 = PT.TestEchoAnthropicSchema(;
        response = anthropic_tool_response("eval_code", Dict{Symbol, Any}(:code => "1+1")),
        status = 200)
    old_confirm = Wink.CONFIG.confirm
    chat3 = Wink.new_chat()
    try
        Wink.CONFIG.autoeval = false
        Wink.CONFIG.confirm = (k, t) -> false
        Wink.CONFIG.max_rounds = 1
        Wink.run_turn!(chat3, "run 1+1"; schema = schema3, io = devnull)
    finally
        Wink.CONFIG.confirm = old_confirm
        Wink.CONFIG.max_rounds = old_rounds
    end
    @test any(m -> m isa PT.ToolMessage && occursin("DECLINED", string(m.content)),
        chat3.history)

    # --- gated tool: approval executes in the real Main ---
    schema4 = PT.TestEchoAnthropicSchema(;
        response = anthropic_tool_response("eval_code",
            Dict{Symbol, Any}(:code => "WINK_AGENT_SENTINEL = 41 + 1")),
        status = 200)
    chat4 = Wink.new_chat()
    try
        Wink.CONFIG.confirm = (k, t) -> true
        Wink.CONFIG.max_rounds = 1
        Wink.run_turn!(chat4, "set the sentinel"; schema = schema4, io = devnull)
    finally
        Wink.CONFIG.confirm = old_confirm
        Wink.CONFIG.max_rounds = old_rounds
    end
    @test Main.WINK_AGENT_SENTINEL == 42

    # --- recovery reflex: repeated eval failures escalate to an introspection
    # demand. The echo replays the same failing eval every round (a blind retry
    # streak); round 1 gets only the api_hint, rounds 2..3 add the system note.
    schema6 = PT.TestEchoAnthropicSchema(;
        response = anthropic_tool_response("eval_code",
            Dict{Symbol, Any}(:code => "system(\"ls\")")),
        status = 200)
    chat6 = Wink.new_chat()
    old_auto = Wink.CONFIG.autoeval
    try
        Wink.CONFIG.autoeval = true
        Wink.CONFIG.max_rounds = 3
        Wink.run_turn!(chat6, "list the files here"; schema = schema6, io = devnull)
    finally
        Wink.CONFIG.autoeval = old_auto
        Wink.CONFIG.max_rounds = old_rounds
    end
    @test any(m -> m isa PT.ToolMessage &&
                   occursin("hint: `system` does not exist", string(m.content)),
        chat6.history)
    users6 = [m for m in chat6.history if m isa PT.UserMessage]
    @test count(m -> occursin("consecutive eval_code failures", m.content),
        users6) == 2
    @test any(m -> occursin("3 consecutive", m.content), users6)

    # streak accounting: only failed evals extend it, anything else resets
    @test Wink.failed_eval_output("ERROR: MethodError: no method matching f()")
    @test Wink.failed_eval_output("parse error: unexpected `)`")
    @test Wink.failed_eval_output("TOOL ERROR: boom")
    @test !Wink.failed_eval_output("value: 42")
    @test !Wink.failed_eval_output(Wink.DECLINED_MSG)

    # --- native path: on an OpenAI-compatible schema the tool exchange renders
    # natively (assistant tool_calls + role "tool"), with no flattened text ---
    openai_response = Dict(
        :choices => [Dict(:message => Dict(:content => nothing,
                :tool_calls => [Dict(:id => "call_1", :type => "function",
                    :function => Dict(:name => "get_doc",
                        :arguments => "{\"name\": \"TestPkg.greet\"}"))]),
            :finish_reason => "tool_calls")],
        :usage => Dict(:prompt_tokens => 7, :completion_tokens => 3))
    schema7 = PT.TestEchoOpenAISchema(; response = openai_response, status = 200)
    chat7 = Wink.new_chat()
    try
        Wink.CONFIG.max_rounds = 2
        Wink.run_turn!(chat7, "docs for greet?"; schema = schema7, io = devnull)
    finally
        Wink.CONFIG.max_rounds = old_rounds
    end
    @test any(m -> m isa PT.ToolMessage, chat7.history)
    rendered = string(schema7.inputs)
    @test occursin("tool_call_id", rendered)         # native tool-result message
    @test occursin("tool_calls", rendered)           # native assistant request
    @test !occursin("<tool_result", rendered)        # no flattened projection

    # --- thinking indicator: non-TTY path is a plain status line, result passes ---
    @test Wink.with_thinking(() -> 42, devnull) == 42

    # --- textual tool-call detection (imitation of the flattened transcript) ---
    tools = ("get_doc", "get_source", "eval_code")
    @test Wink.textual_tool_call("Let me check:\n→ get_doc(name=\"sort\")", tools) ==
          "get_doc"
    @test Wink.textual_tool_call("-> get_source(signature=\"foo\")", tools) ==
          "get_source"
    @test Wink.textual_tool_call("get_doc(name=\"sort\")", tools) == "get_doc"
    @test Wink.textual_tool_call("call notatool(x=1)", tools) === nothing
    @test Wink.textual_tool_call("→ get_doc(...) told me sort sorts.", tools) === nothing
    @test Wink.textual_tool_call("The answer is 42.", tools) === nothing
    @test Wink.textual_tool_call("", tools) === nothing

    # multi-line imitation: a code payload spilling across lines (the shape a
    # live session produced — prose, then a fake call running to message end)
    multiline = """
    I'll start by writing the Models.jl file.

    → eval_code(code="open(\\"src/Models.jl\\", \\"w\\") do io
        write(io, \\"\\"\\"
        module Models
        end
        \\"\\"\\")
    end")"""
    @test Wink.textual_tool_call(multiline, tools) == "eval_code"
    # ...including when the parens never close before the message ends
    @test Wink.textual_tool_call("→ eval_code(code=\"f(x\n= 1\n", tools) ==
          "eval_code"
    # several fake calls with prose between: the last one is the intent
    interleaved = "→ get_doc(name=\"sort\")\nActually, wrong tool.\n" *
                  "→ get_source(signature=\"sort(::Vector{Int})\n    more args\")"
    @test Wink.textual_tool_call(interleaved, tools) == "get_source"
    # a closed call followed by prose is still a mention, even multi-line
    @test Wink.textual_tool_call("→ eval_code(code=\"1+1\")\nwhich returned 2.",
        tools) === nothing

    # --- trailing-code-block detection (the "I'll run this now" + block +
    # end-of-turn shape a live demo produced four times) ---
    @test Wink.trailing_code_block("I'll evaluate this now.\n```julia\nx = 1\n```")
    @test Wink.trailing_code_block("Setup:\n```\nusing Pkg\nPkg.status()\n```\n")
    @test !Wink.trailing_code_block("```julia\nx = 1\n```\nThat's how you'd do it.")
    @test !Wink.trailing_code_block("Use this pattern:\n```bash\nls -la\n```")
    @test !Wink.trailing_code_block("The answer is 42.")
    @test !Wink.trailing_code_block("")

    # a reply ending in a code block is nudged toward execution, once per round
    schema_cb = PT.TestEchoAnthropicSchema(;
        response = anthropic_text_response(
            "I'll define the type now.\n```julia\nstruct P end\n```"),
        status = 200)
    chat_cb = Wink.new_chat()
    try
        Wink.CONFIG.max_rounds = 2
        Wink.run_turn!(chat_cb, "define P"; schema = schema_cb, io = devnull)
    finally
        Wink.CONFIG.max_rounds = old_rounds
    end
    @test count(m -> m isa PT.UserMessage &&
                     occursin("ended with a Julia code block", m.content),
        chat_cb.history) == 2

    # A reply ending in a textual call is nudged instead of accepted as final:
    # the echo replays the same text every round, so the loop nudges once per
    # round and then falls through to the forced-final path.
    schema5 = PT.TestEchoAnthropicSchema(;
        response = anthropic_text_response(
            "No docstring — let me look:\n→ get_source(signature=\"Wink.build_index!\")"),
        status = 200)
    chat5 = Wink.new_chat()
    try
        Wink.CONFIG.max_rounds = 2
        Wink.run_turn!(chat5, "where is the index stored?"; schema = schema5,
            io = devnull)
    finally
        Wink.CONFIG.max_rounds = old_rounds
    end
    @test count(m -> m isa PT.UserMessage && occursin("written as text", m.content),
        chat5.history) == 2

    # --- system prompt content ---
    sp = Wink.system_prompt()
    @test occursin(string(VERSION), sp)
    @test occursin("ground truth", sp)
    @test occursin("TestPkg", sp)   # loaded-module summary
    @test occursin("vocabulary of abstractions", sp)
    @test occursin("weakest tool", sp)

    # --- global chat plumbing ---
    Wink.reset!()
    @test Wink.CHAT[] === nothing
    c = Wink.current_chat()
    @test c isa Wink.Chat
    @test Wink.CHAT[] === c
    Wink.reset!()
end

@testset "standing instructions" begin
    old_instr = Wink.CONFIG.instructions
    old_global = Wink.GLOBAL_INSTRUCTIONS_FILE[]
    try
        # no layers configured -> no section at all
        Wink.CONFIG.instructions = ""
        Wink.GLOBAL_INSTRUCTIONS_FILE[] = joinpath(mktempdir(), "absent.md")
        @test !occursin("Standing instructions", Wink.system_prompt())

        # session layer (configure!)
        Wink.configure!(instructions = "Always explain type-stability implications.")
        sp = Wink.system_prompt()
        @test occursin("Standing instructions", sp)
        @test occursin("Session instructions", sp)
        @test occursin("type-stability implications", sp)

        # global-file layer (via the test hook)
        Wink.CONFIG.instructions = ""
        mktempdir() do dir
            gf = joinpath(dir, "wink.md")
            write(gf, "Prefer StaticArrays for small fixed-size data.")
            Wink.GLOBAL_INSTRUCTIONS_FILE[] = gf
            sp = Wink.system_prompt()
            @test occursin("Global instructions", sp)
            @test occursin("StaticArrays", sp)

            # oversized files are truncated, not dumped wholesale
            write(gf, repeat("x", 20_000))
            @test occursin("[instructions truncated]", Wink.system_prompt())
        end
        Wink.GLOBAL_INSTRUCTIONS_FILE[] = joinpath(mktempdir(), "absent.md")

        # project layer (.wink.md next to the active Project.toml)
        proj = Base.active_project()
        if proj !== nothing
            pf = joinpath(dirname(proj), ".wink.md")
            write(pf, "This project uses BlueStyle formatting.")
            try
                sp = Wink.system_prompt()
                @test occursin("Project instructions", sp)
                @test occursin("BlueStyle", sp)
            finally
                rm(pf; force = true)
            end
        end
    finally
        Wink.CONFIG.instructions = old_instr
        Wink.GLOBAL_INSTRUCTIONS_FILE[] = old_global
    end
end
