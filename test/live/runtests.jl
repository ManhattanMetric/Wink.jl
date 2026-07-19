# Opt-in live tests against a real provider. Run with:
#
#     WINK_LIVE_TESTS=true julia --project -e 'using Pkg; Pkg.test()'
#
# Requires ANTHROPIC_API_KEY or OPENAI_API_KEY in the environment (or a running
# local Ollama server). Never runs on CI.

@testset "live provider round-trip" begin
    Wink.reset!()
    old_auto = Wink.CONFIG.autoeval
    old_rounds = Wink.CONFIG.max_rounds
    try
        Wink.CONFIG.autoeval = true   # non-interactive session: skip the y/N gate
        Wink.CONFIG.max_rounds = 6
        ans = Wink.ask("Call the get_doc tool for TestPkg.greet and then tell me " *
                       "in one short sentence what greet does.")
        @test !isempty(strip(ans))
        @test occursin(r"greet"i, ans)
        chat = Wink.CHAT[]
        @test any(m -> m isa PT.UserMessage && occursin("<tool_result", m.content),
            chat.history)
    finally
        Wink.CONFIG.autoeval = old_auto
        Wink.CONFIG.max_rounds = old_rounds
        Wink.reset!()
    end

    # Semantic search (embeds live when an embedding provider is reachable;
    # otherwise exercises the keyword fallback — either way, no ERROR).
    out = Wink.tool_search_docs("greeting people by name", "5")
    @test !startswith(out, "ERROR")
end
