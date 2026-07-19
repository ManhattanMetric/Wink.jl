@testset "repl helpers" begin
    buf = IOBuffer()

    Wink.handle_meta_command(":config"; io = buf)
    @test occursin("chat_model", String(take!(buf)))

    Wink.handle_meta_command(":autoeval on"; io = buf)
    @test Wink.CONFIG.autoeval
    Wink.handle_meta_command(":autoeval off"; io = buf)
    @test !Wink.CONFIG.autoeval
    take!(buf)
    Wink.handle_meta_command(":autoeval"; io = buf)
    @test occursin("usage", String(take!(buf)))

    Wink.handle_meta_command(":help"; io = buf)
    @test occursin(":reset", String(take!(buf)))

    Wink.handle_meta_command(":bogus"; io = buf)
    @test occursin("unknown command", String(take!(buf)))

    Wink.handle_meta_command(":model"; io = buf)
    @test occursin(Wink.CONFIG.chat_model, String(take!(buf)))

    old_model = Wink.CONFIG.chat_model
    try
        Wink.handle_meta_command(":model claude-haiku-4-5"; io = buf)
        @test Wink.CONFIG.chat_model == "claude-haiku-4-5"
        @test occursin("claude-haiku-4-5", String(take!(buf)))
    finally
        Wink.configure!(chat_model = old_model)
    end

    Wink.current_chat()
    Wink.handle_meta_command(":reset"; io = buf)
    @test Wink.CHAT[] === nothing
    take!(buf)

    Wink.handle_meta_command(":history"; io = buf)
    @test occursin("no conversation", String(take!(buf)))

    Wink.print_answer("**bold** and `code`"; io = buf)
    @test !isempty(String(take!(buf)))
    Wink.print_answer(""; io = buf)
    @test isempty(String(take!(buf)))

    @test Wink.handle_ai_input("   ") === nothing
end
