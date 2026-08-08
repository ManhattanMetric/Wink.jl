@testset "shell" begin
    r = Wink.run_shell_command("echo hello")
    @test r.exitcode == 0
    @test occursin("hello", r.output)

    # stderr is merged into output; nonzero exit is reported, not thrown
    r = Wink.run_shell_command("echo oops >&2; exit 3")
    @test r.exitcode == 3
    @test occursin("oops", r.output)

    # one string through sh -c: pipes and && work
    r = Wink.run_shell_command("printf 'a\\nb\\nc\\n' | wc -l && echo done")
    @test r.exitcode == 0
    @test occursin("3", r.output)
    @test occursin("done", r.output)

    # dir sets the working directory of the child process
    mktempdir() do d
        r = Wink.run_shell_command("pwd"; dir = d)
        @test r.exitcode == 0
        @test occursin(basename(d), r.output)
    end

    @test Wink.format_shell_result(0, "fine\n") == "fine\n"
    @test Wink.format_shell_result(0, "  \n") == "(no output)"
    s = Wink.format_shell_result(2, "bad")
    @test occursin("exit code: 2", s)
    @test occursin("bad", s)
end

@testset "shell gate" begin
    old_confirm = Wink.CONFIG.confirm
    old_auto = Wink.CONFIG.autoeval
    try
        Wink.CONFIG.autoeval = false
        Wink.CONFIG.confirm = (k, t) -> false
        @test Wink.tool_run_shell("echo nope", "") == Wink.DECLINED_MSG

        seen = Ref{Any}(nothing)
        Wink.CONFIG.confirm = (k, t) -> (seen[] = (k, t); true)
        @test occursin("gated", Wink.tool_run_shell("echo gated", ""))
        @test seen[] == (:shell, "echo gated")

        # the confirmation text names the working directory when one is given
        mktempdir() do d
            out = Wink.tool_run_shell("echo where", d)
            @test occursin("where", out)
            @test seen[][1] === :shell
            @test occursin(d, seen[][2])
        end

        Wink.CONFIG.confirm = (k, t) -> error("must not be called")
        Wink.CONFIG.autoeval = true
        @test occursin("auto", Wink.tool_run_shell("echo auto", ""))

        # bad inputs come back as tool errors, without reaching the shell
        @test occursin("ERROR", Wink.tool_run_shell("", ""))
        @test occursin("directory not found",
            Wink.tool_run_shell("echo x", "/no/such/dir/wink"))
    finally
        Wink.CONFIG.confirm = old_confirm
        Wink.CONFIG.autoeval = old_auto
    end

    # registered as a model-facing tool
    @test Wink.has_tool("run_shell")
    tm = Wink.build_tool_map()
    @test haskey(tm, "run_shell")
end
