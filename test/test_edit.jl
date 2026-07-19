@testset "edit" begin
    old_confirm = Wink.CONFIG.confirm
    mktempdir() do dir
        src = joinpath(dir, "editme.jl")
        write(src, """
        module WinkEditTest
        bump(x) = x + 1
        helper() = 1
        end
        """)
        push!(Wink.EXTRA_EDIT_ROOTS, realpath(dir))
        try
            Wink.CONFIG.confirm = (k, t) -> true

            out = Wink.tool_edit_file(src, "nope_not_here", "x")
            @test occursin("not found", out)

            out = Wink.tool_edit_file(src, "1", "2")
            @test occursin("occurs", out)   # ambiguous match refused

            out = Wink.tool_edit_file(src, "bump(x) = x + 1", "bump(x) = x + 2")
            @test startswith(out, "OK") || startswith(out, "EDITED")
            @test occursin("x + 2", read(src, String))
            @test !isempty(Wink.backups())

            out = Wink.tool_edit_file(src, "helper() = 1", "helper() = 1")
            @test startswith(out, "ERROR")   # identical old/new refused

            # the confirmation gate sees a diff-style preview and can decline
            seen = Ref("")
            Wink.CONFIG.confirm = (k, t) -> (seen[] = t; false)
            out = Wink.tool_edit_file(src, "helper() = 1", "helper() = 2")
            @test out == Wink.DECLINED_MSG
            @test occursin("- helper() = 1", seen[])
            @test occursin("+ helper() = 2", seen[])
            @test occursin("helper() = 1", read(src, String))   # untouched
        finally
            Wink.CONFIG.confirm = old_confirm
            pop!(Wink.EXTRA_EDIT_ROOTS)
        end

        # outside the perimeter now that the root is popped
        outside = joinpath(dir, "outside.jl")
        write(outside, "f() = 1\n")
        @test occursin("outside the editable perimeter",
            Wink.tool_edit_file(outside, "f() = 1", "f() = 2"))

        # wrong extension
        toml = joinpath(dir, "some.toml")
        write(toml, "a = 1\n")
        @test occursin("only .jl files", Wink.tool_edit_file(toml, "a = 1", "a = 2"))
    end

    @test occursin("file not found",
        Wink.tool_edit_file("/no/such/wink_file.jl", "a", "b"))
end

# End-to-end: edit a Revise-tracked file and observe the redefinition live.
# Watcher timing can be flaky on shared CI runners — skippable via env.
if get(ENV, "WINK_SKIP_REVISE_E2E", "") != "true"
    @testset "edit + Revise live reload" begin
        old_confirm = Wink.CONFIG.confirm
        mktempdir() do dir
            f = joinpath(dir, "wink_revise_target.jl")
            write(f, "wink_revise_fn() = :original\n")
            Wink.Revise.includet(Main, f)
            @test Base.invokelatest(Main.wink_revise_fn) === :original
            push!(Wink.EXTRA_EDIT_ROOTS, realpath(dir))
            try
                Wink.CONFIG.confirm = (k, t) -> true
                out = Wink.tool_edit_file(f, ":original", ":edited")
                @test startswith(out, "OK") || startswith(out, "EDITED")
                ok = false
                deadline = time() + 10
                while time() < deadline
                    try
                        Wink.Revise.revise()
                    catch
                    end
                    if Base.invokelatest(Main.wink_revise_fn) === :edited
                        ok = true
                        break
                    end
                    sleep(0.2)
                end
                @test ok
            finally
                Wink.CONFIG.confirm = old_confirm
                pop!(Wink.EXTRA_EDIT_ROOTS)
            end
        end
    end
end
