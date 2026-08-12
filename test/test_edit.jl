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

@testset "write_file" begin
    old_confirm = Wink.CONFIG.confirm
    mktempdir() do dir
        push!(Wink.EXTRA_EDIT_ROOTS, realpath(dir))
        try
            # the gate sees the path and full content under kind :write
            seen = Ref{Any}(nothing)
            Wink.CONFIG.confirm = (k, t) -> (seen[] = (k, t); true)
            target = joinpath(dir, "src", "Models.jl")
            out = Wink.tool_write_file(target, "module Models\nend\n")
            @test startswith(out, "OK")
            @test occursin("2 lines", out)
            @test read(target, String) == "module Models\nend\n"   # mkpath worked
            @test seen[][1] === :write
            @test occursin("Models.jl", seen[][2])
            @test occursin("module Models", seen[][2])

            # refuses to overwrite, pointing at edit_file
            out = Wink.tool_write_file(target, "something else")
            @test occursin("already exists", out)
            @test occursin("edit_file", out)
            @test read(target, String) == "module Models\nend\n"

            # non-source text types are allowed (assets need creating too)
            @test startswith(Wink.tool_write_file(joinpath(dir, "static", "app.css"),
                    "body { margin: 0 }\n"), "OK")

            # disallowed extension
            @test occursin("file types",
                Wink.tool_write_file(joinpath(dir, "blob.bin"), "x"))

            # empty content refused
            @test occursin("content is empty",
                Wink.tool_write_file(joinpath(dir, "empty.jl"), "  \n"))

            # declined → nothing written
            Wink.CONFIG.confirm = (k, t) -> false
            declined = joinpath(dir, "declined.jl")
            @test Wink.tool_write_file(declined, "f() = 1\n") == Wink.DECLINED_MSG
            @test !isfile(declined)
        finally
            Wink.CONFIG.confirm = old_confirm
            pop!(Wink.EXTRA_EDIT_ROOTS)
        end

        # outside the perimeter once the root is popped
        Wink.CONFIG.confirm = (k, t) -> true
        try
            @test occursin("outside the editable perimeter",
                Wink.tool_write_file(joinpath(dir, "outside.jl"), "f() = 1\n"))
        finally
            Wink.CONFIG.confirm = old_confirm
        end
    end

    # registered as a model-facing tool
    @test Wink.has_tool("write_file")
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
