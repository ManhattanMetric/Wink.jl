@testset "tools" begin
    tm = Wink.build_tool_map()
    for name in ("list_methods", "get_source", "get_ir", "get_doc", "type_info",
        "module_info", "where_defined", "read_file", "list_names")
        @test haskey(tm, name)
        @test tm[name] isa PT.Tool
    end
    # descriptions survive well past the Anthropic path's 100-char default cut
    @test length(tm["get_source"].description) > 150

    out = Wink.tool_get_source("TestPkg.combine(::Int, ::Int)")
    @test occursin("a + b", out)
    @test occursin("# source:", out)

    out = Wink.tool_get_source("TestPkg.combine")   # multi-method, no types
    @test occursin("2 methods", out)

    out = Wink.tool_get_source("TestPkg.greet")     # bare name, single method
    @test occursin("Hello", out)

    out = Wink.tool_list_methods("TestPkg.combine")
    @test occursin("[1]", out)
    @test occursin("[2]", out)

    out = Wink.tool_get_ir("TestPkg.twice(::Int)", "llvm")
    @test !startswith(out, "ERROR")
    @test !isempty(strip(out))
    out = Wink.tool_get_ir("TestPkg.twice", "llvm")
    @test occursin("requires argument types", out)

    @test occursin("friendly", Wink.tool_get_doc("TestPkg.greet"))
    @test occursin("ultimate question", Wink.tool_get_doc("TestPkg.ANSWER"))
    @test occursin("No documentation found", Wink.tool_get_doc("TestPkg.undocumented"))

    @test occursin("fields", Wink.tool_type_info("TestPkg.Point"))
    @test startswith(Wink.tool_type_info("NoSuchType_XYZ"), "ERROR")

    @test occursin("greet", Wink.tool_module_info("TestPkg", ""))
    @test startswith(Wink.tool_module_info("TestPkg.greet", ""), "ERROR")

    wd = Wink.tool_where_defined("TestPkg.greet(::String)")
    @test occursin("TestPkg.jl:", wd)

    file = String(first(split(wd, ':')))
    rf = Wink.tool_read_file(file, "", "")
    @test occursin("greet", rf)
    @test occursin("    1: ", rf)
    rf2 = Wink.tool_read_file(file, "1", "3")
    @test count('\n', rf2) <= 3
    @test startswith(Wink.tool_read_file("/etc/hosts", "", ""), "ERROR")

    @test occursin("greet", Wink.tool_list_names("friendly greeting"))

    long = repeat("x", 10_000)
    t = Wink.truncate_output(long; limit = 1000)
    @test length(t) < 1600
    @test occursin("truncated", t)
end
