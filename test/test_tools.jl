@testset "tools" begin
    tm = Wink.build_tool_map()
    for name in ("list_methods", "methods_with", "get_source", "get_ir", "get_doc",
        "type_info", "module_info", "list_variables", "where_defined", "read_file",
        "list_names", "search_packages")
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

    mw = Wink.tool_methods_with("String", "TestPkg", "")
    @test occursin("with an argument of type `String`", mw)
    @test occursin("greet", mw)
    @test occursin("combine", mw)
    mw = Wink.tool_methods_with("TestPkg.Point", "TestPkg", "")
    @test occursin("simulate", mw)
    @test startswith(Wink.tool_methods_with("NoSuchType_XYZ", "", ""), "ERROR")
    @test startswith(Wink.tool_methods_with("String", "TestPkg.greet", ""), "ERROR")

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

    @eval Main wink_test_var_xyz = collect(1:100)
    lv = Wink.tool_list_variables("", "wink_test_var")
    @test occursin("wink_test_var_xyz", lv)
    @test occursin("100-element", lv)
    @test occursin("greet", Wink.tool_list_variables("TestPkg", ""))
    @test !occursin("greet", Wink.tool_list_variables("TestPkg", "Point"))
    @test startswith(Wink.tool_list_variables("TestPkg.greet", ""), "ERROR")

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

    if !isempty(Wink.Pkg.Registry.reachable_registries())
        sp = Wink.tool_search_packages("DataFrames")
        @test occursin(r"DataFrames v\d", sp)
        @test occursin("JuliaData/DataFrames.jl", sp)
        # exact match ranks ahead of prefix matches like DataFramesMeta
        @test first(findfirst("  DataFrames v", sp)) <
              first(findfirst("DataFramesMeta", sp))
        @test occursin("loaded", Wink.tool_search_packages("PromptingTools"))
        @test !occursin("_jll", Wink.tool_search_packages("OpenSSL"))
        @test occursin("no packages matching",
            Wink.tool_search_packages("zzqqxxnosuchpkg"))
        @test startswith(Wink.tool_search_packages(""), "ERROR")
    end

    long = repeat("x", 10_000)
    t = Wink.truncate_output(long; limit = 1000)
    @test length(t) < 1600
    @test occursin("truncated", t)
end
