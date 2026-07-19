@testset "introspect" begin
    s = Wink.source(TestPkg.greet, Tuple{String})
    @test s.kind === :file
    @test occursin("greet(name::String)", s.code)
    @test occursin("TestPkg.jl", s.file)
    @test s.line > 0

    # methods defined via eval have no source file -> lowered-IR fallback
    Core.eval(Main, :(__wink_dynamic_fn(x) = x + 1))
    m = which(Main.__wink_dynamic_fn, Tuple{Any})
    s2 = Wink.source(m)
    @test s2.kind === :lowered
    @test !isempty(s2.code)

    for lvl in Wink.IR_LEVELS
        txt = Wink.ir_text(TestPkg.twice, Tuple{Int}, lvl)
        @test txt isa String
        @test !isempty(strip(txt))
    end
    @test_throws Wink.WinkResolveError Wink.ir_text(TestPkg.twice, Tuple{Int}, :bogus)

    md = Wink.docstring("TestPkg.greet")
    @test occursin("friendly greeting", string(md))

    ti = Wink.typeinfo_text(TestPkg.Point)
    @test occursin("struct", ti)
    @test occursin("x :: T", ti)
    @test occursin("Any", ti)
    @test occursin("supertype chain", Wink.typeinfo_text(Vector{Int}))
    ti3 = Wink.typeinfo_text(Integer)
    @test occursin("abstract", ti3)
    @test occursin("subtypes", ti3)

    mi = Wink.modinfo_text(TestPkg)
    @test occursin("greet", mi)
    @test occursin("Point", mi)
    mia = Wink.modinfo_text(TestPkg; all = true)
    @test occursin("undocumented", mia)

    mt = Wink.methodtable_text(TestPkg.combine)
    @test occursin("2 methods", mt)
    @test occursin("combine", mt)
end
