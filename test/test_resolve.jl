@testset "resolve" begin
    rc = Wink.resolve_call("sort(::Vector{Int})")
    @test rc.func === sort
    @test rc.argtypes === Tuple{Vector{Int}}
    @test rc.method isa Method

    rc = Wink.resolve_call("Base.sort")
    @test rc.func === sort
    @test rc.argtypes === nothing && rc.method === nothing

    rc = Wink.resolve_call("TestPkg.greet(::String)")
    @test rc.method isa Method
    @test rc.method.module === TestPkg

    # bare-type and x::T argument forms
    rc = Wink.resolve_call("TestPkg.combine(Int, b::Int)")
    @test rc.argtypes === Tuple{Int, Int}
    @test rc.method isa Method

    @test Wink.resolve_binding("Base.Docs.doc") === Base.Docs.doc
    # REPL is loaded (Wink dependency) but not imported into Main
    @test Wink.resolve_binding("REPL.LineEdit") isa Module

    @test Wink.resolve_type("Vector{Int}") === Vector{Int}
    @test Wink.resolve_type("Array{Float64,2}") === Matrix{Float64}
    @test Vector{Int} <: Wink.resolve_type("Vector{<:Real}")
    @test Wink.resolve_type("typeof(sort)") === typeof(sort)
    @test Wink.resolve_type("TestPkg.Point") === TestPkg.Point
    @test Wink.resolve_type("TestPkg.Point{Float64}") === TestPkg.Point{Float64}

    # unsafe type expressions are rejected before any evaluation happens
    @test_throws Wink.WinkResolveError Wink.resolve_type("typeof(run(`ls`))")
    @test_throws Wink.WinkResolveError Wink.resolve_type("begin 1 end")
    @test !Wink.is_safe_type_expr(:(f(x)))
    @test !Wink.is_safe_type_expr(:(Vector{run(`ls`)}))
    @test Wink.is_safe_type_expr(:(Dict{String, Vector{T}} where {T <: Real}))

    @test_throws Wink.WinkResolveError Wink.resolve_call("nosuchfunction_xyz_123(::Int)")
    @test_throws Wink.WinkResolveError Wink.resolve_binding("Base.nosuchname_xyz")
    @test_throws Wink.WinkResolveError Wink.resolve_call("sort(; rev=true)")

    # resolvable function + types but no matching method
    rc = Wink.resolve_call("TestPkg.greet(::Int)")
    @test rc.method === nothing
    @test rc.argtypes === Tuple{Int}

    b = Wink.resolve_docbinding("TestPkg.greet")
    @test b.mod === TestPkg && b.var === :greet
    b = Wink.resolve_docbinding("Base.@show")
    @test b.var === Symbol("@show")
    b = Wink.resolve_docbinding("TestPkg")
    @test b.var === :TestPkg
end
