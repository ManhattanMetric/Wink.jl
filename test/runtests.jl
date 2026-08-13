using Test
using Wink
using PromptingTools
const PT = PromptingTools

# TestPkg is a tiny documented package fixture used across the suites.
const FIXTURES = joinpath(@__DIR__, "fixtures")
FIXTURES in LOAD_PATH || push!(LOAD_PATH, FIXTURES)
using TestPkg

@testset "Wink.jl" begin
    @testset "smoke" begin
        @test Wink isa Module
        @test TestPkg.greet("Wink") == "Hello, Wink!"
    end
    include("test_resolve.jl")
    include("test_introspect.jl")
    include("test_tools.jl")
    include("test_config.jl")
    include("test_eval.jl")
    include("test_shell.jl")
    include("test_agent.jl")
    include("test_compact.jl")
    include("test_local.jl")
    include("test_repl.jl")
    include("test_edit.jl")
    include("test_rag.jl")
    if get(ENV, "WINK_LIVE_TESTS", "") == "true"
        include("live/runtests.jl")
    end
end
