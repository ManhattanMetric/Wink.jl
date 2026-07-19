using Test
using Wink

# TestPkg is a tiny documented package fixture used across the suites.
const FIXTURES = joinpath(@__DIR__, "fixtures")
FIXTURES in LOAD_PATH || push!(LOAD_PATH, FIXTURES)
using TestPkg

@testset "Wink.jl" begin
    @testset "smoke" begin
        @test Wink isa Module
        @test TestPkg.greet("Wink") == "Hello, Wink!"
    end
end
