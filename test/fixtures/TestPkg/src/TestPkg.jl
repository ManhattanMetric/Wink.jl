"""
    TestPkg

A tiny, fully documented package used as an introspection target by Wink's test
suite. Every binding here exists to exercise a specific tool: source retrieval,
method tables, docstring lookup, type inspection, and RAG corpus construction.
"""
module TestPkg

export greet, twice, combine, Point

"""
    greet(name::String) -> String

Return a friendly greeting for `name`.
"""
greet(name::String) = "Hello, $(name)!"

"""
    twice(x::Int) -> Int

Return `2x`. A simple, concretely-typed target for IR-level introspection
(`code_typed`, `code_llvm`, `code_native`).
"""
twice(x::Int) = 2x

"""
    combine(a, b)

Combine two values. Deliberately has multiple methods so tests can exercise
method-table listing and per-signature source retrieval.
"""
combine(a::Int, b::Int) = a + b
combine(a::String, b::String) = a * b

"""
    Point{T}

A 2D point with coordinates of type `T`. Used to test `type_info` (fields,
parameters, supertype chain).
"""
struct Point{T}
    x::T
    y::T
end

"""
    simulate(p::Point, steps::Int) -> Point

Advance the point `p` through a toy random walk of `steps` steps and return the
final position. This docstring is intentionally long so that Wink's RAG chunker
has a realistic oversized docstring to split.

# Extended help

The random walk implemented here is the simplest possible discrete-time process:
at each step the point moves by one unit in each coordinate. It has no physical
meaning whatsoever; its only purpose is to give the documentation system a body
of text long enough to exceed the chunking threshold used by Wink's embedding
index. To that end, this section pads the docstring with genuinely descriptive,
if entirely unnecessary, prose about the function's behavior and design.

The function is pure with respect to its first argument: the input `Point` is
never mutated, because `Point` is an immutable struct. Each iteration constructs
a fresh `Point` whose coordinates are the previous coordinates advanced by one.
The element type `T` is preserved throughout, so a `Point{Float64}` walks in
floating-point space while a `Point{Int}` walks on the integer lattice. Callers
who need reproducibility should note that this walk is deterministic despite the
name "random walk" — an intentional lie that the documentation cheerfully
acknowledges, since determinism keeps the test suite stable across runs.

Performance characteristics are unremarkable: the loop allocates one immutable
struct per step, which Julia's compiler will typically stack-allocate and often
eliminate entirely after inlining. The function therefore serves as a pleasant
target for `code_llvm` inspection, where the reader can watch the optimizer
collapse the loop into straight-line arithmetic at higher optimization levels.

Numerical behavior is exactly what the arithmetic implies. For integer element
types the walk will wrap on overflow like any other native integer arithmetic
in Julia. For floating-point element types the walk loses precision once the
coordinate magnitude exceeds the density of representable values, which for a
toy fixture is a purely theoretical concern. No guards are implemented against
either condition, and none are planned, because the fixture's contract is with
the test suite rather than with any numerical application.

See also: [`Point`](@ref), [`combine`](@ref), [`twice`](@ref).
"""
function simulate(p::Point, steps::Int)
    for _ in 1:steps
        p = Point(p.x + one(p.x), p.y + one(p.y))
    end
    return p
end

"""
    ANSWER

The answer to the ultimate question, kept around as a documented constant so
tests can look up docs on a non-function binding.
"""
const ANSWER = 42

# An intentionally undocumented function, for testing the "no docs" path.
undocumented(x) = x

end # module TestPkg
