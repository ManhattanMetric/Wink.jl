using Wink
using Documenter

DocMeta.setdocmeta!(Wink, :DocTestSetup, :(using Wink); recursive=true)

makedocs(;
    modules=[Wink],
    authors="Josh Ballanco <josh.ballanco@manhattanmetric.com> and contributors",
    sitename="Wink.jl",
    format=Documenter.HTML(;
        canonical="https://jballanc.github.io/Wink.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jballanc/Wink.jl",
    devbranch="main",
)
