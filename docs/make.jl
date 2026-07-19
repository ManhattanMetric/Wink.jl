using Wink
using Documenter

DocMeta.setdocmeta!(Wink, :DocTestSetup, :(using Wink); recursive=true)

makedocs(;
    modules=[Wink],
    authors="Josh Ballanco <josh.ballanco@manhattanmetric.com> and contributors",
    sitename="Wink.jl",
    format=Documenter.HTML(;
        canonical="https://ManhattanMetric.github.io/Wink.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Tools" => "tools.md",
        "Safety model" => "safety.md",
        "Configuration" => "configuration.md",
        "API reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/ManhattanMetric/Wink.jl",
    devbranch="main",
)
