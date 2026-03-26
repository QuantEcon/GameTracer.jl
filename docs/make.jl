using GameTracer
using Documenter

DocMeta.setdocmeta!(GameTracer, :DocTestSetup, :(using GameTracer); recursive=true)

makedocs(;
    modules=[GameTracer],
    authors="GameTracer.jl contributors",
    sitename="GameTracer.jl",
    format=Documenter.HTML(;
        canonical="https://QuantEcon.github.io/GameTracer.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/QuantEcon/GameTracer.jl",
    devbranch="main",
)
