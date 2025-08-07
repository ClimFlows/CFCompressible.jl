using CFCompressible
using Documenter

DocMeta.setdocmeta!(CFCompressible, :DocTestSetup, :(using CFCompressible); recursive=true)

makedocs(;
    modules=[CFCompressible],
    authors="Thomas Dubos <thomas.dubos@polytechnique.edu> and contributors",
    sitename="CFCompressible.jl",
    format=Documenter.HTML(;
        canonical="https://dubosipsl.github.io/CFCompressible.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/dubosipsl/CFCompressible.jl",
    devbranch="main",
)
