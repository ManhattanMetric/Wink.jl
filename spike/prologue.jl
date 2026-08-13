# Prologue injected at the top of the generated LibLlama.jl: point the
# bindings at the vendored dylib (its @loader_path rpath resolves the ggml
# dependencies sitting beside it).
const libllama = joinpath(@__DIR__, "vendor", "llama-b10405", "libllama.dylib")
