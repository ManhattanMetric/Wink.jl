"""
    Wink

An in-process AI pair-programmer for Julia.

Wink connects a language model to your *running* Julia session: it exposes deep
introspection (method sources, compiler IR, docstrings, types, modules) as tools
the model can call, lets the model evaluate code directly in `Main` and edit
package source on disk (both gated behind your confirmation, with Revise
applying file changes live), and adds an `ai>` REPL mode for driving all of it
conversationally.
"""
module Wink

using PromptingTools
const PT = PromptingTools
using CodeTracking
using Revise
using IOCapture
using ReplMaker
using Scratch
using Serialization
using LinearAlgebra
using Markdown
using REPL
using InteractiveUtils
import Pkg
import Downloads
import JSON3
import JuliaSyntaxHighlighting

include("config.jl")
include("resolve.jl")
include("registry.jl")
include("introspect.jl")
include("tools.jl")
include("evalengine.jl")
include("shell.jl")
include("edit.jl")
include("rag.jl")
include("llamacpp/LibLlama.jl")
const L = LibLlama
include("llamacpp/render.jl")
include("llamacpp/grammar.jl")
include("llamacpp/backend.jl")
include("agent.jl")
include("compact.jl")
include("repl.jl")

export configure!, autoeval!, ask, reset!, compact!, reindex!, index_packages!,
       local_model!

function __init__()
    # CONFIG is constructed during precompilation, so its status_io field holds
    # the precompile process's stderr — a dead handle in this process. Rebind it.
    CONFIG.status_io = stderr
    try
        PT.load_api_keys!()
    catch e
        @debug "Wink: could not refresh API keys" exception = e
    end
    try
        detect_providers!(CONFIG)
    catch e
        @debug "Wink: provider auto-detection failed" exception = e
    end
    if isinteractive()
        if isdefined(Base, :active_repl) && Base.active_repl !== nothing
            # `using Wink` typed at a live prompt
            try
                init_repl()
            catch e
                @debug "Wink: could not initialize the ai> REPL mode" exception = e
            end
        else
            # loaded from startup.jl — the REPL doesn't exist yet
            atreplinit() do repl
                try
                    if repl isa REPL.LineEditREPL
                        isdefined(repl, :interface) ||
                            (repl.interface = REPL.setup_interface(repl))
                        init_repl(; repl)
                    end
                catch e
                    @debug "Wink: could not initialize the ai> REPL mode" exception = e
                end
            end
        end
    end
    return nothing
end

end # module Wink
