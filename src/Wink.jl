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

# Implementation lands in milestone order (see docs):
#   config.jl -> resolve.jl -> introspect.jl -> tools.jl -> evalengine.jl
#   -> agent.jl -> repl.jl -> edit.jl -> rag.jl

end # module Wink
