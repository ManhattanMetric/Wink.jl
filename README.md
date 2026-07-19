# Wink

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ManhattanMetric.github.io/Wink.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ManhattanMetric.github.io/Wink.jl/dev/)
[![Build Status](https://github.com/ManhattanMetric/Wink.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ManhattanMetric/Wink.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

Wink is an **in-process AI pair-programmer for Julia**. Instead of asking a model
to write static code for some other process, Wink drops an `ai>` mode into your
REPL where the model works *inside* your live session:

- **It sees your real code.** Introspection tools give the model actual method
  sources (via CodeTracking), compiler IR at every level (`lowered` through
  `native`), method tables, type layouts, and docstrings — no hallucinated APIs.
- **It runs code where you are.** The model evaluates Julia directly in `Main`,
  so it can inspect your variables, define functions you can call immediately,
  and verify its own work. Every eval is shown to you and gated behind a y/n
  confirmation (toggle with `Wink.autoeval!(true)`).
- **It edits code live.** The model can rewrite a method's source on disk, and
  Revise applies the change to the running session.
- **It searches your docs semantically.** All docstrings of loaded modules are
  embedded into a local index for retrieval-augmented doc search.

Wink is provider-agnostic via
[PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl): it
auto-detects Anthropic (Claude), OpenAI, or a local Ollama server from your
environment.

> **Status: work in progress.** The package is under active initial development
> and not yet registered.
