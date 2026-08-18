# Wink

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
  and verify its own work. Every eval, edit, file write, and shell command is
  shown to you and gated behind a y/n confirmation (toggle with
  `Wink.autoeval!(true)`).
- **It edits code live.** The model can rewrite a method's source on disk, and
  Revise applies the change to the running session.
- **It searches your docs semantically.** Docstrings of loaded modules are
  embedded into a local index for retrieval-augmented doc search.
- **It manages its own context.** Long sessions are compacted in tiers — old
  tool output folds away, resolved threads distill to summaries, and recurring
  patterns can be promoted (with your approval) to session definitions.

## Quickstart

```julia
pkg> add Wink    # registration pending; until then: pkg> add https://github.com/ManhattanMetric/Wink.jl

julia> using Wink
[Wink] AI mode ready — press ')' at an empty julia> prompt. Chat model: claude-opus-5; ...

ai> why is my simulate function slow for Float32 inputs?
  … thinking (claude-opus-5)
  → list_methods(signature="simulate") [0.1s]
  → get_ir(signature="simulate(::Vector{Float32})", level="warntype") [0.8s]
Your `simulate` hits a type instability: ...

ai> fix it
--- proposed edit ---
src/model.jl (line 42):
 - acc = 0
 + acc = zero(eltype(xs))
Run this? [y/N] y
```

Programmatic entry points: `Wink.ask("...")`, `Wink.configure!(...)`,
`Wink.autoeval!(true)`, `Wink.reindex!()`.

## Three ways to bring a model

**1. Commercial APIs** — provider selection is automatic at load via
[PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl):
`ANTHROPIC_API_KEY` → Claude (`claude-opus-5`), else `OPENAI_API_KEY` →
OpenAI, else a local Ollama server. Embeddings for doc search are chosen
separately (Anthropic has no embeddings endpoint).

**2. OpenAI-compatible local servers** — LM Studio, llama.cpp's server, vLLM,
Ollama, and friends:

```julia
Wink.configure!(chat_model = "your-model-name",
                chat_api_base = "http://localhost:1234/v1")
```

**3. In-process pure-Julia inference** — no server, no HTTP, no C: load a GGUF
straight into your session and the forward pass runs in Julia arrays you can
introspect.

```julia
Wink.local_model!("path/to/model.gguf")                       # CPU
Wink.local_model!("path/to/model.gguf"; array = MtlArray)     # + GPU prefill (with `using Metal`)
```

Supported architectures: gemma-3, gemma-4 (MoE), and OLMoE, in F32/F16/BF16
and q4_0 / q4_1 / q8_0 / q6_K quantizations, all validated logit-for-logit
against llama.cpp. Weights are mmap'd zero-copy; a quantized 4GB model loads
in about a second. Tool calls are constrained at the logits — a malformed
call is unrepresentable — on model families with a canonical call syntax
(gemma-4); other families run text-only. On Apple silicon, generation runs
on the CPU (int8 SDOT kernels, at parity with llama.cpp) while
`array = MtlArray` keeps a device twin of the weights for prefill, where the
GPU is ~3× faster. Any GPUArrays-compatible vendor package works the same
way.

**Choosing a local model:** Wink is a tool-driven agent — pick models for
tool-calling ability first. A model that chats beautifully but cannot drive
tools is decorative here.

## Status

**v0.1.0 — initial release.** The commercial-API and OpenAI-compatible-server
routes are the day-to-day paths; in-process inference is the advanced /
exploratory path and the foundation for ongoing work on a Julia-specialized
local model. Development harnesses (llama.cpp oracle batteries, kernel
benchmarks) live on the `olmoe-tuning` branch.
