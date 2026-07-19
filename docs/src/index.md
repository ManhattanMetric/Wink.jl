```@meta
CurrentModule = Wink
```

# Wink

Wink is an **in-process AI pair-programmer for Julia**. Instead of asking a
model to write static code for some other process, Wink drops an `ai>` mode
into your REPL where the model works *inside* your live session:

- **It sees your real code.** Introspection tools give the model actual method
  sources (via CodeTracking), compiler IR at every level (`lowered` through
  `native`), method tables, type layouts, and docstrings — no hallucinated
  APIs.
- **It runs code where you are.** The model evaluates Julia directly in
  `Main`, so it can inspect your variables, define functions you can call
  immediately, and verify its own work. Every eval is shown to you and gated
  behind a y/n confirmation (toggle with [`autoeval!`](@ref)).
- **It edits code live.** The model can rewrite a method's source on disk, and
  Revise applies the change to the running session.
- **It searches your docs semantically.** All docstrings of loaded modules are
  embedded into a local index for retrieval-augmented doc search.

## Quickstart

```julia
julia> using Wink
[Wink] AI mode ready — press ')' at an empty julia> prompt. Chat model: claude-opus-4-8; ...

ai> why is my simulate function slow for Float32 inputs?
  … thinking (claude-opus-4-8)
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

Provider selection is automatic at load time: `ANTHROPIC_API_KEY` → Claude,
else `OPENAI_API_KEY` → OpenAI, else a local Ollama server. See
[Configuration](configuration.md).

Programmatic use works without the REPL mode:

```julia
Wink.ask("What methods does sort have for views?")
Wink.configure!(chat_model = "claude-haiku-4-5")
Wink.autoeval!(true)
Wink.reset!()
```

## Philosophy: grow the language, don't dump code

Wink is prompted to work the way Lisp programmers always have and Julia makes
natural: bottom-up, growing a vocabulary of abstractions until the program
reads like the domain. Ask for a blog engine and it should not emit a wall of
low-level Julia and HTML — it should help you name the layers
(`display_blog_post` composed of `render_markdown`, and once you ask for
comments, `display_comments`), defining and demonstrating each piece live in
your session so you can review and own every level. It is equally steered
*away* from the opposite failure: speculative type hierarchies and macros
where a function would do ("the weakest tool that works"). Standing
instructions (see [Configuration](configuration.md)) can push this default
further, or in a different direction, per project.

## How a turn works

Each `ai>` message runs an agentic loop: the model receives your message plus
a system prompt describing the live session (Julia version, active project,
loaded modules), calls introspection tools as needed, and — with your approval —
evaluates code or edits files. Tool activity is shown as gray status lines;
the final answer renders as Markdown. `Ctrl-C` interrupts a turn.

## Manual pages

- [Tools](tools.md) — what the model can do
- [Safety model](safety.md) — gates, perimeters, and what runs without asking
- [Configuration](configuration.md) — providers, models, and options
- [API reference](api.md)
