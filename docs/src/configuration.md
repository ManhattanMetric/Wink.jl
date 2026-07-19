# Configuration

## Provider auto-detection

At load time Wink picks defaults from the environment (no network calls):

| Condition | Chat model | Doc-search embeddings |
|---|---|---|
| `ANTHROPIC_API_KEY` set | `claude-opus-4-8` | see below |
| else `OPENAI_API_KEY` set | PromptingTools' default (`gpt-5-mini`) | `text-embedding-3-small` |
| else | `llama3.1` via local Ollama | `nomic-embed-text` via local Ollama |

Chat and embeddings are chosen **independently** because Anthropic offers no
embeddings endpoint: with only an Anthropic key, chat uses Claude while doc
search uses a local Ollama embedding model if one is running — and otherwise
degrades to keyword (`apropos`) search with a clear notice.

## Changing settings

```julia
Wink.configure!(
    chat_model = "claude-haiku-4-5",   # any PromptingTools model name or alias
    max_rounds = 20,                   # tool rounds per turn
    max_tokens = 16_000,               # response budget (Anthropic)
    tool_output_limit = 8_000,         # truncation limit per tool result
)
Wink.autoeval!(true)                   # skip y/n confirmation
```

Model names unknown to PromptingTools' registry need a schema so requests
route to the right provider:

```julia
Wink.configure!(chat_model = "my-finetune", chat_schema = PromptingTools.OllamaSchema())
```

Inside `ai>` mode, `:config`, `:model <name>`, and `:autoeval on|off` cover
the common cases. Settings are not persisted across sessions; put a
`Wink.configure!` call in `~/.julia/config/startup.jl` (after `using Wink`)
for durable preferences, or use PromptingTools' own preference system for API
keys and default models.

## Steering Wink with standing instructions

Wink's system prompt has a fixed core (session facts plus ground rules like
"never recite source from memory — call `get_source`"), and on top of it you
can layer your own standing instructions to point the model in a specific
direction. Three layers compose, most global first:

1. **Global** — `~/.julia/config/wink.md`, applied in every session.
2. **Per-project** — a `.wink.md` (or `WINK.md`) next to the active project's
   `Project.toml`. This is the place for project conventions.
3. **Session** — `Wink.configure!(instructions = "...")`, for ad-hoc steering.

Where layers conflict, later (more specific) ones win. None of them can
override the confirmation gates.

Examples of what goes in these files:

```markdown
# ~/.julia/config/wink.md — who you're helping
I'm comfortable with Julia but new to its performance model. When you touch
performance, show the `warntype` evidence and explain what it means.

# .wink.md — project conventions
This package follows BlueStyle. Tests use plain @testset (no snapshot
frameworks). Prefer explaining a fix before applying it; we favor small,
reviewable edits over rewrites.
```

Inspect exactly what the model is told with `:prompt` in `ai>` mode. Prompt
changes (any layer) take effect when the next conversation starts — `:reset`
or [`reset!`](@ref); the files are re-read at that point, so editing `.wink.md`
mid-session is fine.

## The documentation index

`search_docs` builds its embedding index lazily on first use and caches it in
a scratch space keyed by Julia version and embedding model, so subsequent
sessions load instantly. The index is **never** rebuilt implicitly; after
loading new packages run:

```julia
Wink.reindex!()    # or :reindex in ai> mode
```

Search results carry a staleness note whenever the set of loaded modules has
changed since indexing.

## Load-order note (Revise)

Wink loads Revise as a dependency, which makes `get_source` reflect live edits
and lets `edit_file` apply changes to the running session. Revise only tracks
packages loaded *after* it — if you `using Wink` late in a session, packages
loaded earlier still introspect fine (source reads come from disk) but won't
hot-reload. Loading Revise (or Wink) early — e.g. from `startup.jl` — gives
the fullest experience.
