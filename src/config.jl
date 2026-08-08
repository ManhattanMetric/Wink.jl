# Global configuration and provider auto-detection.

"""
    default_confirm(kind::Symbol, text::AbstractString) -> Bool

Show a proposed action (`kind` is `:eval` or `:edit`) and ask the user for y/n
approval on stdin. Returns `false` (deny) in non-interactive sessions, on
Ctrl-C, and on anything but an explicit yes.
"""
function default_confirm(kind::Symbol, text::AbstractString)
    header = kind === :edit ? "--- proposed edit ---" : "--- proposed code ---"
    printstyled(stdout, "\n", header, "\n"; color = :yellow, bold = true)
    println(stdout, text)
    printstyled(stdout, "-"^length(header), "\n"; color = :yellow)
    if !isinteractive()
        println(stdout, "(non-interactive session: auto-denied; set Wink.autoeval!(true) to allow)")
        return false
    end
    resp = try
        Base.prompt(stdin, stdout, "Run this? [y/N]"; default = "n")
    catch e
        e isa InterruptException || rethrow()
        nothing
    end
    return lowercase(strip(something(resp, "n"))) in ("y", "yes")
end

"""
    WinkConfig

Mutable session configuration for Wink. Access the live instance as
`Wink.CONFIG`; prefer [`configure!`](@ref) and [`autoeval!`](@ref) for changes.
"""
Base.@kwdef mutable struct WinkConfig
    chat_model::String = ""
    chat_api_base::String = ""          # "" => provider default; else an OpenAI-compatible base URL
    embed_model::String = ""            # "" => no embedding backend; RAG degrades
    autoeval::Bool = false
    confirm::Function = default_confirm # (kind::Symbol, text) -> Bool
    max_rounds::Int = 12
    max_tokens::Int = 8192
    tool_output_limit::Int = 4_000
    status_io::IO = stderr
    instructions::String = ""           # session-level standing instructions
end

const CONFIG = WinkConfig()

const OLLAMA_CHAT_DEFAULT = "llama3.1"
const OLLAMA_EMBED_DEFAULT = "nomic-embed-text"
const OPENAI_EMBED_DEFAULT = "text-embedding-3-small"
const ANTHROPIC_CHAT_DEFAULT = "claude-opus-4-8"

function ensure_registered!(name::AbstractString, schema::PT.AbstractPromptSchema)
    if !haskey(PT.MODEL_REGISTRY, String(name))
        PT.register_model!(; name = String(name), schema,
            description = "Registered by Wink.jl")
    end
    return String(name)
end

"""
    detect_providers!(cfg::WinkConfig = CONFIG) -> WinkConfig

Pick default chat and embedding models from the API keys PromptingTools sees
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`), falling back to a local Ollama server.
Reads only environment state — never touches the network. The chat and
embedding backends are chosen independently because Anthropic offers no
embeddings endpoint.
"""
function detect_providers!(cfg::WinkConfig = CONFIG)
    anthropic = !isempty(PT.ANTHROPIC_API_KEY)
    openai = !isempty(PT.OPENAI_API_KEY)
    cfg.chat_model = if anthropic
        ANTHROPIC_CHAT_DEFAULT
    elseif openai
        PT.MODEL_CHAT
    else
        ensure_registered!(OLLAMA_CHAT_DEFAULT, PT.OllamaSchema())
    end
    cfg.embed_model = openai ? OPENAI_EMBED_DEFAULT : OLLAMA_EMBED_DEFAULT
    return cfg
end

"""
    configure!(; chat_model, chat_schema, chat_api_base, embed_model, embed_schema,
                 autoeval, max_rounds, max_tokens, tool_output_limit, instructions) -> WinkConfig

Override Wink's configuration. Model names unknown to PromptingTools' registry
are registered when the matching `*_schema` is supplied (e.g.
`configure!(chat_model = "my-model", chat_schema = PromptingTools.OllamaSchema())`);
without a schema an unknown name would route to the default (OpenAI) provider,
so a warning is emitted.

`chat_api_base` points chat requests at an OpenAI-compatible server (LM Studio,
llama.cpp, vLLM, …), e.g.

    configure!(chat_model = "prism-ml/bonsai-27b",
               chat_api_base = "http://localhost:1234/v1")

An unknown `chat_model` is then auto-registered with `CustomOpenAISchema`, and
when no `OPENAI_API_KEY` is set a placeholder key is sent (local servers accept
any key). Pass `chat_api_base = ""` to return to the provider's default URL.

`instructions` sets session-level standing instructions for the model — they
are appended to the system prompt alongside any global
(`~/.julia/config/wink.md`) and per-project (`.wink.md`) instruction files.
Prompt changes take effect when the next conversation starts (`Wink.reset!()`
or `:reset`).
"""
function configure!(; chat_model = nothing, chat_schema = nothing,
        chat_api_base = nothing, embed_model = nothing, embed_schema = nothing,
        autoeval = nothing, max_rounds = nothing, max_tokens = nothing,
        tool_output_limit = nothing, instructions = nothing)
    chat_api_base === nothing || (CONFIG.chat_api_base = String(chat_api_base))
    if chat_model !== nothing
        schema = if chat_schema === nothing && !isempty(CONFIG.chat_api_base) &&
                    !haskey(PT.MODEL_REGISTRY, String(chat_model))
            PT.CustomOpenAISchema()
        else
            chat_schema
        end
        _adopt_model!(chat_model, schema, "chat_model")
        CONFIG.chat_model = String(chat_model)
    end
    if embed_model !== nothing
        _adopt_model!(embed_model, embed_schema, "embed_model")
        CONFIG.embed_model = String(embed_model)
    end
    autoeval === nothing || (CONFIG.autoeval = Bool(autoeval))
    max_rounds === nothing || (CONFIG.max_rounds = Int(max_rounds))
    max_tokens === nothing || (CONFIG.max_tokens = Int(max_tokens))
    tool_output_limit === nothing || (CONFIG.tool_output_limit = Int(tool_output_limit))
    instructions === nothing || (CONFIG.instructions = String(instructions))
    return CONFIG
end

function _adopt_model!(name, schema, which::String)
    if schema !== nothing
        ensure_registered!(name, schema)
    elseif !haskey(PT.MODEL_REGISTRY, String(name))
        @warn "Wink: model `$name` is not in PromptingTools' registry and no schema was given; " *
              "requests will route to the default (OpenAI-compatible) provider. " *
              "Pass `$(which == "chat_model" ? "chat" : "embed")_schema = PromptingTools.AnthropicSchema()` (or similar) to fix routing."
    end
    return nothing
end

"""
    autoeval!(flag::Bool) -> Bool

Enable (`true`) or disable (`false`) automatic execution of model-proposed code
and file edits. When disabled (the default) every `eval_code`/`edit_file` tool
call asks for y/n confirmation first.
"""
autoeval!(flag::Bool) = (CONFIG.autoeval = flag)
