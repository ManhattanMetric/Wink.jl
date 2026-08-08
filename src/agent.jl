# Conversation state and the agentic tool loop.
#
# IMPORTANT DESIGN CONSTRAINT: PromptingTools' Anthropic renderer silently drops
# `AIToolRequest`/`ToolMessage` entries when re-rendering a conversation (its
# render pass only handles system/user/ai messages), so the documented
# "push the tool message back into the conversation" pattern never round-trips
# tool results to Claude. Wink therefore keeps the history as plain
# System/User/AIMessage entries only, flattening each tool exchange into text:
# the assistant's calls become an AIMessage, and the tool results go back as a
# single UserMessage wrapped in <tool_result> blocks. This renders correctly on
# every provider.

"""
    Chat

Conversation state for one Wink session: the (flattened) message history, the
tool registry, and cumulative token counts.
"""
mutable struct Chat
    history::Vector{PT.AbstractMessage}
    tool_map::Dict{String, PT.AbstractTool}
    tokens_in::Int
    tokens_out::Int
end

new_chat() = Chat(PT.AbstractMessage[PT.SystemMessage(system_prompt())],
    build_tool_map(), 0, 0)

const CHAT = Ref{Union{Nothing, Chat}}(nothing)

function current_chat()
    CHAT[] === nothing && (CHAT[] = new_chat())
    return CHAT[]::Chat
end

"""
    reset!()

Discard the global chat; the next `ask`/`ai>` message starts a fresh
conversation (with a freshly generated system prompt).
"""
reset!() = (CHAT[] = nothing; nothing)

has_tool(name::AbstractString) = any(p -> p.first == name, TOOL_SPECS)

function loaded_module_summary()
    stddir = realpath_or(joinpath(Sys.BINDIR, "..", "share", "julia"))
    stdlib, pkgs = String[], String[]
    for m in values(Base.loaded_modules)
        n = string(nameof(m))
        n in ("Main", "Base", "Core") && continue
        d = pkgdir(m)
        push!(d !== nothing && under(realpath_or(d), stddir) ? stdlib : pkgs, n)
    end
    sort!(unique!(pkgs))
    sort!(unique!(stdlib))
    return "Loaded packages: " * (isempty(pkgs) ? "(none)" : join(pkgs, ", ")) *
           "\n- Loaded stdlibs: " * (isempty(stdlib) ? "(none)" : join(stdlib, ", "))
end

# ---- standing instructions ---------------------------------------------------
#
# Three user-steerable layers, composed into the system prompt (most global
# first, most specific last): a global file, a per-project file at the active
# project root, and the runtime `CONFIG.instructions` string. All are optional.

const GLOBAL_INSTRUCTIONS_FILE = Ref(joinpath(homedir(), ".julia", "config", "wink.md"))
const PROJECT_INSTRUCTIONS_NAMES = (".wink.md", "WINK.md")
const INSTRUCTIONS_CHAR_LIMIT = 12_000

function project_instructions_file()
    proj = Base.active_project()
    proj === nothing && return nothing
    for name in PROJECT_INSTRUCTIONS_NAMES
        p = joinpath(dirname(proj), name)
        isfile(p) && return p
    end
    return nothing
end

function read_instructions(path::AbstractString)
    s = try
        String(strip(read(path, String)))
    catch
        return ""
    end
    length(s) > INSTRUCTIONS_CHAR_LIMIT &&
        (s = first(s, INSTRUCTIONS_CHAR_LIMIT) * "\n… [instructions truncated]")
    return s
end

function user_instructions_block()
    sections = String[]
    gf = GLOBAL_INSTRUCTIONS_FILE[]
    if isfile(gf)
        t = read_instructions(gf)
        isempty(t) || push!(sections, "### Global instructions ($gf)\n$t")
    end
    pf = project_instructions_file()
    if pf !== nothing
        t = read_instructions(pf)
        isempty(t) || push!(sections, "### Project instructions ($pf)\n$t")
    end
    s = strip(CONFIG.instructions)
    isempty(s) || push!(sections, "### Session instructions (Wink.configure!)\n$s")
    isempty(sections) && return ""
    return "\n## Standing instructions from the user\n" *
           "Follow these — where they conflict, later sections win, and all of " *
           "them take precedence over the style guidance above. They never " *
           "override the confirmation gates or the rule that introspection " *
           "tools are ground truth.\n\n" *
           join(sections, "\n\n") * "\n"
end

function system_prompt()
    doc_search = !has_tool("search_docs") ? "list_names (keyword) only" :
                 isempty(CONFIG.embed_model) ?
                 "search_docs degrades to keyword matching (no embedding provider)" :
                 "semantic via search_docs (embeddings: $(CONFIG.embed_model))"
    edit_rule = has_tool("edit_file") ?
                """
                - Before edit_file: call where_defined, then read_file to copy the exact
                  current text for old_string. Never construct an edit from memory.
                """ : ""
    base = """
    You are Wink, an AI pair-programmer living INSIDE the user's running Julia
    session — not a code generator for some other process. You share the session:
    code you evaluate runs in the user's Main, sees their variables, and persists
    after your turn ends.

    How you build software:
    Your purpose is not to emit code — it is to help the developer grow a
    vocabulary of abstractions in which their program becomes short, readable,
    and theirs. Julia is built for this: multiple dispatch, a rich type system,
    and homoiconic macros reward bottom-up design. Work into the language:
    - Top-level code should read like the domain: `display_blog_post(post)`
      composed of `render_markdown` and `display_comments` — not a hundred lines
      of inline string-building. When a new feature would swell an existing
      function, name the new concern as its own function and compose it in.
    - Learn the vocabulary that already exists before adding to it: use
      module_info, list_names, and get_source on the user's own code, and extend
      their abstractions rather than inventing parallel ones. New behavior on an
      existing concept is usually a new method on an existing generic function.
    - Build incrementally and live: define one small piece with eval_code,
      demonstrate it on real or sample data so the developer sees it work, then
      compose upward. A definition the developer cannot hold in their head is a
      failure even when it runs.
    - For sizeable tasks, propose the vocabulary first — the handful of names
      with one-line contracts the feature decomposes into — and let the
      developer react before you implement.
    - Choose the weakest tool that works: a function before a type, a type
      before a trait, a macro only when the abstraction is genuinely syntactic
      (new binding forms, compile-time work, notation). Introduce an abstraction
      when the domain names it or a second use appears — never speculatively.
    - Prefer dispatch over type-checking conditionals; give domain types `show`
      methods so they read well at the REPL; name things in the user's domain
      language, not the mechanism's.
    - When you introduce an abstraction, say in one sentence why the seam is
      there. The developer should be able to review and explain every layer.

    Session facts:
    - Julia $(VERSION) on $(Sys.KERNEL) ($(Sys.MACHINE))
    - Active project: $(something(Base.active_project(), "(none)"))
    - Revise is active: edits to package source files apply to the running session
    - Auto-approval of code execution: $(CONFIG.autoeval ? "ON" : "OFF — the user confirms each eval/edit")
    - Documentation search: $doc_search
    - $(loaded_module_summary())

    Ground rules:
    - Introspection tools are ground truth. Never recite source code from memory —
      call get_source. Use list_methods when unsure which method dispatch selects,
      methods_with to find what can be called with a value of a given type,
      list_variables to see what the session currently defines,
      get_ir with level "warntype" as the first stop for performance questions,
      get_doc before assuming an API's behavior, and search_packages before
      recommending or assuming any package exists.
    - Prefer small, verifiable eval_code steps; check intermediate results instead
      of running one large blob. Your code affects the user's real session state.
    $(edit_rule)- If a tool result says DECLINED, the user said no: do not retry the same call;
      briefly explain what you intended or propose an alternative.
    - Be concise. Put code in fenced blocks. When you changed session state, state
      exactly what changed.
    """
    return base * user_instructions_block()
end

status(io::IO, s::AbstractString) = printstyled(io, "  ", s, "\n"; color = :light_black)

function compact_args(args)
    (args === nothing || isempty(args)) && return ""
    parts = String[]
    for (k, v) in args
        s = v isa AbstractString ? v : repr(v)
        length(s) > 60 && (s = first(s, 57) * "…")
        push!(parts, "$k=$(repr(s))")
    end
    return join(parts, ", ")
end

function execute_tool_call(tool_map::AbstractDict, c::PT.ToolMessage)
    out = try
        PT.execute_tool(tool_map, c; throw_on_error = false)
    catch e
        e isa InterruptException && rethrow()
        e
    end
    out isa Exception &&
        return truncate_output("TOOL ERROR: " * sprint(Base.showerror, out))
    return truncate_output(string(out))
end

# Model routing: resolve the alias, look the model up in PT's registry, and use
# its schema class to decide provider-specific api_kwargs (Anthropic's default
# max_tokens of 2048 is far too small for code generation).
function _model_schema(model::AbstractString)
    id = get(PT.MODEL_ALIASES, model, model)
    spec = get(PT.MODEL_REGISTRY, id, nothing)
    return spec === nothing ? nothing : spec.schema
end

function _aitools_call(schema, history, tools)
    sch = schema === nothing ? _model_schema(CONFIG.chat_model) : schema
    api_kwargs = NamedTuple()
    key_kwargs = NamedTuple()
    if sch isa PT.AbstractAnthropicSchema
        api_kwargs = (; max_tokens = CONFIG.max_tokens)
    elseif sch isa Union{PT.CustomOpenAISchema, PT.LocalServerOpenAISchema}
        isempty(CONFIG.chat_api_base) ||
            (api_kwargs = (; url = CONFIG.chat_api_base))
        # OpenAI.jl rejects an empty api key even though local servers ignore
        # it; send a placeholder unless a real key is available as a fallback.
        isempty(PT.OPENAI_API_KEY) && (key_kwargs = (; api_key = "wink-local"))
    end
    kwargs = (; tools, model = CONFIG.chat_model, return_all = false,
        verbose = false, api_kwargs, key_kwargs...)
    return schema === nothing ? PT.aitools(history; kwargs...) :
           PT.aitools(schema, history; kwargs...)
end

function _tally!(chat::Chat, msg)
    try
        chat.tokens_in += msg.tokens[1]
        chat.tokens_out += msg.tokens[2]
    catch
    end
    return nothing
end

"""
    run_turn!(chat::Chat, user_text; schema = nothing, io = CONFIG.status_io) -> String

Run one full user turn: send `user_text` with the tool registry, execute any
tool calls the model makes (looping up to `CONFIG.max_rounds`), and return the
model's final text answer. `schema` overrides provider routing (used by tests
to inject PromptingTools' TestEcho schemas).
"""
function run_turn!(chat::Chat, user_text::AbstractString;
        schema::Union{Nothing, PT.AbstractPromptSchema} = nothing,
        io::IO = CONFIG.status_io)
    push!(chat.history, PT.UserMessage(String(user_text)))
    tools = collect(values(chat.tool_map))
    try
        for _ in 1:CONFIG.max_rounds
            status(io, "… thinking ($(CONFIG.chat_model))")
            msg = _aitools_call(schema, chat.history, tools)
            _tally!(chat, msg)
            calls = msg.tool_calls
            if isempty(calls)
                text = something(msg.content, "")
                push!(chat.history, PT.AIMessage(text))
                return text
            end
            desc = join(("→ $(c.name)($(compact_args(c.args)))" for c in calls), "\n")
            preface = something(msg.content, "")
            push!(chat.history,
                PT.AIMessage(isempty(preface) ? desc : preface * "\n" * desc))
            results = String[]
            for c in calls
                t0 = time()
                out = execute_tool_call(chat.tool_map, c)
                status(io,
                    "→ $(c.name)($(compact_args(c.args))) [$(round(time() - t0; digits = 1))s]")
                push!(results, "<tool_result name=\"$(c.name)\">\n$(out)\n</tool_result>")
            end
            push!(chat.history, PT.UserMessage(join(results, "\n")))
        end
        # Round cap reached: force a final, tool-free answer.
        status(io, "… tool-round limit reached; asking for a final answer")
        push!(chat.history,
            PT.UserMessage("Tool budget exhausted ($(CONFIG.max_rounds) rounds). " *
                           "Summarize your findings and answer now without further tool calls."))
        msg = _aitools_call(schema, chat.history, PT.AbstractTool[])
        _tally!(chat, msg)
        text = something(msg.content, "")
        push!(chat.history, PT.AIMessage(text))
        return text
    catch e
        e isa InterruptException || rethrow()
        push!(chat.history, PT.AIMessage("(interrupted by the user)"))
        return "Interrupted."
    end
end

"""
    ask(text::AbstractString) -> String

Send one message to the model on the global Wink chat (created on first use)
and return the final answer; the full agentic tool loop runs in between.
Conversation context persists across calls — see [`reset!`](@ref).
"""
ask(text::AbstractString) = run_turn!(current_chat(), text)

"""
    history(io::IO = stdout)

Pretty-print the global chat's (flattened) message history and token totals.
"""
function history(io::IO = stdout)
    chat = CHAT[]
    if chat === nothing
        println(io, "(no conversation yet)")
        return nothing
    end
    for msg in chat.history
        role = msg isa PT.SystemMessage ? "system" :
               msg isa PT.UserMessage ? "user" : "assistant"
        printstyled(io, "── ", role, " ", "─"^max(2, 46 - length(role)), "\n";
            color = :light_black)
        println(io, rstrip(msg.content))
    end
    printstyled(io, "tokens: ", chat.tokens_in, " in / ", chat.tokens_out, " out\n";
        color = :light_black)
    return nothing
end
