# Conversation state and the agentic tool loop.
#
# The history is stored NATIVELY: tool exchanges live as `AIToolRequest` (the
# assistant's calls) and `ToolMessage` (each result, keyed by tool_call_id).
# PromptingTools' OpenAI-compatible renderer round-trips both correctly, so on
# that path the provider's own chat template formats past tool calls — nothing
# in the transcript looks like text the model could imitate.
#
# PROVIDER EXCEPTION: PT's Anthropic renderer (as of the 0.94 pin) silently
# drops `AIToolRequest`/`ToolMessage` when re-rendering a conversation (its
# render pass only handles system/user/ai messages). For Anthropic schemas
# only, `flatten_history` projects the native history into plain text at
# render time: calls become "→ tool(...)" gloss lines on an AIMessage, results
# a UserMessage of <tool_result> blocks. Storage is never flattened. Once the
# planned upstream PR teaches PT's Anthropic renderer tool messages, delete
# `flatten_history` and the projection branch in `_aitools_call`.

"""
    Chat

Conversation state for one Wink session: the native message history (tool
exchanges as `AIToolRequest`/`ToolMessage`), the tool registry, and cumulative
token counts.
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
    shell_rule = has_tool("run_shell") ?
                 """
                 - Shell work (git, mkdir/ls/cp, curl, build tools) goes through run_shell
                   as one plain command string — never rewrite it as Julia code. If Julia
                   code must shell out, the only valid form is a backtick literal passed to
                   run, e.g. run(`git -C /path init`): Julia has no system(), run
                   accepts neither plain strings nor vectors, and a backtick literal alone
                   only constructs the command without running it. `cd` inside a command
                   does not persist; use run_shell's dir argument or flags like `git -C`.
                 - EXCEPTION: Julia package operations (Pkg.activate, Pkg.add, ...) run
                   through eval_code, never `julia -e` in run_shell — a subprocess cannot
                   change THIS session's environment, so its Pkg.add installs packages the
                   session still cannot load.
                 """ : ""
    edit_rule = has_tool("edit_file") ?
                """
                - Before edit_file: call where_defined, then read_file to copy the exact
                  current text for old_string. Never construct an edit from memory.
                - New files are created with write_file, existing ones changed with
                  edit_file — never with open/write inside eval_code, and never with
                  heredocs or redirection (cat <<EOF, echo >) in run_shell: file changes
                  must pass through the file tools' gates, where the user sees clean
                  content and Revise applies edits live. If write_file reports a path
                  outside the perimeter, fix the location (cd/Pkg.activate); do not
                  route around it through the shell.
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
    $(shell_rule)$(edit_rule)- If a tool result says DECLINED, the user said no: do not retry the same call;
      briefly explain what you intended or propose an alternative.
    - The "→ tool(...)" lines in this transcript are records of tool calls already
      made. Writing such a line as text runs nothing — invoke tools only through
      the tool-calling interface.
    - Be concise. Put code in fenced blocks. When you changed session state, state
      exactly what changed.
    """
    return base * user_instructions_block()
end

status(io::IO, s::AbstractString) = printstyled(io, "  ", s, "\n"; color = :light_black)

# Diagnostic feedback (loop mechanics, cache hits): only shown when
# CONFIG.debug is on; toggle with configure!(debug = true) or :debug on.
debug_status(io::IO, s::AbstractString) = CONFIG.debug ? status(io, s) : nothing

"""
    with_thinking(f, io) -> result of f()

Run `f` while showing a "thinking" indicator on `io`. On a TTY the indicator
animates in place, cycling "thinking." / ".." / "..." and erasing itself when
`f` returns; elsewhere it degrades to a single static status line. With
`CONFIG.debug` on, the chat model's name is included.
"""
function with_thinking(f::Function, io::IO)
    label = CONFIG.debug ? "thinking ($(CONFIG.chat_model))" : "thinking"
    if !(io isa Base.TTY)
        status(io, label * "...")
        return f()
    end
    done = Ref(false)
    spinner = @async begin
        i = 0
        while !done[]
            print(io, "\r\e[2K")
            printstyled(io, "  ", label, "."^(i % 3 + 1); color = :light_black)
            flush(io)
            i += 1
            sleep(0.35)
        end
        print(io, "\r\e[2K")
        flush(io)
    end
    try
        return f()
    finally
        done[] = true
        wait(spinner)
    end
end

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

# Render-time projection for providers whose PromptingTools renderer drops
# tool messages (see the header comment). Consecutive user-role content is
# merged so role alternation survives (tool results followed by a system
# note, say). Delete once PT's Anthropic renderer learns tool messages.
function flatten_history(history::AbstractVector)
    out = PT.AbstractMessage[]
    push_user! = text -> if !isempty(out) && out[end] isa PT.UserMessage
        out[end] = PT.UserMessage(out[end].content * "\n" * text)
    else
        push!(out, PT.UserMessage(text))
    end
    for m in history
        if m isa PT.ToolMessage
            push_user!("<tool_result name=\"$(something(m.name, "?"))\">\n" *
                       "$(string(something(m.content, "")))\n</tool_result>")
        elseif m isa PT.AIToolRequest
            desc = join(("→ $(c.name)($(compact_args(c.args)))"
                         for c in m.tool_calls), "\n")
            preface = something(m.content, "")
            push!(out, PT.AIMessage(isempty(preface) ? desc : preface * "\n" * desc))
        elseif m isa PT.UserMessage
            push_user!(m.content)
        else
            push!(out, m)
        end
    end
    return out
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

function _aitools_call(schema, history, tools)
    # An in-process model, when loaded, takes the default route; explicit
    # schemas (tests, overrides) still go through PromptingTools.
    schema === nothing && LOCAL_MODEL[] !== nothing &&
        return _local_aitools(LOCAL_MODEL[]::LocalModel, history, tools)
    sch = schema === nothing ? _model_schema(CONFIG.chat_model) : schema
    api_kwargs, key_kwargs = custom_server_kwargs(sch, CONFIG.chat_api_base)
    # Anthropic's default max_tokens of 2048 is far too small for code generation.
    sch isa PT.AbstractAnthropicSchema &&
        (api_kwargs = (; max_tokens = CONFIG.max_tokens))
    # Projection point for the renderer gap (see header comment): Anthropic
    # schemas get a flattened view; everyone else gets the native history.
    msgs = sch isa PT.AbstractAnthropicSchema ? flatten_history(history) : history
    kwargs = (; tools, model = CONFIG.chat_model, return_all = false,
        verbose = false, api_kwargs, key_kwargs...)
    return schema === nothing ? PT.aitools(msgs; kwargs...) :
           PT.aitools(schema, msgs; kwargs...)
end

# A no-tool-call reply that ends by imitating the transcript's "→ tool(...)"
# gloss — on one line or spilling a code payload across many — is an
# unexecuted intent, not an answer: small models imitate the flattened history
# instead of using the tool-calling API. Detection: take the last line that
# opens like a call to a registered tool and walk its parentheses to the end
# of the message. Ending inside the call, or exactly at its close, is an
# intent; trailing prose after the close is a mention of a past call. Returns
# the tool name, or nothing when the text ends like a real answer.
function textual_tool_call(text::AbstractString, tool_names)
    s = rstrip(text)
    isempty(s) && return nothing
    opener = nothing
    for m in eachmatch(r"^\s*(?:→|->)?\s*([A-Za-z_][A-Za-z0-9_!]*)\s*\("m, s)
        String(m.captures[1]) in tool_names && (opener = m)
    end
    opener === nothing && return nothing
    name = String(opener.captures[1])
    rest = SubString(s, findnext('(', s, opener.offset))
    depth = 0
    for (j, c) in pairs(rest)
        c == '(' && (depth += 1)
        c == ')' && (depth -= 1)
        depth == 0 &&
            return isempty(strip(SubString(rest, nextind(rest, j)))) ? name : nothing
    end
    return name   # parens never closed: the message ends mid-call
end

# The recovery reflex, part 2 (part 1 is `api_hint`): spot blind retry streaks.
# A failed eval_code renders as its error text — "ERROR: ..." or
# "parse error: ..." from format_for_model, "TOOL ERROR: ..." from the loop's
# own guard. Successful evals ("value: ...") and DECLINED never match.
failed_eval_output(out::AbstractString) =
    startswith(out, "ERROR") || startswith(out, "parse error:") ||
    startswith(out, "TOOL ERROR")

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
    eval_fail_streak = 0    # consecutive failed eval_code calls, across rounds
    try
        for _ in 1:CONFIG.max_rounds
            msg = with_thinking(() -> _aitools_call(schema, chat.history, tools), io)
            _tally!(chat, msg)
            maybe_compact!(chat, msg, io; schema)
            calls = msg.tool_calls
            if isempty(calls)
                text = something(msg.content, "")
                push!(chat.history, PT.AIMessage(text))
                name = textual_tool_call(text, keys(chat.tool_map))
                if name !== nothing
                    push!(chat.history,
                        PT.UserMessage("(system note: your message ended with " *
                                       "\"$(name)(...)\" written as text, which runs " *
                                       "nothing — NO tool call written as text in that " *
                                       "message ran. The \"→ tool(...)\" lines in this " *
                                       "transcript are records of calls already made. " *
                                       "Invoke $(name) through the tool-calling " *
                                       "interface, one call at a time, to actually " *
                                       "run it.)"))
                    debug_status(io,
                        "model wrote a tool call as text; asking for a real $(name) call")
                    continue
                end
                return text
            end
            push!(chat.history, msg)   # the AIToolRequest itself, calls included
            # narration accompanying tool calls is part of the answer the user
            # is waiting on — show it now, not never
            preface = something(msg.content, "")
            isempty(strip(preface)) || print_answer(preface)
            for c in calls
                t0 = time()
                out = execute_tool_call(chat.tool_map, c)
                # Any successful eval, or any other tool call (introspection is
                # exactly the compliance we want), breaks the streak.
                eval_fail_streak = c.name == "eval_code" && failed_eval_output(out) ?
                                   eval_fail_streak + 1 : 0
                status(io,
                    "→ $(c.name)($(compact_args(c.args))) [$(round(time() - t0; digits = 1))s]")
                push!(chat.history,
                    PT.ToolMessage(; content = out, tool_call_id = c.tool_call_id,
                        raw = c.raw, name = c.name, args = c.args))
            end
            if eval_fail_streak >= 2
                push!(chat.history,
                    PT.UserMessage("(system note: that is $(eval_fail_streak) " *
                                   "consecutive eval_code failures. Stop retrying " *
                                   "variations — the API is not what you remember. " *
                                   "Before the next eval_code call, look it up: " *
                                   "get_doc and list_methods on the function " *
                                   "involved, or search_docs for the task. " *
                                   "Introspection tools are ground truth.)"))
                debug_status(io,
                    "$(eval_fail_streak) consecutive eval failures; demanding introspection")
            end
        end
        # Round cap reached: force a final, tool-free answer.
        debug_status(io, "tool-round limit reached; asking for a final answer")
        push!(chat.history,
            PT.UserMessage("Tool budget exhausted ($(CONFIG.max_rounds) rounds). " *
                           "Summarize your findings and answer now without further tool calls."))
        msg = with_thinking(() -> _aitools_call(schema, chat.history, PT.AbstractTool[]), io)
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

Pretty-print the global chat's message history and token totals.
"""
function history(io::IO = stdout)
    chat = CHAT[]
    if chat === nothing
        println(io, "(no conversation yet)")
        return nothing
    end
    for msg in chat.history
        role, body = if msg isa PT.SystemMessage
            "system", msg.content
        elseif msg isa PT.UserMessage
            "user", msg.content
        elseif msg isa PT.ToolMessage
            "tool: $(something(msg.name, "?"))", string(something(msg.content, ""))
        elseif msg isa PT.AIToolRequest
            desc = join(("→ $(c.name)($(compact_args(c.args)))"
                         for c in msg.tool_calls), "\n")
            preface = something(msg.content, "")
            "assistant", isempty(preface) ? desc : preface * "\n" * desc
        else
            "assistant", string(something(msg.content, ""))
        end
        printstyled(io, "── ", role, " ", "─"^max(2, 46 - length(role)), "\n";
            color = :light_black)
        println(io, rstrip(body))
    end
    printstyled(io, "tokens: ", chat.tokens_in, " in / ", chat.tokens_out, " out\n";
        color = :light_black)
    return nothing
end
