# The ai> REPL mode (ReplMaker-based) plus its meta-commands.
#
# ReplMaker routes the entered line through `handle_ai_input`, whose return
# value is evaluated by the standard REPL response path — returning `nothing`
# after printing ourselves means no extra echo.

const REPL_INITIALIZED = Ref(false)

"""
    init_repl(; start_key = ')', repl = Base.active_repl)

Install the `ai>` REPL mode. Called automatically from `__init__` in
interactive sessions; safe to call again (no-op once installed).
"""
function init_repl(; start_key::Char = ')', repl = Base.active_repl)
    REPL_INITIALIZED[] && return nothing
    ReplMaker.initrepl(handle_ai_input;
        prompt_text = "ai> ",
        prompt_color = :magenta,
        start_key,
        mode_name = :wink,
        sticky_mode = true,
        startup_text = false,
        repl)
    REPL_INITIALIZED[] = true
    doc = isempty(CONFIG.embed_model) ? "keyword only" : "semantic ($(CONFIG.embed_model))"
    printstyled(stderr,
        "[Wink] AI mode ready — press '", start_key, "' at an empty julia> prompt. ",
        "Chat model: ", CONFIG.chat_model, "; doc search: ", doc,
        ". Type :help inside ai> for commands.\n"; color = :light_black)
    return nothing
end

function handle_ai_input(s::AbstractString)
    input = strip(s)
    isempty(input) && return nothing
    if startswith(input, ':')
        handle_meta_command(String(input))
        return nothing
    end
    answer = try
        run_turn!(current_chat(), input; io = CONFIG.status_io)
    catch e
        e isa InterruptException && return nothing
        printstyled(stdout, "[Wink] error: "; color = :red, bold = true)
        println(stdout, sprint(showerror, e))
        # the message is for the user; the backtrace is for debugging Wink
        CONFIG.debug && println(stdout, sprint(Base.show_backtrace, catch_backtrace()))
        return nothing
    end
    print_answer(answer)
    return nothing
end

# Answers are printed verbatim rather than Markdown-rendered: full rendering
# reflows prose and mangles identifiers (snake_case reads as emphasis). Only
# code is styled — fenced blocks get real Julia syntax highlighting, inline
# `spans` a color.
function print_answer(answer::AbstractString; io::IO = stdout)
    isempty(strip(answer)) && return nothing
    fence = nothing        # language of the open fence, or nothing
    code_lines = String[]
    for line in split(answer, '\n')
        m = match(r"^\s*```(\S*)\s*$", line)
        if m !== nothing
            if fence === nothing
                fence = lowercase(something(m.captures[1], ""))
                empty!(code_lines)
            else
                print_code_block(io, join(code_lines, '\n'), fence)
                fence = nothing
            end
        elseif fence !== nothing
            push!(code_lines, String(line))
        else
            print_prose_line(io, line)
        end
    end
    fence === nothing || print_code_block(io, join(code_lines, '\n'), fence)
    return nothing
end

function print_code_block(io::IO, code::AbstractString, lang::AbstractString)
    body = if lang in ("julia", "julia-repl", "jl", "")
        try
            JuliaSyntaxHighlighting.highlight(code)
        catch
            code
        end
    else
        code
    end
    printstyled(io, "```", lang, "\n"; color = :light_black)
    println(io, body)
    printstyled(io, "```\n"; color = :light_black)
    return nothing
end

# Prose is written exactly as the model produced it; only inline `code` spans
# are colored.
function print_prose_line(io::IO, line::AbstractString)
    i = firstindex(line)
    for m in eachmatch(r"`([^`]+)`", line)
        print(io, SubString(line, i, prevind(line, m.offset)))
        printstyled(io, m.captures[1]; color = :cyan)
        i = m.offset + ncodeunits(m.match)
    end
    println(io, SubString(line, i))
    return nothing
end

const HELP_TEXT = """
Wink ai> mode — talk to the model; it can introspect and (with your approval)
run code in this session. Meta-commands:
  :help           show this help
  :reset          start a fresh conversation
  :history        print the conversation so far
  :config         show current configuration
  :autoeval on|off  toggle confirmation-free execution of model code/shell/edits
  :debug on|off   show diagnostic status lines (model name, agent-loop notes)
  :compact        fold stale re-derivable tool results out of the context
                  (runs automatically past context_budget prompt tokens)
  :model [name]   show or set the chat model
  :prompt         show the system prompt a new conversation would get
  :reindex        (re)build the documentation search index
  :pkgindex       fetch package descriptions from GitHub and index them for
                  search_packages (one-time; needs GITHUB_TOKEN or gh login)
Everything else you type is sent to the model. Ctrl-C interrupts a turn.
Backspace at an empty ai> prompt returns to julia>.
"""

function handle_meta_command(cmd::AbstractString; io::IO = stdout)
    parts = split(strip(cmd))
    head = parts[1]
    if head == ":help"
        print(io, HELP_TEXT)
    elseif head == ":reset"
        reset!()
        println(io, "conversation reset")
    elseif head == ":history"
        history(io)
    elseif head == ":config"
        show_config(io)
    elseif head == ":autoeval"
        if length(parts) == 2 && parts[2] in ("on", "off")
            autoeval!(parts[2] == "on")
            println(io, "autoeval ", parts[2],
                parts[2] == "on" ? " — model code runs without confirmation" : "")
        else
            println(io, "usage: :autoeval on|off (currently ",
                CONFIG.autoeval ? "on" : "off", ")")
        end
    elseif head == ":debug"
        if length(parts) == 2 && parts[2] in ("on", "off")
            configure!(debug = parts[2] == "on")
            println(io, "debug output ", parts[2])
        else
            println(io, "usage: :debug on|off (currently ",
                CONFIG.debug ? "on" : "off", ")")
        end
    elseif head == ":compact"
        n = compact!()
        println(io, n == 0 ? "nothing to fold" :
                    "folded $n stale tool result$(n == 1 ? "" : "s")")
    elseif head == ":model"
        if length(parts) >= 2
            configure!(chat_model = String(parts[2]))
            println(io, "chat model set to ", parts[2],
                " (takes full effect after :reset)")
        else
            println(io, "current chat model: ", CONFIG.chat_model)
        end
    elseif head == ":prompt"
        print(io, system_prompt())
        CHAT[] === nothing ||
            printstyled(io,
                "\n(the current conversation keeps the prompt it started with; ",
                ":reset applies changes)\n"; color = :light_black)
    elseif head == ":reindex"
        reindex!()
    elseif head == ":pkgindex"
        try
            index_packages!(; io)
        catch e
            e isa InterruptException || println(io, sprint(showerror, e))
        end
    else
        println(io, "unknown command ", head, " — try :help")
    end
    return nothing
end

function show_config(io::IO = stdout)
    c = CONFIG
    println(io, "chat_model        = ", c.chat_model)
    isempty(c.chat_api_base) ||
        println(io, "chat_api_base     = ", c.chat_api_base)
    println(io, "embed_model       = ",
        isempty(c.embed_model) ? "(none — keyword doc search only)" : c.embed_model)
    isempty(c.embed_api_base) ||
        println(io, "embed_api_base    = ", c.embed_api_base)
    println(io, "autoeval          = ", c.autoeval)
    println(io, "max_rounds        = ", c.max_rounds)
    println(io, "max_tokens        = ", c.max_tokens)
    println(io, "tool_output_limit = ", c.tool_output_limit)
    println(io, "context_budget    = ",
        c.context_budget == 0 ? "0 (auto-compaction off)" : c.context_budget)
    println(io, "debug             = ", c.debug)
    return nothing
end
