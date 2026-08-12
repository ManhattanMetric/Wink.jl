# Context compaction: tier 0 (fold), tier 1 (distill), tier 2 (abstraction
# mining).
#
# Most of a long session's context is tool results that are pure functions of
# the live session — source listings, method tables, docstrings. The session
# itself holds the originals, so the transcript's copies are cache, not state:
# anything folded out can be re-derived, as current ground truth, by calling
# the tool again. Folding therefore replaces stale copies with a one-line
# elision and never touches what is NOT re-derivable: the conversation itself,
# and the results of state-mutating tools (eval_code, run_shell, edit_file),
# which are historical facts. The trigger compares each round's
# provider-reported prompt tokens against CONFIG.context_budget — a soft
# target, deliberately far below any model's hard window: pressure against it
# is treated as a signal, not an emergency.

# Allowlist, not a blocklist: a tool must be known-pure to be foldable, so new
# tools default to being kept.
const FOLDABLE_TOOLS = Set(["list_methods", "methods_with", "get_source",
    "get_ir", "get_doc", "type_info", "module_info", "list_variables",
    "where_defined", "read_file", "list_names", "search_packages",
    "search_docs"])

# Bodies at or below this length stay: the elision line would save nothing.
const FOLD_MIN_CHARS = 200

"""
    fold_result!(m::PT.ToolMessage) -> Bool

Fold one tool-result message in place when its tool is re-derivable and its
body long enough to be worth eliding. Mutating tools' results and short
bodies are left intact; an already-elided result is short, so a second pass
never re-folds.
"""
function fold_result!(m::PT.ToolMessage)
    name = something(m.name, "")
    body = m.content isa AbstractString ? m.content :
           string(something(m.content, ""))
    (name in FOLDABLE_TOOLS && length(body) > FOLD_MIN_CHARS) || return false
    m.content = "(elided during context compaction — $name output is " *
                "re-derivable; call it again for current ground truth)"
    return true
end

is_tool_result_msg(m) = m isa PT.ToolMessage

"""
    fold_history!(history; keep_recent = 2) -> Int

Apply [`fold_result!`](@ref) to every tool-result message in `history` except
the most recent `keep_recent` of them (the model may still be reacting to
those). Returns the number of folded results.
"""
function fold_history!(history::AbstractVector; keep_recent::Integer = 2)
    idxs = [i for i in eachindex(history) if is_tool_result_msg(history[i])]
    length(idxs) <= keep_recent && return 0
    folded = 0
    for i in idxs[1:(end - keep_recent)]
        fold_result!(history[i]) && (folded += 1)
    end
    return folded
end

"""
    compact!(; keep_recent = 2) -> Int

Tier-0 context compaction on the global chat: fold stale, re-derivable tool
results (introspection and search — never eval_code, run_shell, or edit_file,
whose results are historical facts) down to one-line elisions, keeping the
most recent `keep_recent` tool-result messages intact. The model recovers any
elided detail by calling the tool again — the session, not the transcript, is
the source of truth. Runs automatically when a round's prompt tokens exceed
`CONFIG.context_budget`; run it manually here or with `:compact`. Returns the
number of folded results.
"""
function compact!(; keep_recent::Integer = 2)
    chat = CHAT[]
    chat === nothing && return 0
    return fold_history!(chat.history; keep_recent)
end

# ---- tier 1: distill ---------------------------------------------------------
#
# What survives folding is the non-re-derivable residue: intent, decisions and
# their rationale, rejected approaches, task state. When the context is still
# over budget after folding has nothing left to give, a one-shot model pass
# distills the oldest span of the conversation into a compact brief that
# replaces it. The brief records only what the session cannot: code and docs
# are named, never restated, because introspection re-derives them as ground
# truth. Distilling can incorporate an earlier brief into the new one, so
# repeated passes are progressive, not destructive.

const DISTILL_WORD_LIMIT = 400
const DISTILL_KEEP_RECENT = 6     # messages never distilled (the live tail)
const DISTILL_MIN_SPAN = 6        # smaller spans aren't worth a model call —
                                  # and this is the cooldown: a fresh brief
                                  # plus a few rounds stays below it

const DISTILL_PROMPT = """
You are compacting the transcript of an AI pair-programming session running
inside a live Julia REPL. Your summary will REPLACE the transcript span you
are given: whatever you do not record is gone — EXCEPT that everything alive
in the session (definitions, types, docstrings, files) can be re-derived
later with introspection tools, so never restate code or documentation: name
it. Record, in this order, only what cannot be re-derived:
- Goal: what the user is building or learning; constraints and preferences
  they stated.
- Done: names and locations of definitions created or files edited; session
  state changed (packages added, data loaded, commands run).
- Decisions: choices made and why — especially approaches tried and REJECTED,
  and why, which is unrecoverable once this span is gone.
- Open: unfinished work; what was about to happen next.
Plain prose under those four labels, at most $(DISTILL_WORD_LIMIT) words
total, no code blocks.
"""

const BRIEF_HEADER = "(session brief — the earlier conversation was compacted " *
                     "into this summary; definitions it names are live in the " *
                     "session and re-derivable with introspection tools)"

# The distillable span: everything after the system prompt, protecting the
# most recent keep_recent messages, with the cut walked back so the first
# message kept after the brief is an assistant message (AIMessage or
# AIToolRequest). The brief is a user message, so this keeps role alternation
# intact for the Anthropic projection AND guarantees no ToolMessage is
# orphaned from the AIToolRequest that requested it.
function distill_span(history::AbstractVector; keep_recent::Integer = DISTILL_KEEP_RECENT)
    hi = length(history) - keep_recent
    while hi >= 2 && !(history[hi + 1] isa Union{PT.AIMessage, PT.AIToolRequest})
        hi -= 1
    end
    (hi >= 2 && hi - 1 >= DISTILL_MIN_SPAN) || return nothing
    return 2:hi
end

function render_span(history::AbstractVector, r::AbstractRange)
    parts = String[]
    for m in history[r]
        if m isa PT.ToolMessage
            push!(parts, "[tool result: $(something(m.name, "?"))]\n" *
                         string(something(m.content, "")))
        elseif m isa PT.AIToolRequest
            desc = join(("→ $(c.name)($(compact_args(c.args)))"
                         for c in m.tool_calls), "\n")
            preface = something(m.content, "")
            push!(parts, "[assistant]\n" *
                         (isempty(preface) ? desc : preface * "\n" * desc))
        else
            role = m isa PT.UserMessage ? "user" :
                   m isa PT.AIMessage ? "assistant" : "system"
            push!(parts, "[$role]\n$(m.content)")
        end
    end
    return join(parts, "\n\n")
end

"""
    distill_history!(chat::Chat; schema = nothing, io = CONFIG.status_io,
                     keep_recent = DISTILL_KEEP_RECENT) -> Bool

Tier-1 compaction: replace the oldest span of the conversation with a model-
written brief of its non-re-derivable content (goal, work done, decisions and
rejected approaches, open threads). `promoted` names definitions just mined
from this span (see [`mine_abstractions!`](@ref)) so the brief can reference
them. On any failure — the call erroring, or an empty brief — the history is
left untouched. Returns whether a distillation happened.
"""
function distill_history!(chat::Chat; schema = nothing, io::IO = CONFIG.status_io,
        keep_recent::Integer = DISTILL_KEEP_RECENT,
        promoted::Vector{String} = String[])
    r = distill_span(chat.history; keep_recent)
    r === nothing && return false
    # The span fits the model's hard window even though it broke the soft
    # budget: budget << window by design, and folding already ran.
    span_text = render_span(chat.history, r)
    isempty(promoted) ||
        (span_text *= "\n\n[compaction note] These definitions were just " *
                      "promoted out of this span's recurring patterns and are " *
                      "live in the session — record them under Done by name: " *
                      join(promoted, ", "))
    brief = try
        msgs = PT.AbstractMessage[PT.SystemMessage(DISTILL_PROMPT),
            PT.UserMessage(span_text)]
        m = _aitools_call(schema, msgs, PT.AbstractTool[])
        _tally!(chat, m)
        String(strip(something(m.content, "")))
    catch e
        e isa InterruptException && rethrow()
        debug_status(io, "distillation call failed: $(sprint(showerror, e))")
        return false
    end
    isempty(brief) && return false
    replaced = length(r)
    splice!(chat.history, r, [PT.UserMessage(BRIEF_HEADER * "\n\n" * brief)])
    status(io, "distilled $replaced messages into a session brief")
    return true
end

# ---- tier 2: abstraction mining ----------------------------------------------
#
# The thesis payoff. Token pressure on a span is treated as Rule-of-Three
# evidence: if the transcript kept re-spelling a pattern, that recurrence is
# the license Wink's own ethos demands before naming an abstraction — the
# mining pass may only propose what the span has already earned, never
# speculate to save tokens. It runs on the same doomed span as distillation,
# and must run FIRST: after the distill the code lumps (the evidence) are
# gone. Every proposal goes through the confirmation gate individually, so
# compaction becomes an interactive refactoring moment; accepted definitions
# land in the live session, and the distill pass is told their names so the
# brief can reference them.

const MINE_PROMPT = """
You are reviewing a span of an AI pair-programming transcript, from a live
Julia session, that is about to be compacted away. Your job is abstraction
mining: find code patterns the session kept re-spelling that deserve to
become named definitions.

Rules:
- Propose only what the transcript has EARNED: a pattern must appear at
  least twice, or be one code lump too large to hold in the head, before it
  deserves a name. Never invent speculative abstractions.
- Choose the weakest form that works: a function before a type, a type
  before a macro.
- If a definition already visible in the span captures the pattern, do not
  re-propose it.
- Each proposal must be complete, runnable Julia beginning with a docstring
  that states what it is for — the docstring is how it will be found later.
- Name things in the user's domain language, not the mechanism's.

Make one propose_abstraction call per proposal. If nothing has been earned,
make no calls and reply with the single word: none.
"""

"""
    propose_abstraction(name::String, definition::String, rationale::String) -> String

Propose promoting a recurring pattern from the transcript span into a named,
documented Julia definition. `definition` must be complete, valid Julia
source, beginning with a docstring, that defines `name`; `rationale` is one
sentence naming the recurrence that earned it.
"""
tool_propose_abstraction(name::String, definition::String, rationale::String) =
    "recorded"

propose_tool() = collect(values(PT.tool_call_signature(tool_propose_abstraction;
    name = "propose_abstraction", max_description_length = 4000)))

"""
    mine_abstractions!(chat::Chat; schema = nothing, io = CONFIG.status_io,
                       keep_recent = DISTILL_KEEP_RECENT) -> Vector{String}

Tier-2 compaction: ask the model to mine the doomed span for recurring code
patterns worth promoting into named definitions. Each proposal is shown
through the confirmation gate (kind `:abstract`, rationale included) and, if
accepted, evaluated in the live `Main`; definitions that fail to evaluate are
skipped, and names already defined are never re-proposed. Returns the names
installed.
"""
function mine_abstractions!(chat::Chat; schema = nothing, io::IO = CONFIG.status_io,
        keep_recent::Integer = DISTILL_KEEP_RECENT)
    r = distill_span(chat.history; keep_recent)
    r === nothing && return String[]
    msg = try
        m = _aitools_call(schema,
            PT.AbstractMessage[PT.SystemMessage(MINE_PROMPT),
                PT.UserMessage(render_span(chat.history, r))],
            propose_tool())
        _tally!(chat, m)
        m
    catch e
        e isa InterruptException && rethrow()
        debug_status(io, "abstraction mining call failed: $(sprint(showerror, e))")
        return String[]
    end
    installed = String[]
    for c in msg.tool_calls
        name = strip(string(get(c.args, :name, "")))
        code = string(get(c.args, :definition, ""))
        why = strip(string(get(c.args, :rationale, "")))
        (isempty(name) || isempty(strip(code))) && continue
        if isdefined(Main, Symbol(name))
            debug_status(io, "mined `$name` is already defined; skipped")
            continue
        end
        shown = (isempty(why) ? "" : "# why: $why\n") * code
        if !(CONFIG.autoeval || CONFIG.confirm(:abstract, shown))
            status(io, "declined abstraction proposal `$name`")
            continue
        end
        res = eval_in_main(code)
        if res.ok
            push!(installed, String(name))
            status(io, "promoted recurring pattern into `$name`")
        else
            debug_status(io, "mined definition `$name` failed to evaluate; skipped")
        end
    end
    return installed
end

# ---- trigger -----------------------------------------------------------------

# Called once per agent round with that round's reply, whose token counts are
# the provider's own report of the prompt we just sent. Escalation ladder:
# fold first; only when a fold yields nothing (so its savings, if any, have
# already been reflected in a later token report) do mining and distillation
# run — mining first, on the same span, while the evidence still exists.
function maybe_compact!(chat, msg, io::IO; schema = nothing)
    budget = CONFIG.context_budget
    budget > 0 || return nothing
    prompt_tokens = try
        msg.tokens[1]
    catch
        0
    end
    prompt_tokens > budget || return nothing
    folded = fold_history!(chat.history)
    if folded > 0
        status(io, "context at $prompt_tokens tokens (budget $budget): " *
                   "folded $folded stale tool results")
        return nothing
    end
    promoted = mine_abstractions!(chat; schema, io)
    distill_history!(chat; schema, io, promoted) ||
        debug_status(io, "context at $prompt_tokens tokens exceeds budget " *
                         "$budget; nothing to fold and span too small to distill")
    return nothing
end
