# Context compaction, tiers 0 (fold) and 1 (distill).
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
# is treated as a signal, not an emergency. Tier 2 (gated abstraction
# mining) is planned on top of these two passes.

# Allowlist, not a blocklist: a tool must be known-pure to be foldable, so new
# tools default to being kept.
const FOLDABLE_TOOLS = Set(["list_methods", "methods_with", "get_source",
    "get_ir", "get_doc", "type_info", "module_info", "list_variables",
    "where_defined", "read_file", "list_names", "search_packages",
    "search_docs"])

# Bodies at or below this length stay: the elision line would save nothing.
const FOLD_MIN_CHARS = 200

const FOLD_BLOCK_RE = r"<tool_result name=\"([A-Za-z_!]+)\">\n(.*?)\n</tool_result>"s

"""
    fold_message(content) -> (folded_content::String, count::Int)

Fold the re-derivable tool-result blocks inside one flattened tool-result
message, leaving mutating tools' results and short bodies intact.
"""
function fold_message(content::AbstractString)
    count = Ref(0)
    out = replace(String(content), FOLD_BLOCK_RE => function (block)
        m = match(FOLD_BLOCK_RE, block)
        name, body = m.captures
        (name in FOLDABLE_TOOLS && length(body) > FOLD_MIN_CHARS) || return block
        count[] += 1
        return "<tool_result name=\"$name\">\n(elided during context " *
               "compaction — $name output is re-derivable; call it again for " *
               "current ground truth)\n</tool_result>"
    end)
    return out, count[]
end

is_tool_result_msg(m) = m isa PT.UserMessage && startswith(m.content, "<tool_result")

"""
    fold_history!(history; keep_recent = 2) -> Int

Apply [`fold_message`](@ref) to every tool-result message in `history` except
the most recent `keep_recent` of them (the model may still be reacting to
those). Returns the number of folded result blocks.
"""
function fold_history!(history::AbstractVector; keep_recent::Integer = 2)
    idxs = [i for i in eachindex(history) if is_tool_result_msg(history[i])]
    length(idxs) <= keep_recent && return 0
    folded = 0
    for i in idxs[1:(end - keep_recent)]
        content, n = fold_message(history[i].content)
        n == 0 && continue
        history[i] = PT.UserMessage(content)
        folded += n
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
# message kept after the brief is an AIMessage — the brief is a user message,
# and user/assistant alternation must survive the splice.
function distill_span(history::AbstractVector; keep_recent::Integer = DISTILL_KEEP_RECENT)
    hi = length(history) - keep_recent
    while hi >= 2 && !(history[hi + 1] isa PT.AIMessage)
        hi -= 1
    end
    (hi >= 2 && hi - 1 >= DISTILL_MIN_SPAN) || return nothing
    return 2:hi
end

function render_span(history::AbstractVector, r::AbstractRange)
    parts = String[]
    for m in history[r]
        role = m isa PT.UserMessage ? "user" :
               m isa PT.AIMessage ? "assistant" : "system"
        push!(parts, "[$role]\n$(m.content)")
    end
    return join(parts, "\n\n")
end

"""
    distill_history!(chat::Chat; schema = nothing, io = CONFIG.status_io,
                     keep_recent = DISTILL_KEEP_RECENT) -> Bool

Tier-1 compaction: replace the oldest span of the conversation with a model-
written brief of its non-re-derivable content (goal, work done, decisions and
rejected approaches, open threads). On any failure — the call erroring, or an
empty brief — the history is left untouched. Returns whether a distillation
happened.
"""
function distill_history!(chat::Chat; schema = nothing, io::IO = CONFIG.status_io,
        keep_recent::Integer = DISTILL_KEEP_RECENT)
    r = distill_span(chat.history; keep_recent)
    r === nothing && return false
    # The span fits the model's hard window even though it broke the soft
    # budget: budget << window by design, and folding already ran.
    span_text = render_span(chat.history, r)
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

# ---- trigger -----------------------------------------------------------------

# Called once per agent round with that round's reply, whose token counts are
# the provider's own report of the prompt we just sent. Escalation ladder:
# fold first; only when a fold yields nothing (so its savings, if any, have
# already been reflected in a later token report) does distillation run.
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
    distill_history!(chat; schema, io) ||
        debug_status(io, "context at $prompt_tokens tokens exceeds budget " *
                         "$budget; nothing to fold and span too small to " *
                         "distill (tier 2 not yet implemented)")
    return nothing
end
