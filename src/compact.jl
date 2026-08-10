# Context compaction — tier 0: the deterministic "fold" pass.
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
# is treated as a signal, not an emergency. Later tiers (narrative
# distillation, abstraction mining) will build on this pass.

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

# Called once per agent round with that round's reply, whose token counts are
# the provider's own report of the prompt we just sent.
function maybe_compact!(chat, msg, io::IO)
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
    else
        debug_status(io, "context at $prompt_tokens tokens exceeds budget " *
                         "$budget but nothing is foldable (tiers 1-2 not yet implemented)")
    end
    return nothing
end
