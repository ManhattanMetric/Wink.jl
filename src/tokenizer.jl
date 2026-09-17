# Pure-Julia tokenizer for SentencePiece-style vocabularies (▁ for spaces,
# <0xXX> byte fallback), built entirely from GGUF metadata. Two merge rules
# share one vocabulary shape, selected by `tokenizer.ggml.model`:
#
#   "llama" (gemma-3 and other SPM models) mirrors llama.cpp's
#   llm_tokenizer_spm: text is partitioned around special tokens (longest
#   match first), each fragment's spaces become ▁, it is split into UTF-8
#   characters, and adjacent symbols merge greedily — highest vocab SCORE
#   first, leftmost on ties.
#
#   "gemma4" mirrors llama.cpp's llm_tokenizer_bpe with the GEMMA4
#   pre-tokenizer: gemma-4 ships a merge list, and its vocabulary is
#   merge-RANKED, not score-driven. Same special-token partitioning and ▁
#   escaping, then the text splits into runs of non-newlines and runs of
#   newlines (a newline run that is itself a token stays whole), characters
#   merge lowest merge rank first, leftmost on ties, and BOS is always added
#   (llama.cpp overrides add_bos for gemma-4). The two rules agree on most
#   text but not all — whitespace-heavy code is where they part.
#
# Either way, symbols still unmatched fall back to <0xXX> byte tokens.
# llama.cpp's tokenizer is the dev-time oracle for both
# (test/test_tokenizers.jl checks the recorded outputs).

module SPMTokenizer

using ..GGUF: GGUFFile, metadata

export Tokenizer, tokenize, detokenize, piece

const T_NORMAL = 1
const T_UNKNOWN = 2
const T_CONTROL = 3
const T_USER = 4
const T_BYTE = 6

const WS = "▁"   # ▁

# Pieces llama.cpp promotes to CONTROL at vocab load whatever type the GGUF
# declares (its end-of-generation, end-of-turn, and fill-in-the-middle
# detection by text) — so they partition out of raw text only under
# parse_special. Gemma-4 ships <|tool_response>, <turn|>, and <eos> this way.
# llama.cpp promotes only the first EOT/EOM/FIM match per kind; promoting all
# differs only for vocabularies declaring several non-control matches.
const LLAMA_CONTROL_LOOKING = Set([
    "_<EOT>", "[e~[", "[EOS]", "[EOT]", "[PAD]", "</s>", "<|call|>", "<|calls|>",
    "<|code_middle|>", "<|code_prefix|>", "<|code_suffix|>", "<|end_of_text|>",
    "<|end|>", "<|endoftext|>", "<|eom_id|>", "<|eot_id|>", "<|file_sep|>",
    "<|fim_middle|>", "<|fim_pad|>", "<|fim_prefix|>", "<|fim_repo|>",
    "<|fim_suffix|>", "<|flush|>", "<|im_end|>", "<|middle|>", "<|prefix|>",
    "<|repo_name|>", "<|return|>", "<|suffix|>", "<|tool_response>",
    "<end_of_turn>", "<end_of_utterance>", "<｜end▁of▁sentence｜>", "<eos>",
    "<EOT>", "<fim_middle>", "<fim_pad>", "<fim_prefix>", "<fim_suffix>",
    "<fim-middle>", "<fim-pad>", "<fim-prefix>", "<fim-repo>", "<fim-suffix>",
    "<｜fim▁begin｜>", "<｜fim▁end｜>", "<｜fim▁hole｜>", "<MID>", "▁<MID>", "<PAD>",
    "<PRE>", "▁<PRE>", "<REPO>", "<reponame>", "<SUF>", "▁<SUF>", "<turn|>"])

struct Tokenizer
    pieces::Vector{String}
    scores::Vector{Float32}
    types::Vector{Int32}
    id_of::Dict{String, Int}          # piece → 0-based id
    # raw-text-matched pieces, longest-first: (piece, id, type). USER_DEFINED
    # entries (e.g. gemma's literal multi-space runs "  ", "   ") match
    # ALWAYS; CONTROL/UNKNOWN entries match only under parse_special — this
    # asymmetry is llama.cpp's semantic, verified by the oracle battery.
    specials::Vector{Tuple{String, Int, Int32}}
    byte_ids::Vector{Int}             # byte value + 1 → id (-1 when absent)
    bos::Int
    eos::Int
    unk::Int
    add_bos::Bool
    add_space_prefix::Bool
    # merge-ranked vocabularies (gemma-4): (left, right) → rank; empty for
    # score-driven SPM
    merge_rank::Dict{Tuple{String, String}, Int}
end

ranked(t::Tokenizer) = !isempty(t.merge_rank)

# llama.cpp splits each merge at the first space after its first byte
function _merge_ranks(merges)
    ranks = Dict{Tuple{String, String}, Int}()
    for (rank, m) in enumerate(String.(merges))
        ncodeunits(m) < 2 && continue
        sp = findnext(==(' '), m, nextind(m, 1))
        sp === nothing && continue
        get!(ranks, (m[1:prevind(m, sp)], m[(sp + 1):end]), rank - 1)
    end
    return ranks
end

function Tokenizer(f::GGUFFile)
    pieces = String.(metadata(f, "tokenizer.ggml.tokens"))
    scores = Float32.(metadata(f, "tokenizer.ggml.scores", zeros(Float32, length(pieces))))
    gemma4 = metadata(f, "tokenizer.ggml.model", "llama") == "gemma4"
    gemma4 && !haskey(f.meta, "tokenizer.ggml.merges") &&
        error("gemma4 tokenizer metadata has no tokenizer.ggml.merges")
    types = Int32[p in LLAMA_CONTROL_LOOKING ? T_CONTROL : ty for (p, ty) in
                  zip(pieces, metadata(f, "tokenizer.ggml.token_type"))]
    # llama.cpp's gemma-4 workaround: beside <|tool_response>, </s> is text
    if "<|tool_response>" in pieces && "</s>" in pieces
        types[findfirst(==("</s>"), pieces)] = T_NORMAL
    end
    id_of = Dict{String, Int}(p => i - 1 for (i, p) in enumerate(pieces))
    specials = sort!([(p, i - 1, types[i]) for (i, p) in enumerate(pieces)
                      if types[i] in (T_CONTROL, T_USER, T_UNKNOWN)];
        by = s -> ncodeunits(first(s)), rev = true)
    byte_ids = fill(-1, 256)
    for b in 0:255
        id = get(id_of, "<0x" * uppercase(string(b; base = 16, pad = 2)) * ">", -1)
        byte_ids[b + 1] = id
    end
    return Tokenizer(pieces, scores, types, id_of, specials, byte_ids,
        Int(metadata(f, "tokenizer.ggml.bos_token_id", 1)),
        Int(metadata(f, "tokenizer.ggml.eos_token_id", 2)),
        Int(metadata(f, "tokenizer.ggml.unknown_token_id", 0)),
        gemma4 || Bool(metadata(f, "tokenizer.ggml.add_bos_token", true)),
        !gemma4 && Bool(metadata(f, "tokenizer.ggml.add_space_prefix", true)),
        gemma4 ? _merge_ranks(metadata(f, "tokenizer.ggml.merges")) :
            Dict{Tuple{String, String}, Int}())
end

# ---- a small max-heap of merge candidates -------------------------------------
# entries: (score, -left_index, left, right, merged_size) — tuple order gives
# llama.cpp's priority: highest score, then leftmost.

function heap_push!(h::Vector, x)
    push!(h, x)
    i = length(h)
    while i > 1
        p = i >> 1
        h[p] < h[i] || break
        h[p], h[i] = h[i], h[p]
        i = p
    end
end

function heap_pop!(h::Vector)
    top = h[1]
    h[1] = h[end]
    pop!(h)
    i, n = 1, length(h)
    while true
        c = 2i
        c > n && break
        c < n && h[c + 1] > h[c] && (c += 1)
        h[i] < h[c] || break
        h[i], h[c] = h[c], h[i]
        i = c
    end
    return top
end

# ---- the SPM merge over one fragment ------------------------------------------

mutable struct Sym
    start::Int   # byte index into the fragment
    n::Int       # byte length (0 = merged away)
    prev::Int
    next::Int
end

function spm_fragment!(out::Vector{Int}, t::Tokenizer, frag::AbstractString)
    isempty(frag) && return out
    s = replace(frag, ' ' => WS)
    syms = Sym[]
    i = firstindex(s)
    while i <= lastindex(s)
        j = nextind(s, i)
        push!(syms, Sym(i, j - i, length(syms), length(syms) + 2))
        i = j
    end
    syms[end].next = 0
    text_of(sy) = SubString(s, sy.start, prevind(s, sy.start + sy.n))
    heap = Tuple{Float32, Int, Int, Int, Int}[]
    function try_bigram(li, ri)
        (li < 1 || ri < 1) && return
        merged = SubString(s, syms[li].start,
            prevind(s, syms[ri].start + syms[ri].n))
        id = get(t.id_of, merged, -1)
        id < 0 && return
        heap_push!(heap, (t.scores[id + 1], -li, li, ri, ncodeunits(merged)))
    end
    for k in 1:(length(syms) - 1)
        try_bigram(k, k + 1)
    end
    while !isempty(heap)
        (_, _, li, ri, sz) = heap_pop!(heap)
        (syms[li].n == 0 || syms[ri].n == 0 ||
         syms[li].n + syms[ri].n != sz) && continue   # stale entry
        syms[li].n += syms[ri].n
        syms[ri].n = 0
        syms[li].next = syms[ri].next
        syms[ri].next > 0 && (syms[syms[ri].next].prev = li)
        try_bigram(syms[li].prev, li)
        try_bigram(li, syms[li].next)
    end
    k = 1
    while k > 0
        sy = syms[k]
        if sy.n > 0
            txt = text_of(sy)
            id = get(t.id_of, txt, -1)
            if id >= 0
                push!(out, id)
            else
                for b in codeunits(txt)
                    bid = t.byte_ids[Int(b) + 1]
                    push!(out, bid >= 0 ? bid : t.unk)
                end
            end
        end
        k = sy.next
    end
    return out
end

# ---- the merge-ranked (gemma-4) rule over one fragment -------------------------

# one symbol's text as its token, or its bytes as <0xXX> tokens (dropped when
# the vocabulary lacks them, as llama.cpp does)
function _emit!(out::Vector{Int}, t::Tokenizer, txt::AbstractString)
    id = get(t.id_of, txt, -1)
    if id >= 0
        push!(out, id)
    else
        for b in codeunits(txt)
            bid = t.byte_ids[Int(b) + 1]
            bid >= 0 && push!(out, bid)
        end
    end
    return out
end

function _ranked_word!(out::Vector{Int}, t::Tokenizer, word::AbstractString)
    if all(==('\n'), word) && haskey(t.id_of, word)
        return push!(out, t.id_of[word])      # a newline run that is a token
    end
    syms = Sym[]
    i = firstindex(word)
    while i <= lastindex(word)
        j = nextind(word, i)
        push!(syms, Sym(i, j - i, length(syms), length(syms) + 2))
        i = j
    end
    syms[end].next = 0
    text_of(k) = SubString(word, syms[k].start, prevind(word, syms[k].start + syms[k].n))
    # max-heap entries (-rank, -left, left, right, merged_size): lowest rank
    # first, leftmost on ties — llama.cpp's llm_bigram_bpe order
    heap = Tuple{Int, Int, Int, Int, Int}[]
    function try_bigram(li, ri)
        (li < 1 || ri < 1) && return
        rank = get(t.merge_rank, (String(text_of(li)), String(text_of(ri))), -1)
        rank < 0 && return
        heap_push!(heap, (-rank, -li, li, ri, syms[li].n + syms[ri].n))
    end
    for k in 1:(length(syms) - 1)
        try_bigram(k, k + 1)
    end
    while !isempty(heap)
        (_, _, li, ri, sz) = heap_pop!(heap)
        (syms[li].n == 0 || syms[ri].n == 0 ||
         syms[li].n + syms[ri].n != sz) && continue   # stale entry
        syms[li].n += syms[ri].n
        syms[ri].n = 0
        syms[li].next = syms[ri].next
        syms[ri].next > 0 && (syms[syms[ri].next].prev = li)
        try_bigram(syms[li].prev, li)
        try_bigram(li, syms[li].next)
    end
    k = 1
    while k > 0
        syms[k].n > 0 && _emit!(out, t, text_of(k))
        k = syms[k].next
    end
    return out
end

function ranked_fragment!(out::Vector{Int}, t::Tokenizer, frag::AbstractString)
    isempty(frag) && return out
    for m in eachmatch(r"[^\n]+|\n+", replace(frag, ' ' => WS))
        _ranked_word!(out, t, m.match)
    end
    return out
end

"""
    tokenize(t::Tokenizer, text; add_special = false, parse_special = true)

Tokenize to 0-based ids, mirroring llama.cpp for this vocabulary's merge
rule (score-driven SPM, or merge-ranked for gemma-4). `add_special`
prepends BOS when the model requests it; `parse_special` matches special
tokens (longest first) as single ids instead of tokenizing their text.
"""
function tokenize(t::Tokenizer, text::AbstractString;
        add_special::Bool = false, parse_special::Bool = true)
    out = Int[]
    add_special && t.add_bos && push!(out, t.bos)
    isempty(text) && return out
    first_frag = Ref(true)
    frag!(fr) = begin
        isempty(fr) && return
        pre = first_frag[] && t.add_space_prefix ? " " * fr : fr
        first_frag[] = false
        ranked(t) ? ranked_fragment!(out, t, pre) : spm_fragment!(out, t, pre)
    end
    if !isempty(t.specials)
        i = firstindex(text)
        fragstart = i
        while i <= lastindex(text)
            hit = 0
            for (p, id, ty) in t.specials
                (ty == T_USER || parse_special) || continue
                if startswith(SubString(text, i), p)
                    frag!(SubString(text, fragstart, prevind(text, i)))
                    push!(out, id)
                    first_frag[] = false
                    i += ncodeunits(p)
                    fragstart = i
                    hit = 1
                    break
                end
            end
            hit == 1 || (i = nextind(text, i))
        end
        frag!(SubString(text, fragstart, lastindex(text)))
    else
        frag!(text)
    end
    return out
end

"""
    piece(t::Tokenizer, id; special = true) -> String

Render one 0-based token id as text: byte tokens become their byte, ▁ becomes
space, control pieces render only when `special`.
"""
function piece(t::Tokenizer, id::Integer; special::Bool = true)
    p = t.pieces[id + 1]
    ty = t.types[id + 1]
    ty == T_BYTE && return String([parse(UInt8, p[2:(end - 1)])])
    ty in (T_CONTROL, T_UNKNOWN) && return special ? p : ""
    ty == T_USER && return p
    return replace(p, WS => " ")
end

detokenize(t::Tokenizer, ids; special::Bool = false) =
    join(piece(t, id; special) for id in ids)

end # module SPMTokenizer
