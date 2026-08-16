# Pure-Julia SentencePiece (SPM / "llama"-model) tokenizer, built entirely
# from the vocab, scores, and token types stored in the GGUF metadata. This
# removes the last load-bearing llama.cpp dependency from the pure inference
# path — llama.cpp's tokenizer remains only as a dev-time oracle
# (test_tokenizer.jl validates against it string-for-string).
#
# Semantics mirror llama.cpp's llm_tokenizer_spm: text is partitioned around
# special tokens (longest match first when parse_special), each fragment has
# its spaces escaped to ▁ (U+2581), is split into UTF-8 character symbols,
# and adjacent symbols are greedily merged — highest vocab score first,
# leftmost on ties — until no merge is possible; anything still unmatched
# falls back to the <0xXX> byte tokens.

module SPMTokenizer

using ..GGUF: GGUFFile, metadata

export Tokenizer, tokenize, detokenize, piece

const T_NORMAL = 1
const T_UNKNOWN = 2
const T_CONTROL = 3
const T_USER = 4
const T_BYTE = 6

const WS = "▁"   # ▁

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
end

function Tokenizer(f::GGUFFile)
    pieces = String.(metadata(f, "tokenizer.ggml.tokens"))
    scores = Float32.(metadata(f, "tokenizer.ggml.scores"))
    types = Int32.(metadata(f, "tokenizer.ggml.token_type"))
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
        Bool(metadata(f, "tokenizer.ggml.add_bos_token", true)),
        Bool(metadata(f, "tokenizer.ggml.add_space_prefix", true)))
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

"""
    tokenize(t::Tokenizer, text; add_special = false, parse_special = true)

Tokenize to 0-based ids, mirroring llama.cpp's SPM tokenizer. `add_special`
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
        spm_fragment!(out, t, pre)
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
