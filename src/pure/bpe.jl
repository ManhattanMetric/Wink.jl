# Pure-Julia GPT-2 byte-level BPE tokenizer ("gpt2" tokenizer model in GGUF
# metadata) — the OLMo/OLMoE family's tokenizer, and the second of the two
# tokenizer families the pure backend speaks (SPM being the first). Built
# entirely from the vocab, merges, and token types in the GGUF metadata;
# llama.cpp remains a dev-time oracle only (test_bpe.jl).
#
# Semantics mirror llama.cpp's llm_tokenizer_bpe: text is partitioned around
# special tokens with the same asymmetry the SPM oracle battery established
# (USER_DEFINED pieces — e.g. OLMoE's literal multi-space runs — match
# ALWAYS; CONTROL/UNKNOWN only under parse_special); each fragment is split
# into words by the pre-tokenizer regex; each word's bytes are mapped into
# the GPT-2 byte alphabet and merged pairwise by lowest merge rank.

module BPETokenizer

using ..GGUF: GGUFFile, metadata

export Tokenizer, tokenize, detokenize, piece

const T_NORMAL = 1
const T_UNKNOWN = 2
const T_CONTROL = 3
const T_USER = 4
const T_BYTE = 6

# the GPT-2 byte<->alphabet bijection: printable latin-1 maps to itself,
# everything else to characters starting at U+0100
function _byte_maps()
    printable = Set{Int}(vcat(Int('!'):Int('~'), Int('¡'):Int('¬'),
        Int('®'):Int('ÿ')))
    b2u = Vector{Char}(undef, 256)
    u2b = Dict{Char, UInt8}()
    n = 0
    for b in 0:255
        c = b in printable ? Char(b) : Char(256 + (n += 1) - 1)
        b2u[b + 1] = c
        u2b[c] = UInt8(b)
    end
    return b2u, u2b
end

# pre-tokenizer patterns by GGUF `tokenizer.ggml.pre`. "olmo" shares the
# classic GPT-2 pattern in llama.cpp; unknown pre types fall back to it
# with a warning rather than an error (the oracle battery is the arbiter).
function _pre_regex(pre::AbstractString)
    gpt2 = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    pre in ("gpt2", "olmo", "mpt", "jais") && return gpt2
    @warn "unrecognized BPE pre-tokenizer \"$pre\"; using the GPT-2 pattern"
    return gpt2
end

struct Tokenizer
    pieces::Vector{String}
    types::Vector{Int32}
    id_of::Dict{String, Int}          # piece (alphabet form) → 0-based id
    merge_rank::Dict{Tuple{String, String}, Int}
    specials::Vector{Tuple{String, Int, Int32}}   # longest-first raw matches
    b2u::Vector{Char}
    u2b::Dict{Char, UInt8}
    pre_regex::Regex
    bos::Int
    eos::Int
    add_bos::Bool
end

function Tokenizer(f::GGUFFile)
    pieces = String.(metadata(f, "tokenizer.ggml.tokens"))
    types = Int32.(metadata(f, "tokenizer.ggml.token_type"))
    id_of = Dict{String, Int}(p => i - 1 for (i, p) in enumerate(pieces))
    merge_rank = Dict{Tuple{String, String}, Int}()
    for (rank, m) in enumerate(String.(metadata(f, "tokenizer.ggml.merges")))
        sp = findfirst(' ', m)   # alphabet strings never contain a raw space
        merge_rank[(m[1:prevind(m, sp)], m[(sp + 1):end])] = rank
    end
    specials = sort!([(p, i - 1, types[i]) for (i, p) in enumerate(pieces)
                      if types[i] in (T_CONTROL, T_USER, T_UNKNOWN)];
        by = s -> ncodeunits(first(s)), rev = true)
    b2u, u2b = _byte_maps()
    return Tokenizer(pieces, types, id_of, merge_rank, specials, b2u, u2b,
        _pre_regex(String(metadata(f, "tokenizer.ggml.pre", "gpt2"))),
        Int(metadata(f, "tokenizer.ggml.bos_token_id", -1)),
        Int(metadata(f, "tokenizer.ggml.eos_token_id", -1)),
        Bool(metadata(f, "tokenizer.ggml.add_bos_token", false)))
end

# ---- the BPE merge over one pre-tokenized word --------------------------------

function _bpe_word!(out::Vector{Int}, t::Tokenizer, word::Vector{String})
    while length(word) > 1
        best, bi = typemax(Int), 0
        for i in 1:(length(word) - 1)
            r = get(t.merge_rank, (word[i], word[i + 1]), typemax(Int))
            if r < best
                best = r
                bi = i
            end
        end
        bi == 0 && break
        word[bi] *= word[bi + 1]
        deleteat!(word, bi + 1)
    end
    for w in word
        id = get(t.id_of, w, -1)
        if id >= 0
            push!(out, id)
        else
            for ch in w   # unreachable for a complete byte alphabet, but safe
                id2 = get(t.id_of, string(ch), -1)
                id2 >= 0 && push!(out, id2)
            end
        end
    end
    return out
end

function bpe_fragment!(out::Vector{Int}, t::Tokenizer, frag::AbstractString)
    for m in eachmatch(t.pre_regex, frag)
        word = [string(t.b2u[Int(b) + 1]) for b in codeunits(m.match)]
        _bpe_word!(out, t, word)
    end
    return out
end

"""
    tokenize(t::Tokenizer, text; add_special = false, parse_special = true)

Tokenize to 0-based ids, mirroring llama.cpp's BPE tokenizer. `add_special`
prepends BOS when the model requests it; `parse_special` matches
CONTROL/UNKNOWN special tokens as single ids (USER_DEFINED pieces always
match, per the oracle-established asymmetry).
"""
function tokenize(t::Tokenizer, text::AbstractString;
        add_special::Bool = false, parse_special::Bool = true)
    out = Int[]
    add_special && t.add_bos && t.bos >= 0 && push!(out, t.bos)
    isempty(text) && return out
    if !isempty(t.specials)
        i = firstindex(text)
        fragstart = i
        while i <= lastindex(text)
            hit = false
            for (p, id, ty) in t.specials
                (ty == T_USER || parse_special) || continue
                if startswith(SubString(text, i), p)
                    bpe_fragment!(out, t,
                        SubString(text, fragstart, prevind(text, i)))
                    push!(out, id)
                    i += ncodeunits(p)
                    fragstart = i
                    hit = true
                    break
                end
            end
            hit || (i = nextind(text, i))
        end
        bpe_fragment!(out, t, SubString(text, fragstart, lastindex(text)))
    else
        bpe_fragment!(out, t, text)
    end
    return out
end

"""
    piece(t::Tokenizer, id; special = true) -> String

Render one 0-based token id as text: normal pieces decode through the byte
alphabet, control pieces render only when `special`, user-defined pieces
(stored raw) render as themselves.
"""
function piece(t::Tokenizer, id::Integer; special::Bool = true)
    p = t.pieces[id + 1]
    ty = t.types[id + 1]
    ty in (T_CONTROL, T_UNKNOWN) && return special ? p : ""
    ty == T_USER && return p
    bytes = UInt8[]
    for ch in p
        b = get(t.u2b, ch, nothing)
        b === nothing ? append!(bytes, codeunits(string(ch))) : push!(bytes, b)
    end
    return String(bytes)
end

detokenize(t::Tokenizer, ids; special::Bool = false) =
    join(piece(t, id; special) for id in ids)

end # module BPETokenizer
