# Synthetic inference fixtures: a GGUF writer plus tiny, deterministic models
# and vocabularies for every architecture and tokenizer family Wink runs
# in-process. Nothing here needs model weights or llama.cpp. The same builders
# are used once, offline, to record llama.cpp's output for these exact files
# (test/fixtures/inference_goldens.jl), so the in-process engine is checked
# against the reference implementation without either being present in CI.
#
# Everything is generated from a SplitMix64 stream rather than Julia's RNG,
# whose output is not guaranteed stable across Julia versions — the goldens
# are only meaningful if the files are byte-identical everywhere.

module InferenceFixtures

using Wink

export SplitMix, write_gguf, fnv1a, spm_vocab_meta, bpe_vocab_meta,
    tiny_gemma3!, tiny_gemma4!, tiny_olmoe!, qblocks, QBLOCK

# ---- deterministic randomness -------------------------------------------------

mutable struct SplitMix
    s::UInt64
end

function next!(r::SplitMix)
    r.s += 0x9e3779b97f4a7c15
    z = r.s
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

unit(r::SplitMix) = Float32(next!(r) >> 40) / Float32(1 << 24)   # [0, 1)
sym(r::SplitMix, s) = s * (2unit(r) - 1)
byte(r::SplitMix) = UInt8(next!(r) & 0xff)

# FNV-1a over file bytes: a version-stable fingerprint for the goldens
function fnv1a(bytes::AbstractVector{UInt8})
    h = 0xcbf29ce484222325
    for b in bytes
        h = (h ⊻ b) * 0x00000100000001b3
    end
    return h
end

# ---- GGUF writer --------------------------------------------------------------

const TYPECODE = Dict{Type, UInt32}(UInt8 => 0, Int8 => 1, UInt16 => 2,
    Int16 => 3, UInt32 => 4, Int32 => 5, Float32 => 6, Bool => 7, String => 8,
    UInt64 => 10, Int64 => 11, Float64 => 12)

wstr(io, s) = (write(io, UInt64(ncodeunits(s))); write(io, s))

function wvalue(io, v)
    if v isa AbstractVector
        T = v isa Vector{String} ? String : eltype(v)
        write(io, TYPECODE[T])
        write(io, UInt64(length(v)))
        foreach(x -> wscalar(io, x), v)
    else
        wscalar(io, v)
    end
end
wscalar(io, v::String) = wstr(io, v)
wscalar(io, v::Bool) = write(io, UInt8(v))
wscalar(io, v) = write(io, v)
typecode(v) = v isa AbstractVector ? UInt32(9) : TYPECODE[typeof(v)]

"""
    write_gguf(path, meta, tensors; alignment = 32, version = 3)

`meta` is a vector of `key => value` pairs (value types select GGUF metadata
types; vectors become arrays). `tensors` is a vector of
`(name, typ, dims, bytes)` tuples, written in order with alignment padding.
"""
function write_gguf(path, meta, tensors; alignment::Int = 32, version = 3)
    open(path, "w") do io
        write(io, UInt32(0x46554747), UInt32(version))
        write(io, UInt64(length(tensors)), UInt64(length(meta)))
        for (k, v) in meta
            wstr(io, k)
            write(io, typecode(v))
            wvalue(io, v)
        end
        off = 0
        offsets = Int[]
        for (name, typ, dims, bytes) in tensors
            wstr(io, name)
            write(io, UInt32(length(dims)))
            foreach(d -> write(io, UInt64(d)), dims)
            write(io, UInt32(typ))
            write(io, UInt64(off))
            push!(offsets, off)
            off = cld(off + length(bytes), alignment) * alignment
        end
        pad(n) = write(io, zeros(UInt8, n))
        pad(cld(position(io), alignment) * alignment - position(io))
        start = position(io)
        for ((_, _, _, bytes), o) in zip(tensors, offsets)
            pad(start + o - position(io))
            write(io, bytes)
        end
    end
    return path
end

f32bytes(a) = collect(reinterpret(UInt8, vec(Float32.(a))))

# ---- quantized blocks ---------------------------------------------------------
#
# Random but VALID ggml blocks (finite f16 scales in a sane range). The dense
# twin of a quantized tensor is its exact dequantization, so dense and
# quantized variants of a model carry identical weights.

const QBLOCK = Dict(:q4_0 => (2, 32, 18), :q4_1 => (3, 32, 20),
    :q8_0 => (8, 32, 34), :q6_K => (14, 256, 210))

f16(x) = reinterpret(UInt8, [Float16(x)])

"""
    qblocks(r, kind, nrow, ncol; scale) -> bytes

`ncol` columns of `nrow` weights each (nrow a multiple of the block size),
column-major like every ggml tensor.
"""
function qblocks(r::SplitMix, kind::Symbol, nrow::Int, ncol::Int;
        scale::Float32 = 0.03f0)
    _, qk, bpb = QBLOCK[kind]
    nrow % qk == 0 || error("$kind needs rows divisible by $qk")
    out = UInt8[]
    for _ in 1:(ncol * (nrow ÷ qk))
        if kind === :q4_0
            append!(out, f16(scale * (0.5f0 + unit(r))))
            append!(out, [byte(r) for _ in 1:16])
        elseif kind === :q4_1
            d = scale * (0.5f0 + unit(r))
            append!(out, f16(d))
            append!(out, f16(-7.5f0 * d))
            append!(out, [byte(r) for _ in 1:16])
        elseif kind === :q8_0
            append!(out, f16(scale / 20 * (0.5f0 + unit(r))))
            append!(out, [reinterpret(UInt8, Int8(round(sym(r, 127f0))))
                          for _ in 1:32])
        else # :q6_K — ql, qh, 16 int8 sub-scales, then the f16 super-scale
            append!(out, [byte(r) for _ in 1:192])
            append!(out, [UInt8(8 + next!(r) % 56) for _ in 1:16])
            append!(out, f16(scale / 40 * (0.5f0 + unit(r))))
        end
    end
    return out
end

const QTYPE = Dict(:q4_0 => Wink.Quant.Q4_0Matrix, :q4_1 => Wink.Quant.Q4_1Matrix,
    :q8_0 => Wink.Quant.Q8_0Matrix, :q6_K => Wink.Quant.Q6_KMatrix)

# a weight tensor: quantized bytes, or (quant = false) its exact F32 twin
function weight(r, name, kind, nrow, ncol; quant::Bool, scale = 0.03f0)
    bytes = qblocks(r, kind, nrow, ncol; scale)
    quant && return (name, QBLOCK[kind][1], [nrow, ncol], bytes)
    dense = Array(QTYPE[kind](bytes, nrow, ncol))
    return (name, 0, [nrow, ncol], f32bytes(dense))
end

# an expert stack [nrow, ncol, ne]: per-expert blocks laid end to end
function experts(r, name, kind, nrow, ncol, ne; quant::Bool, scale = 0.03f0)
    slabs = [qblocks(r, kind, nrow, ncol; scale) for _ in 1:ne]
    quant && return (name, QBLOCK[kind][1], [nrow, ncol, ne], reduce(vcat, slabs))
    dense = reduce(vcat, [vec(Array(QTYPE[kind](s, nrow, ncol))) for s in slabs])
    return (name, 0, [nrow, ncol, ne], f32bytes(dense))
end

normw(r, name, n) = (name, 0, [n], f32bytes([1 + sym(r, 0.2f0) for _ in 1:n]))
vec32(r, name, n, s) = (name, 0, [n], f32bytes([sym(r, s) for _ in 1:n]))
dense(r, name, nrow, ncol, s) =
    (name, 0, [nrow, ncol], f32bytes([sym(r, s) for _ in 1:(nrow * ncol)]))

# ---- vocabularies ---------------------------------------------------------------

const SPM_SPECIALS = [("<pad>", 3), ("<eos>", 3), ("<bos>", 3), ("<unk>", 2),
    ("<start_of_turn>", 3), ("<end_of_turn>", 3), ("  ", 4), ("   ", 4)]

const SPM_PIECES = ["▁", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
    "k", "l", "m", "n", "o", "p", "r", "s", "t", "u", "w", "y", ".", ",", "!",
    "=", "(", ")", "1", "2", "3", "▁t", "he", "▁the", "▁a", "in", "▁in", "ll",
    "lo", "▁h", "hel", "hello", "▁hello", "or", "ld", "▁w", "wor", "world",
    "▁world", "un", "▁f", "▁fun", "ct", "on", "ion", "ction", "▁function",
    "ul", "ia", "▁j", "▁jul", "▁julia", "▁=", "12", "123"]

# gemma-4 vocabularies are merge-ranked (llama.cpp loads them as BPE over
# SPM-style pieces); each multi-symbol piece above as one binary merge
const SPM_MERGES = ["▁ t", "h e", "▁t he", "▁ a", "i n", "▁ in", "l l", "l o",
    "▁ h", "he l", "hel lo", "▁ hello", "o r", "l d", "▁ w", "w or", "wor ld",
    "▁ world", "u n", "▁ f", "▁f un", "c t", "o n", "i on", "ct ion",
    "▁fun ction", "u l", "i a", "▁ j", "▁j ul", "▁jul ia", "▁ =", "1 2", "12 3"]

"""
    spm_vocab_meta(; model = "llama") -> (meta pairs, n_vocab)

A small SentencePiece vocabulary exercising control, unknown, user-defined
(gemma-style literal space runs), byte-fallback, and scored normal pieces.
"""
function spm_vocab_meta(; model::String = "llama")
    toks = String[]
    types = Int32[]
    scores = Float32[]
    for (p, t) in SPM_SPECIALS
        push!(toks, p); push!(types, t); push!(scores, 0)
    end
    for b in 0:255
        push!(toks, "<0x" * uppercase(string(b; base = 16, pad = 2)) * ">")
        push!(types, 6); push!(scores, 0)
    end
    for (i, p) in enumerate(SPM_PIECES)
        push!(toks, p); push!(types, 1)
        push!(scores, Float32(length(p)) - 0.001f0 * i)
    end
    meta = Pair{String, Any}[
        "tokenizer.ggml.model" => model,
        "tokenizer.ggml.pre" => "default",
        "tokenizer.ggml.tokens" => toks,
        "tokenizer.ggml.scores" => scores,
        "tokenizer.ggml.token_type" => types,
        "tokenizer.ggml.bos_token_id" => UInt32(2),
        "tokenizer.ggml.eos_token_id" => UInt32(1),
        "tokenizer.ggml.unknown_token_id" => UInt32(3),
        "tokenizer.ggml.padding_token_id" => UInt32(0),
        "tokenizer.ggml.add_bos_token" => true,
        "tokenizer.ggml.add_space_prefix" => false]
    model == "gemma4" && push!(meta, "tokenizer.ggml.merges" => SPM_MERGES)
    return meta, length(toks)
end

# GPT-2's byte-to-alphabet bijection, written out independently of Wink's
function gpt2_alphabet()
    keep = vcat(Int('!'):Int('~'), Int('¡'):Int('¬'), Int('®'):Int('ÿ'))
    out = Vector{Char}(undef, 256)
    n = 0
    for b in 0:255
        out[b + 1] = b in keep ? Char(b) : Char(256 + (n += 1) - 1)
    end
    return out
end

const BPE_MERGES = ["Ġ t", "h e", "Ġt he", "l l", "he ll", "hell o", "Ġ w",
    "o r", "Ġw or", "l d", "Ġwor ld", "Ġ hello", "i n", "Ġ in", "Ġ j", "u l",
    "Ġj ul", "i a", "Ġjul ia", "1 2", "12 3", "' s"]

"""
    bpe_vocab_meta() -> (meta pairs, n_vocab)

A small GPT-2 byte-level BPE vocabulary: the full 256-symbol alphabet, a
merge ladder, control specials, and a user-defined space run.
"""
function bpe_vocab_meta()
    toks = ["<|endoftext|>", "<|padding|>", "   "]
    types = Int32[3, 3, 4]
    for c in gpt2_alphabet()
        push!(toks, string(c)); push!(types, 1)
    end
    for m in BPE_MERGES
        push!(toks, replace(m, " " => "")); push!(types, 1)
    end
    meta = Pair{String, Any}[
        "tokenizer.ggml.model" => "gpt2",
        "tokenizer.ggml.pre" => "olmo",
        "tokenizer.ggml.tokens" => toks,
        "tokenizer.ggml.token_type" => types,
        "tokenizer.ggml.merges" => BPE_MERGES,
        "tokenizer.ggml.bos_token_id" => UInt32(0),
        "tokenizer.ggml.eos_token_id" => UInt32(0),
        "tokenizer.ggml.padding_token_id" => UInt32(1),
        "tokenizer.ggml.add_bos_token" => false]
    return meta, length(toks)
end

# ---- tiny models ----------------------------------------------------------------

"""
    tiny_gemma3!(path; quant = false, seed = 0x6733) -> path

Six layers (the sixth global, the rest sliding-window with a 4-token
window), MQA with head_dim 32, tied embeddings. `quant = true` stores the
same weights as q4_0 (attention/FFN) and q8_0 (embeddings).
"""
function tiny_gemma3!(path; quant::Bool = false, seed::Integer = 0x6733)
    r = SplitMix(UInt64(seed))
    ne, hd, nh, nkv, nff, nl = 64, 32, 2, 1, 128, 6
    vmeta, nv = spm_vocab_meta()
    meta = vcat(Pair{String, Any}[
        "general.architecture" => "gemma3",
        "gemma3.context_length" => UInt32(128),
        "gemma3.embedding_length" => UInt32(ne),
        "gemma3.block_count" => UInt32(nl),
        "gemma3.feed_forward_length" => UInt32(nff),
        "gemma3.attention.head_count" => UInt32(nh),
        "gemma3.attention.head_count_kv" => UInt32(nkv),
        "gemma3.attention.key_length" => UInt32(hd),
        "gemma3.attention.value_length" => UInt32(hd),
        "gemma3.attention.layer_norm_rms_epsilon" => 1.0f-6,
        "gemma3.attention.sliding_window" => UInt32(4),
        "gemma3.rope.freq_base" => 1.0f6], vmeta)
    ts = Any[weight(r, "token_embd.weight", :q8_0, ne, nv; quant, scale = 0.2f0),
        normw(r, "output_norm.weight", ne)]
    for i in 0:(nl - 1)
        p = "blk.$i."
        append!(ts, [normw(r, p * "attn_norm.weight", ne),
            weight(r, p * "attn_q.weight", :q4_0, ne, hd * nh; quant),
            weight(r, p * "attn_k.weight", :q4_0, ne, hd * nkv; quant),
            weight(r, p * "attn_v.weight", :q4_0, ne, hd * nkv; quant),
            weight(r, p * "attn_output.weight", :q4_0, hd * nh, ne; quant),
            normw(r, p * "attn_q_norm.weight", hd),
            normw(r, p * "attn_k_norm.weight", hd),
            normw(r, p * "post_attention_norm.weight", ne),
            normw(r, p * "ffn_norm.weight", ne),
            weight(r, p * "ffn_gate.weight", :q4_0, ne, nff; quant),
            weight(r, p * "ffn_up.weight", :q4_0, ne, nff; quant),
            weight(r, p * "ffn_down.weight", :q4_0, nff, ne; quant),
            normw(r, p * "post_ffw_norm.weight", ne)])
    end
    return write_gguf(path, meta, ts)
end

"""
    tiny_gemma4!(path; seed = 0x6734) -> path

The 26B-A4B shape in miniature: five sliding-window layers (head_dim 32,
2 KV heads) and one global layer (head_dim 64, 1 KV head, no attn_v, rope
factor table), dense FFN beside a 4-expert top-2 MoE, QAT scale tensors,
per-layer output scale, logit softcap, tied q6_K embeddings.
"""
function tiny_gemma4!(path; seed::Integer = 0x6734)
    r = SplitMix(UInt64(seed))
    ne, nh, nff, nfe, nex, nl = 256, 4, 128, 64, 4, 6
    swa = Bool[1, 1, 1, 1, 1, 0]
    vmeta, nv = spm_vocab_meta(; model = "gemma4")
    meta = vcat(Pair{String, Any}[
        "general.architecture" => "gemma4",
        "gemma4.context_length" => UInt32(128),
        "gemma4.embedding_length" => UInt32(ne),
        "gemma4.embedding_length_per_layer_input" => UInt32(0),
        "gemma4.block_count" => UInt32(nl),
        "gemma4.feed_forward_length" => UInt32(nff),
        "gemma4.expert_feed_forward_length" => UInt32(nfe),
        "gemma4.expert_count" => UInt32(nex),
        "gemma4.expert_used_count" => UInt32(2),
        "gemma4.attention.head_count" => UInt32(nh),
        "gemma4.attention.head_count_kv" => Int32[s ? 2 : 1 for s in swa],
        "gemma4.attention.key_length" => UInt32(64),
        "gemma4.attention.value_length" => UInt32(64),
        "gemma4.attention.key_length_swa" => UInt32(32),
        "gemma4.attention.value_length_swa" => UInt32(32),
        "gemma4.attention.layer_norm_rms_epsilon" => 1.0f-6,
        "gemma4.attention.shared_kv_layers" => UInt32(0),
        "gemma4.attention.sliding_window" => UInt32(4),
        "gemma4.attention.sliding_window_pattern" => swa,
        "gemma4.rope.dimension_count" => UInt32(64),
        "gemma4.rope.dimension_count_swa" => UInt32(32),
        "gemma4.rope.freq_base" => 1.0f6,
        "gemma4.rope.freq_base_swa" => 1.0f4,
        "gemma4.final_logit_softcapping" => 30.0f0], vmeta)
    ts = Any[weight(r, "token_embd.weight", :q6_K, ne, nv; quant = true,
            scale = 0.02f0),
        normw(r, "output_norm.weight", ne),
        ("rope_freqs.weight", 0, [32], f32bytes([1 + unit(r) for _ in 1:32]))]
    for (i, s) in enumerate(swa)
        p = "blk.$(i - 1)."
        hd, nkv = s ? (32, 2) : (64, 1)
        append!(ts, [normw(r, p * "attn_norm.weight", ne),
            weight(r, p * "attn_q.weight", :q4_0, ne, hd * nh; quant = true),
            weight(r, p * "attn_k.weight", :q4_0, ne, hd * nkv; quant = true)])
        s && push!(ts, weight(r, p * "attn_v.weight", :q4_0, ne, hd * nkv;
            quant = true))
        append!(ts, [weight(r, p * "attn_output.weight", :q4_0, hd * nh, ne;
                quant = true),
            normw(r, p * "attn_q_norm.weight", hd),
            normw(r, p * "attn_k_norm.weight", hd),
            normw(r, p * "post_attention_norm.weight", ne),
            (p * "layer_output_scale.weight", 0, [1], f32bytes([0.8f0])),
            normw(r, p * "ffn_norm.weight", ne),
            weight(r, p * "ffn_gate.weight", :q4_0, ne, nff; quant = true),
            weight(r, p * "ffn_up.weight", :q4_0, ne, nff; quant = true),
            weight(r, p * "ffn_down.weight", :q4_0, nff, ne; quant = true),
            normw(r, p * "post_ffw_norm.weight", ne),
            normw(r, p * "post_ffw_norm_1.weight", ne),
            normw(r, p * "pre_ffw_norm_2.weight", ne),
            normw(r, p * "post_ffw_norm_2.weight", ne),
            dense(r, p * "ffn_gate_inp.weight", ne, nex, 0.5f0),
            (p * "ffn_gate_inp.scale", 0, [ne],
                f32bytes([1 + sym(r, 0.1f0) for _ in 1:ne])),
            experts(r, p * "ffn_gate_up_exps.weight", :q4_0, ne, 2nfe, nex;
                quant = true),
            experts(r, p * "ffn_down_exps.weight", :q4_0, nfe, ne, nex;
                quant = true),
            (p * "ffn_down_exps.scale", 0, [nex],
                f32bytes([1 + sym(r, 0.2f0) for _ in 1:nex]))])
    end
    return write_gguf(path, meta, ts)
end

"""
    tiny_olmoe!(path; quant = false, seed = 0x0105) -> path

Two layers, 4 experts routed top-2, full-width QK-norms, untied head.
`quant = true` stores the same weights as q4_0 (with layer 0's
ffn_down_exps as q4_1, as llama.cpp's mixed recipes do) and a q8_0 head.
"""
function tiny_olmoe!(path; quant::Bool = false, seed::Integer = 0x0105)
    r = SplitMix(UInt64(seed))
    ne, nh, nff, nex, nl = 64, 2, 64, 4, 2
    vmeta, nv = bpe_vocab_meta()
    meta = vcat(Pair{String, Any}[
        "general.architecture" => "olmoe",
        "olmoe.context_length" => UInt32(128),
        "olmoe.embedding_length" => UInt32(ne),
        "olmoe.block_count" => UInt32(nl),
        "olmoe.feed_forward_length" => UInt32(nff),
        "olmoe.expert_count" => UInt32(nex),
        "olmoe.expert_used_count" => UInt32(2),
        "olmoe.attention.head_count" => UInt32(nh),
        "olmoe.attention.head_count_kv" => UInt32(nh),
        "olmoe.attention.layer_norm_rms_epsilon" => 1.0f-5,
        "olmoe.rope.freq_base" => 1.0f4], vmeta)
    ts = Any[weight(r, "token_embd.weight", :q4_0, ne, nv; quant, scale = 0.2f0),
        weight(r, "output.weight", :q8_0, ne, nv; quant, scale = 0.3f0),
        normw(r, "output_norm.weight", ne)]
    for i in 0:(nl - 1)
        p = "blk.$i."
        append!(ts, [normw(r, p * "attn_norm.weight", ne),
            weight(r, p * "attn_q.weight", :q4_0, ne, ne; quant),
            weight(r, p * "attn_k.weight", :q4_0, ne, ne; quant),
            weight(r, p * "attn_v.weight", :q4_0, ne, ne; quant),
            weight(r, p * "attn_output.weight", :q4_0, ne, ne; quant),
            normw(r, p * "attn_q_norm.weight", ne),
            normw(r, p * "attn_k_norm.weight", ne),
            normw(r, p * "ffn_norm.weight", ne),
            dense(r, p * "ffn_gate_inp.weight", ne, nex, 0.5f0),
            experts(r, p * "ffn_gate_exps.weight", :q4_0, ne, nff, nex; quant),
            experts(r, p * "ffn_up_exps.weight", :q4_0, ne, nff, nex; quant),
            experts(r, p * "ffn_down_exps.weight", i == 0 ? :q4_1 : :q4_0,
                nff, ne, nex; quant)])
    end
    return write_gguf(path, meta, ts)
end

end # module InferenceFixtures
