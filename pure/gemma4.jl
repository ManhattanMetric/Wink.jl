# Gemma-4 (26B-A4B class) inference in pure Julia — the daily-driver MoE,
# transcribed from llama.cpp b10405's src/models/gemma4.cpp (fetched and
# studied, not guessed) and validated against that exact build as oracle.
#
# The architecture, for the record, because little of it is guessable:
#   - attention scale is 1.0 — no 1/sqrt(head_dim)
#   - per-layer geometry: sliding-window layers use head_dim 256 with 8 KV
#     heads; every 6th layer is global with head_dim 512 and 2 KV heads,
#     rope base 1e6 and a shared rope_freqs FACTOR table (theta /= factor)
#   - Q and K get per-head RMS-norms; V gets a WEIGHTLESS RMS-norm
#   - QAT ".scale" tensors post-multiply their matmuls ({1} scalars for
#     dense weights, {n_expert} for expert stacks); ".input_scale" tensors
#     exist in the file but are NOT applied in the graph — oracle parity
#   - every layer is dense FFN + MoE in parallel: the dense (shared-expert)
#     branch and the 128-expert branch each have their own pre/post norms,
#     the router reads a THIRD view of the residual stream (weightless
#     rms-norm, scaled by 1/sqrt(n_embd), gated by a per-channel vector),
#     top-8 softmax weights are RENORMALIZED, experts are fused gate_up
#     (gate first half, up second) with GELU, and the branch outputs are
#     summed, post-normed once more, residual-added, then multiplied by a
#     per-layer scalar out_scale
#   - final logits soft-capped at 30: logits = 30 * tanh(logits / 30)
#   - token_embd is q6_K (gather only); output head is its own q4_0 matrix
#
# Expert weights are per-expert Q4_0Matrix views into the mmap — the 13.4GB
# file is never copied, and dequantization happens inside mul!.

module Gemma4

using LinearAlgebra
using KernelAbstractions
import Adapt

using ..GGUF: GGUFFile, tensor, metadata, raw_tensor
using ..Quant: Q4_0Matrix

export load_model, forward, generate, KVCache, step!

struct Layer
    is_swa::Bool
    hd::Int
    nh::Int
    nkv::Int
    attn_norm::AbstractVector{Float32}
    wq::AbstractMatrix{Float32}
    wk::AbstractMatrix{Float32}
    wv::Union{Nothing, AbstractMatrix{Float32}}  # absent on some layers: V = raw K projection
    wo::AbstractMatrix{Float32}
    wq_s::Float32
    wk_s::Float32
    wv_s::Float32
    wo_s::Float32
    q_norm::AbstractVector{Float32}
    k_norm::AbstractVector{Float32}
    attn_post_norm::AbstractVector{Float32}
    out_scale::Float32
    ffn_norm::AbstractVector{Float32}
    ffn_gate::AbstractMatrix{Float32}
    ffn_up::AbstractMatrix{Float32}
    ffn_down::AbstractMatrix{Float32}
    ffn_gate_s::Float32
    ffn_up_s::Float32
    ffn_down_s::Float32
    ffn_post_norm::AbstractVector{Float32}
    ffn_post_norm_1::AbstractVector{Float32}
    ffn_pre_norm_2::AbstractVector{Float32}
    ffn_post_norm_2::AbstractVector{Float32}
    gate_inp::AbstractMatrix{Float32}
    gate_inp_s::AbstractVector{Float32}
    gate_up_exps::Vector{<:AbstractMatrix{Float32}}   # per-expert [n_embd, 2n_ff]
    down_exps::Vector{<:AbstractMatrix{Float32}}      # per-expert [n_ff, n_embd]
    down_exps_s::Vector{Float32}
end

struct Model
    embd::AbstractMatrix{Float32}     # q6_K, gather only
    output::AbstractMatrix{Float32}   # separate head
    layers::Vector{Layer}
    out_norm::AbstractVector{Float32}
    rope_factors::AbstractVector{Float32}  # shared, global layers only
    n_embd::Int
    n_expert_used::Int
    eps::Float32
    rope_global::Float32
    rope_local::Float32
    n_swa::Int
    softcap::Float32
end

_scalar(f, name, default = 1.0f0) =
    haskey(f.tensors, name) ? Float32(tensor(f, name)[1]) : default

# per-expert quantized slabs as zero-copy views of a 3-D q4_0 tensor
function _expert_slabs(f::GGUFFile, name::AbstractString)
    bytes, dims, typ = raw_tensor(f, name)
    typ == 2 || error("$name: expected q4_0 experts, got ggml type $typ")
    nrow, ncol, ne = dims
    per = (nrow ÷ 32) * 18 * ncol
    return [Q4_0Matrix(view(bytes, ((e - 1) * per + 1):(e * per)), nrow, ncol)
            for e in 1:ne]
end

function load_model(f::GGUFFile)
    arch = metadata(f, "general.architecture")
    arch == "gemma4" || error("this backend implements gemma4; file is $arch")
    Int(metadata(f, "gemma4.embedding_length_per_layer_input", 0)) == 0 ||
        error("per-layer input embeddings (gemma-4 E-series) not supported")
    nl = Int(metadata(f, "gemma4.block_count"))
    swa_pattern = Bool.(metadata(f, "gemma4.attention.sliding_window_pattern"))
    hd_full = Int(metadata(f, "gemma4.attention.key_length"))
    hd_swa = Int(metadata(f, "gemma4.attention.key_length_swa"))
    layers = map(0:(nl - 1)) do i
        p = "blk.$i."
        swa = swa_pattern[i + 1]
        hd = swa ? hd_swa : hd_full
        wq = tensor(f, p * "attn_q.weight")
        wk = tensor(f, p * "attn_k.weight")
        Layer(swa, hd, size(wq, 2) ÷ hd, size(wk, 2) ÷ hd,
            vec(tensor(f, p * "attn_norm.weight")),
            wq, wk,
            haskey(f.tensors, p * "attn_v.weight") ?
                tensor(f, p * "attn_v.weight") : nothing,
            tensor(f, p * "attn_output.weight"),
            _scalar(f, p * "attn_q.scale"), _scalar(f, p * "attn_k.scale"),
            _scalar(f, p * "attn_v.scale"), _scalar(f, p * "attn_output.scale"),
            vec(tensor(f, p * "attn_q_norm.weight")),
            vec(tensor(f, p * "attn_k_norm.weight")),
            vec(tensor(f, p * "post_attention_norm.weight")),
            _scalar(f, p * "layer_output_scale.weight"),
            vec(tensor(f, p * "ffn_norm.weight")),
            tensor(f, p * "ffn_gate.weight"), tensor(f, p * "ffn_up.weight"),
            tensor(f, p * "ffn_down.weight"),
            _scalar(f, p * "ffn_gate.scale"), _scalar(f, p * "ffn_up.scale"),
            _scalar(f, p * "ffn_down.scale"),
            vec(tensor(f, p * "post_ffw_norm.weight")),
            vec(tensor(f, p * "post_ffw_norm_1.weight")),
            vec(tensor(f, p * "pre_ffw_norm_2.weight")),
            vec(tensor(f, p * "post_ffw_norm_2.weight")),
            Matrix{Float32}(tensor(f, p * "ffn_gate_inp.weight")),
            vec(tensor(f, p * "ffn_gate_inp.scale")),
            _expert_slabs(f, p * "ffn_gate_up_exps.weight"),
            _expert_slabs(f, p * "ffn_down_exps.weight"),
            haskey(f.tensors, p * "ffn_down_exps.scale") ?
                vec(tensor(f, p * "ffn_down_exps.scale")) :
                ones(Float32, length(_expert_slabs(f, p * "ffn_down_exps.weight"))))
    end
    embd = tensor(f, "token_embd.weight")
    return Model(embd,
        haskey(f.tensors, "output.weight") ?
            tensor(f, "output.weight") : embd,   # QAT ties the head to embd
        layers,
        vec(tensor(f, "output_norm.weight")),
        haskey(f.tensors, "rope_freqs.weight") ?
            vec(tensor(f, "rope_freqs.weight")) : Float32[],
        Int(metadata(f, "gemma4.embedding_length")),
        Int(metadata(f, "gemma4.expert_used_count")),
        Float32(metadata(f, "gemma4.attention.layer_norm_rms_epsilon")),
        Float32(metadata(f, "gemma4.rope.freq_base", 1.0f6)),
        Float32(metadata(f, "gemma4.rope.freq_base_swa", 1.0f4)),
        Int(metadata(f, "gemma4.attention.sliding_window")),
        Float32(metadata(f, "gemma4.final_logit_softcapping", 0.0f0)))
end

# ---- building blocks ----------------------------------------------------------

function norm1(X::AbstractArray, w::AbstractVector, eps)
    ms = sum(abs2, X; dims = 1) ./ Float32(size(X, 1))
    return (X .* w) ./ sqrt.(ms .+ eps)
end

# weightless RMS-norm (gemma-4 uses it on V and on the router input)
function norm0(X::AbstractArray, eps)
    ms = sum(abs2, X; dims = 1) ./ Float32(size(X, 1))
    return X ./ sqrt.(ms .+ eps)
end

gelu_tanh(x) = 0.5f0 * x * (1 + tanh(0.7978845608028654f0 * (x + 0.044715f0 * x^3)))

@kernel function _rope_kernel!(X, base::Float32, pos0::Int32, half::Int32,
        factors)
    i, h, t = @index(Global, NTuple)
    θ = Float32(pos0 + t - 1) * base^(-2.0f0 * (i - 1) / (2.0f0 * half)) / factors[i]
    c, s = cos(θ), sin(θ)
    x1 = X[i, h, t]
    x2 = X[i + half, h, t]
    X[i, h, t] = x1 * c - x2 * s
    X[i + half, h, t] = x1 * s + x2 * c
end

function rope!(X::AbstractArray{Float32, 3}, base::Float32, factors;
        pos0::Int = 0)
    half = size(X, 1) ÷ 2
    fac = length(factors) == half ? factors :
        fill!(KernelAbstractions.allocate(get_backend(X), Float32, half), 1.0f0)
    _rope_kernel!(get_backend(X))(X, base, Int32(pos0), Int32(half), fac;
        ndrange = (half, size(X, 2), size(X, 3)))
    return X
end

# Single-token attention as two fused kernels (one launch each) instead of a
# per-head loop of small matmuls — at 48 layers the launch count is what
# throttles the GPU, not the math. Scores mask the sliding window in-kernel.
@kernel function _attn1_scores!(S, @Const(K), @Const(q), hpk::Int32,
        wstart::Int32)
    pos, hh = @index(Global, NTuple)
    if pos < Int(wstart)
        @inbounds S[pos, hh] = -Inf32
    else
        kv = (hh - 1) ÷ Int(hpk) + 1
        s = 0.0f0
        @inbounds for d in 1:size(K, 1)
            s = muladd(K[d, kv, pos], q[d, hh], s)
        end
        @inbounds S[pos, hh] = s
    end
end

@kernel function _attn1_out!(att, @Const(V), @Const(S), hpk::Int32, hd::Int32)
    d, hh = @index(Global, NTuple)
    kv = (hh - 1) ÷ Int(hpk) + 1
    s = 0.0f0
    @inbounds for pos in 1:size(S, 1)
        s = muladd(V[d, kv, pos], S[pos, hh], s)
    end
    @inbounds att[(hh - 1) * Int(hd) + d, 1] = s
end

mutable struct KVCache{A <: AbstractArray{Float32, 3}}
    K::Vector{A}
    V::Vector{A}
    n::Int
    capacity::Int
end

KVCache(m::Model; capacity::Int = 4096) = KVCache(
    [fill!(similar(L.attn_norm, L.hd, L.nkv, capacity), 0.0f0) for L in m.layers],
    [fill!(similar(L.attn_norm, L.hd, L.nkv, capacity), 0.0f0) for L in m.layers],
    0, capacity)

# ---- the forward pass ---------------------------------------------------------

function step!(m::Model, c::KVCache, toks::Vector{<:Integer})
    T = length(toks)
    P = c.n
    P + T <= c.capacity ||
        error("KV cache full ($(c.capacity)); allocate a larger KVCache")
    inv_sqrt_embd = 1.0f0 / sqrt(Float32(m.n_embd))
    h0 = m.embd[:, collect(Int, toks) .+ 1] .* sqrt(Float32(m.n_embd))
    h = copyto!(similar(c.K[1], m.n_embd, T), h0)
    kq = reshape(0:(P + T - 1), :, 1)
    qq = reshape(P:(P + T - 1), 1, :)
    for (il, L) in enumerate(m.layers)
        base = L.is_swa ? m.rope_local : m.rope_global
        factors = L.is_swa ? Float32[] : m.rope_factors
        hd, nh, nkv = L.hd, L.nh, L.nkv
        xn = norm1(h, L.attn_norm, m.eps)
        q = rope!(norm1(reshape((L.wq' * xn) .* L.wq_s, hd, nh, T),
                L.q_norm, m.eps), base, factors; pos0 = P)
        k = rope!(norm1(reshape((L.wk' * xn) .* L.wk_s, hd, nkv, T),
                L.k_norm, m.eps), base, factors; pos0 = P)
        vsrc = L.wv === nothing ? (L.wk' * xn) .* L.wk_s :
                                  (L.wv' * xn) .* L.wv_s
        v = norm0(reshape(vsrc, hd, nkv, T), m.eps)
        c.K[il][:, :, (P + 1):(P + T)] = k
        c.V[il][:, :, (P + 1):(P + T)] = v
        att = similar(h, hd * nh, T)
        if T == 1
            Kp = P + 1
            wstart = L.is_swa ? max(1, Kp - m.n_swa + 1) : 1
            S = similar(h, Kp, nh)
            _attn1_scores!(get_backend(S))(S, c.K[il], reshape(q, hd, nh),
                Int32(nh ÷ nkv), Int32(wstart); ndrange = (Kp, nh))
            S = exp.(S .- maximum(S; dims = 1))
            S ./= sum(S; dims = 1)
            _attn1_out!(get_backend(att))(att, c.V[il], S, Int32(nh ÷ nkv),
                Int32(hd); ndrange = (hd, nh))
        else
            for hh in 1:nh
                kv = 1 + (hh - 1) * nkv ÷ nh
                Kc = @view c.K[il][:, kv, 1:(P + T)]
                Vc = @view c.V[il][:, kv, 1:(P + T)]
                S = Kc' * q[:, hh, :]          # attention scale is 1.0!
                S = L.is_swa ?
                    ifelse.((kq .<= qq) .& (qq .- kq .< m.n_swa), S, -Inf32) :
                    ifelse.(kq .<= qq, S, -Inf32)
                S = exp.(S .- maximum(S; dims = 1))
                S ./= sum(S; dims = 1)
                att[(1 + (hh - 1) * hd):(hh * hd), :] = Vc * S
            end
        end
        attno = (L.wo' * att) .* L.wo_s
        attn_out = norm1(attno, L.attn_post_norm, m.eps) .+ h
        # dense (shared-expert) branch
        mlpn = norm1(attn_out, L.ffn_norm, m.eps)
        mlp = (L.ffn_down' * (gelu_tanh.((L.ffn_gate' * mlpn) .* L.ffn_gate_s) .*
                              ((L.ffn_up' * mlpn) .* L.ffn_up_s))) .* L.ffn_down_s
        mlp = norm1(mlp, L.ffn_post_norm_1, m.eps)
        # router (its own view of the residual stream)
        tmp = (norm0(attn_out, m.eps) .* inv_sqrt_embd) .* L.gate_inp_s
        rlog = L.gate_inp' * tmp                       # n_expert × T
        p = exp.(rlog .- maximum(rlog; dims = 1))
        p ./= sum(p; dims = 1)
        pc = Array(p)                                  # routing is host-side
        # expert branch
        moe_in = norm1(attn_out, L.ffn_pre_norm_2, m.eps)
        moe = fill!(similar(attn_out), 0.0f0)
        nf = size(L.down_exps[1], 1)
        sel = [partialsortperm(view(pc, :, t), 1:m.n_expert_used; rev = true)
               for t in 1:T]
        wsum = [max(sum(pc[e, t] for e in sel[t]), 6.103515625f-5) for t in 1:T]
        if T == 1
            for e in sel[1]                            # scalar weights, no
                GU = L.gate_up_exps[e]' * moe_in       # gather/scatter/upload
                act = gelu_tanh.(view(GU, 1:nf, :)) .*
                      view(GU, (nf + 1):(2nf), :)
                moe .+= (L.down_exps[e]' * act) .*
                        (L.down_exps_s[e] * pc[e, 1] / wsum[1])
            end
        else
            for e in 1:length(L.gate_up_exps)
                cols = [t for t in 1:T if e in sel[t]]
                isempty(cols) && continue
                GU = L.gate_up_exps[e]' * moe_in[:, cols]  # 2n_ff × |cols|
                act = gelu_tanh.(view(GU, 1:nf, :)) .*
                      view(GU, (nf + 1):(2nf), :)
                Y = (L.down_exps[e]' * act) .* L.down_exps_s[e]
                w = copyto!(similar(Y, 1, length(cols)),
                    Float32[pc[e, t] / wsum[t] for t in cols])
                moe[:, cols] = moe[:, cols] .+ Y .* w  # gather/scatter, not a
            end                                        # view broadcast: GPU-safe
        end
        moe = norm1(moe, L.ffn_post_norm_2, m.eps)
        cur = norm1(mlp .+ moe, L.ffn_post_norm, m.eps) .+ attn_out
        h = cur .* L.out_scale
    end
    c.n += T
    logits = m.output' * norm1(h, m.out_norm, m.eps)
    m.softcap > 0 &&
        (logits = m.softcap .* tanh.(logits ./ m.softcap))
    return logits
end

# ---- device movement ----------------------------------------------------------
#
# `adapt(MtlArray, m)` moves the model to the GPU. Two fields stay host-side
# on purpose: embd (the q6_K table is GATHERED by getindex — scalar access,
# so it reads straight from the mmap) and down_exps_s (read as scalars during
# expert dispatch). The tied output head IS adapted — logits are a matmul.

Adapt.adapt_structure(to, L::Layer) = Layer(
    L.is_swa, L.hd, L.nh, L.nkv,
    Adapt.adapt(to, L.attn_norm),
    Adapt.adapt(to, L.wq), Adapt.adapt(to, L.wk),
    L.wv === nothing ? nothing : Adapt.adapt(to, L.wv),
    Adapt.adapt(to, L.wo),
    L.wq_s, L.wk_s, L.wv_s, L.wo_s,
    Adapt.adapt(to, L.q_norm), Adapt.adapt(to, L.k_norm),
    Adapt.adapt(to, L.attn_post_norm), L.out_scale,
    Adapt.adapt(to, L.ffn_norm),
    Adapt.adapt(to, L.ffn_gate), Adapt.adapt(to, L.ffn_up),
    Adapt.adapt(to, L.ffn_down),
    L.ffn_gate_s, L.ffn_up_s, L.ffn_down_s,
    Adapt.adapt(to, L.ffn_post_norm), Adapt.adapt(to, L.ffn_post_norm_1),
    Adapt.adapt(to, L.ffn_pre_norm_2), Adapt.adapt(to, L.ffn_post_norm_2),
    Adapt.adapt(to, L.gate_inp), Adapt.adapt(to, L.gate_inp_s),
    [Adapt.adapt(to, W) for W in L.gate_up_exps],
    [Adapt.adapt(to, W) for W in L.down_exps],
    L.down_exps_s)

Adapt.adapt_structure(to, m::Model) = Model(
    m.embd,
    Adapt.adapt(to, m.output),
    [Adapt.adapt(to, L) for L in m.layers],
    Adapt.adapt(to, m.out_norm), Adapt.adapt(to, m.rope_factors),
    m.n_embd, m.n_expert_used, m.eps, m.rope_global, m.rope_local,
    m.n_swa, m.softcap)

forward(m::Model, toks::Vector{<:Integer}) =
    step!(m, KVCache(m; capacity = length(toks)), collect(Int, toks))

function generate(m::Model, toks::Vector{<:Integer}; max_tokens::Int = 32,
        eog::Vector{Int} = Int[])
    c = KVCache(m; capacity = length(toks) + max_tokens)
    logits = step!(m, c, collect(Int, toks))
    out = Int[]
    for _ in 1:max_tokens
        next = Int(argmax(view(logits, :, size(logits, 2)))) - 1
        next in eog && break
        push!(out, next)
        logits = step!(m, c, [next])
    end
    return out
end

end # module Gemma4
