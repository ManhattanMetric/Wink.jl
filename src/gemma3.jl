# Gemma-3 (dense) inference in generic-array Julia. The forward pass is
# written against AbstractArray + broadcast + matmul, so the SAME code runs
# on CPU `Array`s and on any GPUArrays backend (MtlArray, CuArray, ROCArray,
# oneArray) — move a model with `Adapt.adapt(MtlArray, m)` and everything
# follows. The one custom kernel (RoPE) is KernelAbstractions, compiled per
# backend, CPU included: the pattern future quantized-expert kernels will
# use. No scalar indexing anywhere on the hot path, so GPU array types run
# it without fallback warnings.
#
# Architecture notes (oracle-verified against llama.cpp, see run.jl and
# test_swa.jl): scaled embeddings; ready-to-multiply RMSNorm weights; MQA
# with per-head QK-norm; NEOX RoPE with split bases (1e6 global every 6th
# layer, 1e4 sliding-window local); sandwich post-norms; GELU-tanh gated
# FFN; tied output head. Sliding-window attention is masking over cached
# positions; generation is KV-cached via `step!`.

module Gemma3

using LinearAlgebra
using KernelAbstractions
import Adapt

using ..GGUF: GGUFFile, tensor, metadata

export load_model, forward, generate, KVCache, step!

struct Layer
    attn_norm::AbstractVector{Float32}
    wq::AbstractMatrix{Float32}
    wk::AbstractMatrix{Float32}
    wv::AbstractMatrix{Float32}
    wo::AbstractMatrix{Float32}
    q_norm::AbstractVector{Float32}
    k_norm::AbstractVector{Float32}
    post_attn_norm::AbstractVector{Float32}
    ffn_norm::AbstractVector{Float32}
    w_gate::AbstractMatrix{Float32}
    w_up::AbstractMatrix{Float32}
    w_down::AbstractMatrix{Float32}
    post_ffn_norm::AbstractVector{Float32}
end
Adapt.@adapt_structure Layer

struct Model
    embd::AbstractMatrix{Float32}  # n_embd × vocab (tied output head)
    layers::Vector{Layer}
    out_norm::AbstractVector{Float32}
    n_head::Int
    n_kv::Int
    head_dim::Int
    eps::Float32
    rope_global::Float32
    rope_local::Float32
    swa_every::Int
    n_swa::Int
end

# layers is a plain Vector of structs: map adapt over it rather than trying
# to turn it into a device array
Adapt.adapt_structure(to, m::Model) = Model(Adapt.adapt(to, m.embd),
    [Adapt.adapt(to, L) for L in m.layers], Adapt.adapt(to, m.out_norm),
    m.n_head, m.n_kv, m.head_dim, m.eps, m.rope_global, m.rope_local,
    m.swa_every, m.n_swa)

function load_model(f::GGUFFile)
    arch = metadata(f, "general.architecture")
    arch == "gemma3" || error("this backend implements gemma3; file is $arch")
    nl = Int(metadata(f, "gemma3.block_count"))
    layers = map(0:(nl - 1)) do i
        p = "blk.$i."
        Layer(vec(tensor(f, p * "attn_norm.weight")),
            tensor(f, p * "attn_q.weight"), tensor(f, p * "attn_k.weight"),
            tensor(f, p * "attn_v.weight"), tensor(f, p * "attn_output.weight"),
            vec(tensor(f, p * "attn_q_norm.weight")),
            vec(tensor(f, p * "attn_k_norm.weight")),
            vec(tensor(f, p * "post_attention_norm.weight")),
            vec(tensor(f, p * "ffn_norm.weight")),
            tensor(f, p * "ffn_gate.weight"), tensor(f, p * "ffn_up.weight"),
            tensor(f, p * "ffn_down.weight"),
            vec(tensor(f, p * "post_ffw_norm.weight")))
    end
    return Model(tensor(f, "token_embd.weight"), layers,
        vec(tensor(f, "output_norm.weight")),
        Int(metadata(f, "gemma3.attention.head_count")),
        Int(metadata(f, "gemma3.attention.head_count_kv")),
        Int(metadata(f, "gemma3.attention.key_length")),
        Float32(metadata(f, "gemma3.attention.layer_norm_rms_epsilon")),
        Float32(metadata(f, "gemma3.rope.freq_base", 1.0f6)),
        Float32(metadata(f, "gemma3.rope.freq_base_swa", 1.0f4)),
        6,
        Int(metadata(f, "gemma3.attention.sliding_window")))
end

# ---- generic building blocks --------------------------------------------------

# RMSNorm over dim 1, broadcast across everything else — works for n_embd × T
# matrices and head_dim × heads × T arrays alike.
function norm1(X::AbstractArray, w::AbstractVector, eps)
    ms = sum(abs2, X; dims = 1) ./ Float32(size(X, 1))
    return (X .* w) ./ sqrt.(ms .+ eps)
end

gelu_tanh(x) = 0.5f0 * x * (1 + tanh(0.7978845608028654f0 * (x + 0.044715f0 * x^3)))

# NEOX RoPE as a KernelAbstractions kernel: one instance per (rot-pair, head,
# position), compiled for whichever backend owns X.
@kernel function _rope_kernel!(X, base::Float32, pos0::Int32, half::Int32)
    i, h, t = @index(Global, NTuple)
    θ = Float32(pos0 + t - 1) * base^(-2.0f0 * (i - 1) / (2.0f0 * half))
    c, s = cos(θ), sin(θ)
    x1 = X[i, h, t]
    x2 = X[i + half, h, t]
    X[i, h, t] = x1 * c - x2 * s
    X[i + half, h, t] = x1 * s + x2 * c
end

function rope!(X::AbstractArray{Float32, 3}, base::Float32; pos0::Int = 0)
    half = size(X, 1) ÷ 2
    _rope_kernel!(get_backend(X))(X, base, Int32(pos0), Int32(half);
        ndrange = (half, size(X, 2), size(X, 3)))
    return X
end

# ---- KV cache -----------------------------------------------------------------

"""
    KVCache(m::Model; capacity = 4096)

Per-layer key/value cache, allocated on the same device as the model's
arrays. `n` counts the positions already decoded.
"""
mutable struct KVCache{A <: AbstractArray{Float32, 3}}
    K::Vector{A}
    V::Vector{A}
    n::Int
    capacity::Int
end

KVCache(m::Model; capacity::Int = 4096) = KVCache(
    [fill!(similar(m.embd, m.head_dim, m.n_kv, capacity), 0.0f0) for _ in m.layers],
    [fill!(similar(m.embd, m.head_dim, m.n_kv, capacity), 0.0f0) for _ in m.layers],
    0, capacity)

# device-resident index vector for the embedding gather
function _device_ints(ref::AbstractArray, xs::Vector{Int})
    out = KernelAbstractions.allocate(get_backend(ref), Int, length(xs))
    copyto!(out, xs)
    return out
end

# ---- the forward pass ---------------------------------------------------------

"""
    step!(m::Model, c::KVCache, toks::Vector{<:Integer}) -> logits (vocab × T)

Decode a chunk of new (0-based) token ids against the cache. Attention is
batched matmul per head; causal and sliding-window constraints are one
broadcasted mask over (key position, query position).
"""
function step!(m::Model, c::KVCache, toks::Vector{<:Integer})
    T = length(toks)
    P = c.n
    P + T <= c.capacity ||
        error("KV cache full ($(c.capacity)); allocate a larger KVCache")
    hd, nh, nkv = m.head_dim, m.n_head, m.n_kv
    scale = 1.0f0 / sqrt(Float32(hd))
    idx = _device_ints(m.embd, collect(Int, toks) .+ 1)
    h = m.embd[:, idx] .* sqrt(Float32(size(m.embd, 1)))
    kq = reshape(0:(P + T - 1), :, 1)          # key positions (0-based)
    qq = reshape(P:(P + T - 1), 1, :)          # query positions
    for (il, L) in enumerate(m.layers)
        loc = il % m.swa_every != 0
        base = loc ? m.rope_local : m.rope_global
        xn = norm1(h, L.attn_norm, m.eps)
        q = rope!(norm1(reshape(L.wq' * xn, hd, nh, T), L.q_norm, m.eps),
            base; pos0 = P)
        k = rope!(norm1(reshape(L.wk' * xn, hd, nkv, T), L.k_norm, m.eps),
            base; pos0 = P)
        c.K[il][:, :, (P + 1):(P + T)] = k
        c.V[il][:, :, (P + 1):(P + T)] = reshape(L.wv' * xn, hd, nkv, T)
        att = similar(h, hd * nh, T)
        for hh in 1:nh
            kv = 1 + (hh - 1) * nkv ÷ nh       # shared kv head (GQA/MQA)
            Kc = @view c.K[il][:, kv, 1:(P + T)]
            Vc = @view c.V[il][:, kv, 1:(P + T)]
            S = (Kc' * q[:, hh, :]) .* scale   # (P+T) × T
            # mask FUSED into one device broadcast: the ranges are isbits and
            # ride into the kernel; a separate range-only broadcast would
            # materialize a CPU BitMatrix and poison GPU compilation
            S = loc ?
                ifelse.((kq .<= qq) .& (qq .- kq .< m.n_swa), S, -Inf32) :
                ifelse.(kq .<= qq, S, -Inf32)
            S = exp.(S .- maximum(S; dims = 1))
            S ./= sum(S; dims = 1)
            att[(1 + (hh - 1) * hd):(hh * hd), :] = Vc * S
        end
        h .+= norm1(L.wo' * att, L.post_attn_norm, m.eps)
        xn2 = norm1(h, L.ffn_norm, m.eps)
        act = gelu_tanh.(L.w_gate' * xn2) .* (L.w_up' * xn2)
        h .+= norm1(L.w_down' * act, L.post_ffn_norm, m.eps)
    end
    c.n += T
    return m.embd' * norm1(h, m.out_norm, m.eps)
end

"""
    forward(m::Model, toks::Vector{<:Integer}) -> logits (vocab × T)

Full forward over a sequence (fresh cache); every position's logits — the
oracle-comparison entry point.
"""
forward(m::Model, toks::Vector{<:Integer}) =
    step!(m, KVCache(m; capacity = length(toks)), collect(Int, toks))

"""
    generate(m::Model, toks; max_tokens = 32, eog = Int[]) -> Vector{Int}

Greedy generation with a KV cache: one prefill, then one step per token.
"""
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

end # module Gemma3
