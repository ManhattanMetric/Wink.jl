# Gemma-3 (dense) inference in pure Julia: plain arrays, LinearAlgebra, and
# ~150 lines of the actual math. Weights come from the pure GGUF reader; the
# only ambition here is CORRECTNESS, validated logit-for-logit against
# llama.cpp on the same file (see run.jl). No KV cache — generation re-runs
# the full forward on the growing sequence, which at 270M parameters is cheap
# and keeps every intermediate inspectable.
#
# Architecture notes (verified against the GGUF metadata and llama.cpp's
# gemma3 graph):
#   - embeddings scaled by sqrt(n_embd)
#   - RMSNorm weights are stored ready-to-multiply (conversion folds the +1)
#   - GQA/MQA: n_head queries share n_head_kv key/value heads
#   - per-head QK-norm before RoPE
#   - NEOX-style RoPE; base 1e6 on global layers, 10k on sliding-window
#     layers (every 6th layer is global, the rest are local)
#   - sandwich norms: post-attention and post-FFN RMSNorms before residuals
#   - GELU(tanh) gated FFN; tied output head
#
# Generation is KV-cached: `step!` decodes a chunk of new tokens against the
# cache, and sliding-window attention on local layers is a SLICE of the
# cached keys/values (the window's lower bound), so sequences past the
# 512-token window are handled exactly — validated against the llama.cpp
# oracle on long prompts (test_swa.jl).

module Gemma3

using LinearAlgebra

using ..GGUF: GGUFFile, tensor, metadata

export load_model, forward, generate, KVCache, step!

struct Layer
    attn_norm::Vector{Float32}
    wq::Matrix{Float32}
    wk::Matrix{Float32}
    wv::Matrix{Float32}
    wo::Matrix{Float32}
    q_norm::Vector{Float32}
    k_norm::Vector{Float32}
    post_attn_norm::Vector{Float32}
    ffn_norm::Vector{Float32}
    w_gate::Matrix{Float32}
    w_up::Matrix{Float32}
    w_down::Matrix{Float32}
    post_ffn_norm::Vector{Float32}
end

struct Model
    embd::Matrix{Float32}          # n_embd × vocab (tied output head)
    layers::Vector{Layer}
    out_norm::Vector{Float32}
    n_head::Int
    n_kv::Int
    head_dim::Int
    eps::Float32
    rope_global::Float32
    rope_local::Float32
    swa_every::Int                 # every Nth layer attends globally
    n_swa::Int                     # sliding window (sequence-length ceiling here)
end

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

rmsnorm(x::AbstractVector, w::AbstractVector, eps) =
    (x ./ sqrt(sum(abs2, x) / length(x) + eps)) .* w

function norm_cols(X::AbstractMatrix, w::AbstractVector, eps)
    out = similar(X)
    for j in axes(X, 2)
        out[:, j] = rmsnorm(view(X, :, j), w, eps)
    end
    return out
end

gelu_tanh(x) = 0.5f0 * x * (1 + tanh(0.7978845608028654f0 * (x + 0.044715f0 * x^3)))

# NEOX RoPE in place on head_dim × n_heads × T; absolute positions start at
# pos0 (0-based), so cached decoding rotates by the true sequence position.
function rope!(X::Array{Float32, 3}, base::Float32; pos0::Int = 0)
    hd, nh, T = size(X)
    half = hd ÷ 2
    for t in 1:T, h in 1:nh, i in 0:(half - 1)
        θ = (pos0 + t - 1) * base^(-2.0f0 * i / hd)
        c, s = cos(θ), sin(θ)
        x1, x2 = X[i + 1, h, t], X[i + 1 + half, h, t]
        X[i + 1, h, t] = x1 * c - x2 * s
        X[i + 1 + half, h, t] = x1 * s + x2 * c
    end
    return X
end

"""
    KVCache(m::Model; capacity = 4096)

Per-layer key/value cache. `n` counts the positions already decoded.
"""
mutable struct KVCache
    K::Vector{Array{Float32, 3}}   # head_dim × n_kv × capacity, per layer
    V::Vector{Array{Float32, 3}}
    n::Int
    capacity::Int
end

KVCache(m::Model; capacity::Int = 4096) = KVCache(
    [zeros(Float32, m.head_dim, m.n_kv, capacity) for _ in m.layers],
    [zeros(Float32, m.head_dim, m.n_kv, capacity) for _ in m.layers],
    0, capacity)

"""
    step!(m::Model, c::KVCache, toks::Vector{<:Integer}) -> Matrix{Float32}

Decode a chunk of new (0-based) token ids against the cache and append their
keys/values to it. Returns vocab × T logits for the new positions. Local
(sliding-window) layers attend only to the last `n_swa` positions — done by
slicing the cache, so long sequences stay exact.
"""
function step!(m::Model, c::KVCache, toks::Vector{<:Integer})
    T = length(toks)
    P = c.n
    P + T <= c.capacity ||
        error("KV cache full ($(c.capacity)); allocate a larger KVCache")
    hd, nh, nkv = m.head_dim, m.n_head, m.n_kv
    scale = 1.0f0 / sqrt(Float32(hd))
    h = m.embd[:, toks .+ 1] .* sqrt(Float32(size(m.embd, 1)))
    for (il, L) in enumerate(m.layers)
        local_layer = il % m.swa_every != 0
        base = local_layer ? m.rope_local : m.rope_global
        xn = norm_cols(h, L.attn_norm, m.eps)
        q = reshape(L.wq' * xn, hd, nh, T)
        k = reshape(L.wk' * xn, hd, nkv, T)
        v = reshape(L.wv' * xn, hd, nkv, T)
        for t in 1:T
            for hh in 1:nh
                q[:, hh, t] = rmsnorm(view(q, :, hh, t), L.q_norm, m.eps)
            end
            for hh in 1:nkv
                k[:, hh, t] = rmsnorm(view(k, :, hh, t), L.k_norm, m.eps)
            end
        end
        rope!(q, base; pos0 = P)
        rope!(k, base; pos0 = P)
        c.K[il][:, :, (P + 1):(P + T)] = k
        c.V[il][:, :, (P + 1):(P + T)] = v
        att = Matrix{Float32}(undef, hd * nh, T)
        for hh in 1:nh
            kv = 1 + (hh - 1) * nkv ÷ nh           # shared kv head (GQA/MQA)
            for tq in 1:T
                p = P + tq - 1                     # absolute 0-based position
                lo = local_layer ? max(1, p - m.n_swa + 2) : 1
                Kv = @view c.K[il][:, kv, lo:(p + 1)]
                Vv = @view c.V[il][:, kv, lo:(p + 1)]
                s = (Kv' * view(q, :, hh, tq)) .* scale
                s .= exp.(s .- maximum(s))
                s ./= sum(s)
                att[(1 + (hh - 1) * hd):(hh * hd), tq] = Vv * s
            end
        end
        h .+= norm_cols(L.wo' * att, L.post_attn_norm, m.eps)
        xn2 = norm_cols(h, L.ffn_norm, m.eps)
        act = gelu_tanh.(L.w_gate' * xn2) .* (L.w_up' * xn2)
        h .+= norm_cols(L.w_down' * act, L.post_ffn_norm, m.eps)
    end
    c.n += T
    return m.embd' * norm_cols(h, m.out_norm, m.eps)
end

"""
    forward(m::Model, toks::Vector{<:Integer}) -> Matrix{Float32}

Full forward over a sequence (fresh cache); vocab × T logits for every
position — the oracle-comparison entry point.
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
        next = argmax(view(logits, :, size(logits, 2))) - 1
        next in eog && break
        push!(out, next)
        logits = step!(m, c, [next])
    end
    return out
end

end # module Gemma3
