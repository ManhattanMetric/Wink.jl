# OLMoE (mixture-of-experts) inference in generic-array Julia — the MoE
# machinery built unquantized before the quantized 26B work, on the model
# family already earmarked for the fine-tuning direction. Same discipline as
# gemma3.jl: correctness first, validated logit-for-logit against llama.cpp
# (test_olmoe.jl).
#
# Architecture (olmoe, from GGUF metadata): 16 layers, n_embd 2048, MHA
# 16/16 heads (head_dim 128), FULL-WIDTH QK-RMSNorms applied before the head
# split (unlike gemma's per-head norms), NEOX RoPE base 1e4, plain residuals
# (no sandwich norms), untied output head — and the point of the exercise:
# a 64-expert SwiGLU FFN with 8 experts routed per token, softmax router
# probabilities used as weights WITHOUT top-k renormalization.
#
# The expert dispatch is grouped: for each expert, the tokens routed to it
# are gathered and processed as one gemm, so prefill cost scales with
# activated-expert work, not with expert count.

module OLMoE

using LinearAlgebra
using KernelAbstractions
import Adapt

using ..GGUF: GGUFFile, tensor, metadata, raw_tensor
using ..Quant: Q4_0Stack, Q4_1Matrix

export load_model, forward, generate, KVCache, step!

# a 3-D dense expert tensor with the same per-expert indexing interface as
# Q4_0Stack, so moe_ffn! is agnostic to quantization
struct DenseStack{A <: AbstractArray{Float32, 3}}
    data::A
end
Base.getindex(s::DenseStack, e::Integer) = view(s.data, :, :, e)
Base.length(s::DenseStack) = size(s.data, 3)
Adapt.adapt_structure(to, s::DenseStack) = DenseStack(Adapt.adapt(to, s.data))

# abstract fields: quantized files mix storage types (q4_0 attn/experts,
# q8_0/q6_K embeddings), and expert stacks may be dense or quantized
struct Layer
    attn_norm::AbstractVector{Float32}
    wq::AbstractMatrix{Float32}
    wk::AbstractMatrix{Float32}
    wv::AbstractMatrix{Float32}
    wo::AbstractMatrix{Float32}
    q_norm::AbstractVector{Float32}
    k_norm::AbstractVector{Float32}
    ffn_norm::AbstractVector{Float32}
    w_router::AbstractMatrix{Float32}         # n_embd × n_expert
    gate_exps::Union{DenseStack, Q4_0Stack, Vector}  # [n_embd, n_ff] × n_expert
    up_exps::Union{DenseStack, Q4_0Stack, Vector}
    down_exps::Union{DenseStack, Q4_0Stack, Vector}
end
Adapt.@adapt_structure Layer

struct Model
    embd::AbstractMatrix{Float32}     # n_embd × vocab
    output::AbstractMatrix{Float32}   # n_embd × vocab (untied head)
    layers::Vector{Layer}
    out_norm::AbstractVector{Float32}
    n_head::Int
    n_kv::Int
    head_dim::Int
    n_expert_used::Int
    eps::Float32
    rope_base::Float32
end

Adapt.adapt_structure(to, m::Model) = Model(Adapt.adapt(to, m.embd),
    Adapt.adapt(to, m.output), [Adapt.adapt(to, L) for L in m.layers],
    Adapt.adapt(to, m.out_norm), m.n_head, m.n_kv, m.head_dim,
    m.n_expert_used, m.eps, m.rope_base)

# an expert stack: zero-copy q4_0 view of the 3-D tensor, or dense
function _experts(f::GGUFFile, name::AbstractString)
    typ = f.tensors[name].typ
    if typ == 2
        bytes, dims, _ = raw_tensor(f, name)
        return Q4_0Stack(bytes, dims[1], dims[2], dims[3])
    elseif typ == 3   # q4_1 slabs (llama.cpp bumps a few tensors in q4_0 files)
        bytes, dims, _ = raw_tensor(f, name)
        per = (dims[1] ÷ 32) * 20 * dims[2]
        return [Q4_1Matrix(view(bytes, ((e - 1) * per + 1):(e * per)),
                    dims[1], dims[2]) for e in 1:dims[3]]
    end
    return DenseStack(tensor(f, name))
end

function load_model(f::GGUFFile)
    arch = metadata(f, "general.architecture")
    arch == "olmoe" || error("this backend implements olmoe; file is $arch")
    nl = Int(metadata(f, "olmoe.block_count"))
    layers = map(0:(nl - 1)) do i
        p = "blk.$i."
        Layer(vec(tensor(f, p * "attn_norm.weight")),
            tensor(f, p * "attn_q.weight"), tensor(f, p * "attn_k.weight"),
            tensor(f, p * "attn_v.weight"), tensor(f, p * "attn_output.weight"),
            vec(tensor(f, p * "attn_q_norm.weight")),
            vec(tensor(f, p * "attn_k_norm.weight")),
            vec(tensor(f, p * "ffn_norm.weight")),
            tensor(f, p * "ffn_gate_inp.weight"),
            _experts(f, p * "ffn_gate_exps.weight"),
            _experts(f, p * "ffn_up_exps.weight"),
            _experts(f, p * "ffn_down_exps.weight"))
    end
    ne = Int(metadata(f, "olmoe.embedding_length"))
    nh = Int(metadata(f, "olmoe.attention.head_count"))
    return Model(tensor(f, "token_embd.weight"), tensor(f, "output.weight"),
        layers, vec(tensor(f, "output_norm.weight")),
        nh, Int(metadata(f, "olmoe.attention.head_count_kv")), ne ÷ nh,
        Int(metadata(f, "olmoe.expert_used_count")),
        Float32(metadata(f, "olmoe.attention.layer_norm_rms_epsilon")),
        Float32(metadata(f, "olmoe.rope.freq_base", 1.0f4)))
end

# ---- shared building blocks (mirrors gemma3.jl; unify on graduation) ----------

function norm1(X::AbstractArray, w::AbstractVector, eps)
    ms = sum(abs2, X; dims = 1) ./ Float32(size(X, 1))
    return (X .* w) ./ sqrt.(ms .+ eps)
end

silu(x) = x / (1.0f0 + exp(-x))

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

function _device_ints(ref::AbstractArray, xs::Vector{Int})
    out = KernelAbstractions.allocate(get_backend(ref), Int, length(xs))
    copyto!(out, xs)
    return out
end

# ---- the routed FFN -----------------------------------------------------------

"""
    moe_ffn!(out, L::Layer, xn, n_used) -> out

Router softmax over experts, top-`n_used` per token, grouped dispatch: each
activated expert processes its routed tokens as one gemm; outputs accumulate
weighted by the (unrenormalized) router probabilities.
"""
function moe_ffn!(out::AbstractMatrix, L::Layer, xn::AbstractMatrix, n_used::Int)
    T = size(xn, 2)
    r = L.w_router' * xn                       # n_expert × T
    p = exp.(r .- maximum(r; dims = 1))
    p ./= sum(p; dims = 1)
    pc = Array(p)                              # routing decisions on the CPU
    sel = [partialsortperm(view(pc, :, t), 1:n_used; rev = true) for t in 1:T]
    for e in 1:length(L.gate_exps)
        cols = [t for t in 1:T if e in sel[t]]
        isempty(cols) && continue
        Xs = xn[:, cols]
        G = L.gate_exps[e]' * Xs
        U = L.up_exps[e]' * Xs
        Y = L.down_exps[e]' * (silu.(G) .* U)
        w = reshape([pc[e, t] for t in cols], 1, :)
        view(out, :, cols) .+= Y .* w
    end
    return out
end

# ---- the forward pass ---------------------------------------------------------

function step!(m::Model, c::KVCache, toks::Vector{<:Integer})
    T = length(toks)
    P = c.n
    P + T <= c.capacity ||
        error("KV cache full ($(c.capacity)); allocate a larger KVCache")
    hd, nh, nkv = m.head_dim, m.n_head, m.n_kv
    scale = 1.0f0 / sqrt(Float32(hd))
    idx = _device_ints(m.embd, collect(Int, toks) .+ 1)
    h = m.embd[:, idx]                          # no embedding scale in olmoe
    kq = reshape(0:(P + T - 1), :, 1)
    qq = reshape(P:(P + T - 1), 1, :)
    for (il, L) in enumerate(m.layers)
        xn = norm1(h, L.attn_norm, m.eps)
        # FULL-WIDTH QK-norms, then the head split
        q = rope!(reshape(norm1(L.wq' * xn, L.q_norm, m.eps), hd, nh, T),
            m.rope_base; pos0 = P)
        k = rope!(reshape(norm1(L.wk' * xn, L.k_norm, m.eps), hd, nkv, T),
            m.rope_base; pos0 = P)
        c.K[il][:, :, (P + 1):(P + T)] = k
        c.V[il][:, :, (P + 1):(P + T)] = reshape(L.wv' * xn, hd, nkv, T)
        att = similar(h, hd * nh, T)
        for hh in 1:nh
            kv = 1 + (hh - 1) * nkv ÷ nh
            Kc = @view c.K[il][:, kv, 1:(P + T)]
            Vc = @view c.V[il][:, kv, 1:(P + T)]
            S = (Kc' * q[:, hh, :]) .* scale
            S = ifelse.(kq .<= qq, S, -Inf32)
            S = exp.(S .- maximum(S; dims = 1))
            S ./= sum(S; dims = 1)
            att[(1 + (hh - 1) * hd):(hh * hd), :] = Vc * S
        end
        h = h .+ L.wo' * att                    # plain residuals, no sandwich
        moe = fill!(similar(h), 0.0f0)
        moe_ffn!(moe, L, norm1(h, L.ffn_norm, m.eps), m.n_expert_used)
        h = h .+ moe
    end
    c.n += T
    return m.output' * norm1(h, m.out_norm, m.eps)
end

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

end # module OLMoE
