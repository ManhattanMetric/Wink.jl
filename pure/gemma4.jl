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
using ..Quant: Q4_0Matrix, Q4_0Stack, q4_dot, perexp

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
    gate_up_exps::Q4_0Stack           # [n_embd, 2n_ff] × n_expert, one buffer
    down_exps::Q4_0Stack              # [n_ff, n_embd] × n_expert, one buffer
    down_exps_s::AbstractVector{Float32}
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

# an expert stack as a zero-copy view of the whole 3-D q4_0 tensor
function _expert_stack(f::GGUFFile, name::AbstractString)
    bytes, dims, typ = raw_tensor(f, name)
    typ == 2 || error("$name: expected q4_0 experts, got ggml type $typ")
    return Q4_0Stack(bytes, dims[1], dims[2], dims[3])
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
            _expert_stack(f, p * "ffn_gate_up_exps.weight"),
            _expert_stack(f, p * "ffn_down_exps.weight"),
            haskey(f.tensors, p * "ffn_down_exps.scale") ?
                vec(tensor(f, p * "ffn_down_exps.scale")) :
                ones(Float32, Int(metadata(f, "gemma4.expert_count"))))
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

# In-place building blocks: on Metal, every allocation is a ~1.4ms
# pipeline stall (fresh device buffers until GC), so the forward pass runs
# entirely in named arena buffers (see buf!) and these helpers write into
# their destination. Y === X is safe: the reduction runs first and the
# apply step is element-aligned.

function norm1!(Y, ms, X, w, eps)
    fill!(ms, 0.0f0)
    Base.mapreducedim!(abs2, +, ms, X)
    Y .= (X .* w) ./ sqrt.(ms ./ Float32(size(X, 1)) .+ eps)
    return Y
end

# weightless RMS-norm (gemma-4 uses it on V and on the router input)
function norm0!(Y, ms, X, eps)
    fill!(ms, 0.0f0)
    Base.mapreducedim!(abs2, +, ms, X)
    Y .= X ./ sqrt.(ms ./ Float32(size(X, 1)) .+ eps)
    return Y
end

function softmax!(S, mx, ssum)
    fill!(mx, -Inf32)
    Base.mapreducedim!(identity, max, mx, S)
    S .= exp.(S .- mx)
    fill!(ssum, 0.0f0)
    Base.mapreducedim!(identity, +, ssum, S)
    S ./= ssum
    return S
end

gelu_tanh(x) = 0.5f0 * x * (1 + tanh(0.7978845608028654f0 * (x + 0.044715f0 * x^3)))

@kernel function _rope_kernel!(X, base::Float32, pos0::Int32, half::Int32,
        factors, hasfac::Int32)
    i, h, t = @index(Global, NTuple)
    fac = hasfac != 0 ? factors[i] : 1.0f0
    θ = Float32(pos0 + t - 1) * base^(-2.0f0 * (i - 1) / (2.0f0 * half)) / fac
    c, s = cos(θ), sin(θ)
    x1 = X[i, h, t]
    x2 = X[i + half, h, t]
    X[i, h, t] = x1 * c - x2 * s
    X[i + half, h, t] = x1 * s + x2 * c
end

function rope!(X::AbstractArray{Float32, 3}, base::Float32, factors;
        pos0::Int = 0)
    half = size(X, 1) ÷ 2
    _rope_kernel!(get_backend(X))(X, base, Int32(pos0), Int32(half), factors,
        Int32(length(factors) == half); ndrange = (half, size(X, 2), size(X, 3)))
    return X
end

# Attention as two fused kernels (one launch each, any batch size) instead
# of a per-head loop of small matmuls — at 48 layers the launch count is
# what throttles the GPU, not the math. The causal + sliding-window mask is
# decided in-kernel (nswa <= 0 means global).
@kernel function _attn_scores!(S, @Const(K), @Const(q), hpk::Int32, P::Int32,
        nswa::Int32)
    pos, hh, t = @index(Global, NTuple)
    qpos = Int(P) + t
    if pos <= qpos && (Int(nswa) <= 0 || qpos - pos < Int(nswa))
        kv = (hh - 1) ÷ Int(hpk) + 1
        s = 0.0f0
        @inbounds for d in 1:size(K, 1)
            s = muladd(K[d, kv, pos], q[d, hh, t], s)
        end
        @inbounds S[pos, hh, t] = s
    else
        @inbounds S[pos, hh, t] = -Inf32
    end
end

# in-place column softmax over dim 1 — one launch, view-friendly (the
# mapreducedim! path collapses on strided views of the padded S buffer)
@kernel function _softmax_dim1!(S)
    hh, t = @index(Global, NTuple)
    n = size(S, 1)
    mx = -Inf32
    @inbounds for i in 1:n
        mx = max(mx, S[i, hh, t])
    end
    ssum = 0.0f0
    @inbounds for i in 1:n
        e = exp(S[i, hh, t] - mx)
        S[i, hh, t] = e
        ssum += e
    end
    inv = 1.0f0 / ssum
    @inbounds for i in 1:n
        S[i, hh, t] *= inv
    end
end

@kernel function _attn_out!(att, @Const(V), @Const(S), hpk::Int32, hd::Int32)
    d, hh, t = @index(Global, NTuple)
    kv = (hh - 1) ÷ Int(hpk) + 1
    s = 0.0f0
    @inbounds for pos in 1:size(S, 1)
        s = muladd(V[d, kv, pos], S[pos, hh, t], s)
    end
    @inbounds att[(hh - 1) * Int(hd) + d, t] = s
end

# Each token's top-K experts, selected and weighted ON the device — the
# host round-trip per layer was what kept Metal's command batching from
# ever forming a batch. Serial repeated-max over n_expert probs per token;
# the down-projection's QAT scale folds into the weight here.
@kernel function _router_topk!(sel, w, @Const(p), @Const(down_s), K::Int32,
        clampmin::Float32)
    t = @index(Global)
    wsum = 0.0f0
    @inbounds for k in 1:Int(K)
        best = -Inf32
        bi = 1
        for e in 1:size(p, 1)
            v = p[e, t]
            taken = false
            for kk in 1:(k - 1)
                taken |= Int(sel[kk, t]) == e
            end
            if !taken && v > best
                best = v
                bi = e
            end
        end
        sel[k, t] = Int32(bi)
        w[k, t] = best
        wsum += best
    end
    wsum = max(wsum, clampmin)
    @inbounds for k in 1:Int(K)
        w[k, t] = w[k, t] / wsum * down_s[Int(sel[k, t])]
    end
end

# small dense f32 matvec as a kernel: the router matmul must not route to
# MPS, whose separate command buffer would flush the batch every layer
@kernel function _dense_matvec!(Y, @Const(W), @Const(X))
    j, t = @index(Global, NTuple)
    s = 0.0f0
    @inbounds for i in 1:size(W, 1)
        s = muladd(W[i, j], X[i, t], s)
    end
    @inbounds Y[j, t] = s
end

# The fused MoE: kernel A computes gated activations for every
# (expert, token) pair, kernel B has each (channel, token) thread sum its
# own n_used experts — 2 launches per layer at ANY batch size, where the
# per-expert loop needed ~6 launches (plus an upload) per ACTIVE expert.
# Expert weights are addressed by byte stride inside the layer's single
# stack buffer; the top-8 routing weights (with the down-projection's QAT
# scale folded in) arrive as one small upload.
@kernel function _moe_act!(act, @Const(data), @Const(sel), @Const(X),
        nb::Int32, nf::Int32, K::Int32, per::Int)
    i, kt = @index(Global, NTuple)
    k = (kt - 1) % Int(K) + 1
    t = (kt - 1) ÷ Int(K) + 1
    @inbounds e = Int(sel[k, t])
    base = (e - 1) * per
    g = q4_dot(data, base + (i - 1) * Int(nb) * 18, X, t, Int(nb))
    u = q4_dot(data, base + (i - 1 + Int(nf)) * Int(nb) * 18, X, t, Int(nb))
    @inbounds act[i, kt] = gelu_tanh(g) * u
end

@kernel function _moe_down!(moe, @Const(data), @Const(sel), @Const(w),
        @Const(act), nb::Int32, K::Int32, per::Int)
    d, t = @index(Global, NTuple)
    s = 0.0f0
    @inbounds for k in 1:Int(K)
        e = Int(sel[k, t])
        s = muladd(Float32(w[k, t]),
            q4_dot(data, (e - 1) * per + (d - 1) * Int(nb) * 18,
                act, (t - 1) * Int(K) + k, Int(nb)), s)
    end
    @inbounds moe[d, t] = s
end

mutable struct KVCache{A <: AbstractArray{Float32, 3}}
    K::Vector{A}
    V::Vector{A}
    n::Int
    capacity::Int
    scratch::Dict{Any, Any}   # named reusable buffers — see buf!
end

KVCache(m::Model; capacity::Int = 4096) = KVCache(
    [fill!(similar(L.attn_norm, L.hd, L.nkv, capacity), 0.0f0) for L in m.layers],
    [fill!(similar(L.attn_norm, L.hd, L.nkv, capacity), 0.0f0) for L in m.layers],
    0, capacity, Dict{Any, Any}())

# the arena: step! intermediates keyed by use-site name + dims, allocated
# once and reused every token (per-layer geometry differences key apart
# naturally; dynamic lookup costs ~100ns against ~ms device allocations)
@inline buf!(c::KVCache, ref, name::Symbol, dims::Int...) =
    get!(() -> similar(ref, Float32, dims), c.scratch, (name, dims...))
@inline tbuf!(c::KVCache, ref, ::Type{T}, name::Symbol, dims::Int...) where {T} =
    get!(() -> similar(ref, T, dims), c.scratch, (name, dims...))

# ---- the forward pass ---------------------------------------------------------

function step!(m::Model, c::KVCache, toks::Vector{<:Integer})
    T = length(toks)
    P = c.n
    P + T <= c.capacity ||
        error("KV cache full ($(c.capacity)); allocate a larger KVCache")
    inv_sqrt_embd = 1.0f0 / sqrt(Float32(m.n_embd))
    ne = m.n_embd
    h = buf!(c, c.K[1], :h, ne, T)
    copyto!(h, m.embd[:, collect(Int, toks) .+ 1] .* sqrt(Float32(ne)))
    ongpu = !(get_backend(h) isa KernelAbstractions.CPU)
    kq = reshape(0:(P + T - 1), :, 1)
    qq = reshape(P:(P + T - 1), 1, :)
    msn = buf!(c, h, :msn, 1, T)          # shared column-norm scratch
    for (il, L) in enumerate(m.layers)
        base = L.is_swa ? m.rope_local : m.rope_global
        hd, nh, nkv = L.hd, L.nh, L.nkv
        facs = L.is_swa ? view(m.rope_factors, 1:0) : m.rope_factors
        xn = norm1!(buf!(c, h, :xn, ne, T), msn, h, L.attn_norm, m.eps)
        q2 = mul!(buf!(c, h, :q, hd * nh, T), L.wq', xn)
        q2 .*= L.wq_s
        q = reshape(q2, hd, nh, T)
        msq = buf!(c, h, :msq, 1, nh, T)
        norm1!(q, msq, q, L.q_norm, m.eps)
        rope!(q, base, facs; pos0 = P)
        k2 = mul!(buf!(c, h, :k, hd * nkv, T), L.wk', xn)
        k2 .*= L.wk_s
        k3 = reshape(k2, hd, nkv, T)
        msk = buf!(c, h, :msk, 1, nkv, T)
        norm1!(k3, msk, k3, L.k_norm, m.eps)
        rope!(k3, base, facs; pos0 = P)
        v2 = mul!(buf!(c, h, :v, hd * nkv, T),
            (L.wv === nothing ? L.wk : L.wv)', xn)
        v2 .*= L.wv === nothing ? L.wk_s : L.wv_s
        v3 = reshape(v2, hd, nkv, T)
        norm0!(v3, msk, v3, m.eps)
        c.K[il][:, :, (P + 1):(P + T)] = k3
        c.V[il][:, :, (P + 1):(P + T)] = v3
        att = buf!(c, h, :att, hd * nh, T)
        if T == 1 || ongpu
            Kp = P + T
            # generation: padded buffer + view so the key is stable across
            # tokens; prefill: exact contiguous buffer (new dims only once
            # per chunk — reductions and kernels prefer contiguous)
            S = if T == 1
                Kpad = min(cld(Kp, 512) * 512, c.capacity)
                view(buf!(c, h, :S, Kpad, nh, T), 1:Kp, :, :)
            else
                buf!(c, h, :S, Kp, nh, T)
            end
            _attn_scores!(get_backend(h))(S, c.K[il], q, Int32(nh ÷ nkv),
                Int32(P), Int32(L.is_swa ? m.n_swa : 0); ndrange = (Kp, nh, T))
            _softmax_dim1!(get_backend(h))(S; ndrange = (nh, T))
            _attn_out!(get_backend(h))(att, c.V[il], S, Int32(nh ÷ nkv),
                Int32(hd); ndrange = (hd, nh, T))
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
        attno = mul!(buf!(c, h, :attno, ne, T), L.wo', att)
        attno .*= L.wo_s
        attn_out = norm1!(buf!(c, h, :attn_out, ne, T), msn, attno,
            L.attn_post_norm, m.eps)
        attn_out .+= h
        # dense (shared-expert) branch
        mlpn = norm1!(buf!(c, h, :mlpn, ne, T), msn, attn_out, L.ffn_norm, m.eps)
        nfd = size(L.ffn_gate, 2)
        g = mul!(buf!(c, h, :ffg, nfd, T), L.ffn_gate', mlpn)
        u = mul!(buf!(c, h, :ffu, nfd, T), L.ffn_up', mlpn)
        g .= gelu_tanh.(g .* L.ffn_gate_s) .* (u .* L.ffn_up_s)
        mlp = mul!(buf!(c, h, :mlp, ne, T), L.ffn_down', g)
        mlp .*= L.ffn_down_s
        norm1!(mlp, msn, mlp, L.ffn_post_norm_1, m.eps)
        # router (its own view of the residual stream)
        tmp = norm0!(buf!(c, h, :rt, ne, T), msn, attn_out, m.eps)
        tmp .= tmp .* inv_sqrt_embd .* L.gate_inp_s
        nex = size(L.gate_inp, 2)
        p = buf!(c, h, :p, nex, T)
        if ongpu && T == 1
            # T=1: MPS would flush the command batch every layer; a serial
            # matvec kernel batches. Prefill amortizes the flush, and MPS
            # GEMM wins decisively at large T.
            _dense_matvec!(get_backend(h))(p, L.gate_inp, tmp;
                ndrange = (nex, T))
        else
            mul!(p, L.gate_inp', tmp)
        end
        softmax!(p, buf!(c, h, :mxp, 1, T), buf!(c, h, :smp, 1, T))
        # expert branch
        moe_in = norm1!(buf!(c, h, :moein, ne, T), msn, attn_out,
            L.ffn_pre_norm_2, m.eps)
        moe = buf!(c, h, :moe, ne, T)
        nf = L.down_exps.nrow
        K = m.n_expert_used
        if ongpu
            gu, dn = L.gate_up_exps, L.down_exps
            seld = tbuf!(c, h, Int32, :seld, K, T)
            wd = buf!(c, h, :wd, K, T)
            _router_topk!(get_backend(h))(seld, wd, p, L.down_exps_s,
                Int32(K), 6.103515625f-5; ndrange = T)
            act = buf!(c, h, :moeact, nf, K * T)
            _moe_act!(get_backend(h))(act, gu.data, seld, moe_in,
                Int32(gu.nrow ÷ 32), Int32(nf), Int32(K), perexp(gu);
                ndrange = (nf, K * T))
            _moe_down!(get_backend(h))(moe, dn.data, seld, wd, act,
                Int32(dn.nrow ÷ 32), Int32(K), perexp(dn);
                ndrange = (ne, T))
        else
            fill!(moe, 0.0f0)
            pc = Array(p)
            sel = [partialsortperm(view(pc, :, t), 1:K; rev = true)
                   for t in 1:T]
            wsum = [max(sum(pc[e, t] for e in sel[t]), 6.103515625f-5)
                    for t in 1:T]
            if T == 1
                for e in sel[1]                        # scalar weights, no
                    GU = mul!(buf!(c, h, :GU, 2nf, 1), # gather/scatter
                        L.gate_up_exps[e]', moe_in)
                    ga = view(GU, 1:nf, :)
                    ga .= gelu_tanh.(ga) .* view(GU, (nf + 1):(2nf), :)
                    Ye = mul!(buf!(c, h, :expY, ne, 1), L.down_exps[e]', ga)
                    moe .+= Ye .* (L.down_exps_s[e] * pc[e, 1] / wsum[1])
                end
            else
                for e in 1:length(L.gate_up_exps)
                    cols = [t for t in 1:T if e in sel[t]]
                    isempty(cols) && continue
                    GU = L.gate_up_exps[e]' * moe_in[:, cols]
                    ga = gelu_tanh.(view(GU, 1:nf, :)) .*
                         view(GU, (nf + 1):(2nf), :)
                    Y = (L.down_exps[e]' * ga) .* L.down_exps_s[e]
                    w = copyto!(similar(Y, 1, length(cols)),
                        Float32[pc[e, t] / wsum[t] for t in cols])
                    moe[:, cols] = moe[:, cols] .+ Y .* w
                end
            end
        end
        norm1!(moe, msn, moe, L.ffn_post_norm_2, m.eps)
        mlp .+= moe
        norm1!(mlp, msn, mlp, L.ffn_post_norm, m.eps)
        h .= (mlp .+ attn_out) .* L.out_scale
    end
    c.n += T
    xnf = norm1!(buf!(c, h, :xnf, ne, T), msn, h, m.out_norm, m.eps)
    logits = mul!(buf!(c, h, :logits, size(m.output, 2), T), m.output', xnf)
    m.softcap > 0 &&
        (logits .= m.softcap .* tanh.(logits ./ m.softcap))
    return logits    # valid until the next step! on this cache
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
    Adapt.adapt(to, L.gate_up_exps), Adapt.adapt(to, L.down_exps),
    Adapt.adapt(to, L.down_exps_s))

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
