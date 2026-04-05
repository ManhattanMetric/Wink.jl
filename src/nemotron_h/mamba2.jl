struct NemotronHMambaLayer
    cfg::NemotronHConfig
    layer_idx::Int
    in_proj_w::Matrix{Float32}
    conv1d_w::Matrix{Float32}
    conv1d_b::Vector{Float32}
    dt_bias::Vector{Float32}
    A_log::Vector{Float32}
    norm_w::Vector{Float32}
    D::Vector{Float32}
    out_proj_w::Matrix{Float32}
end

Functors.@functor NemotronHMambaLayer

function NemotronHMambaLayer(cfg::NemotronHConfig, layer_idx::Int)
    H = cfg.hidden_size
    d_inner = mamba_intermediate_size(cfg)
    d_conv = mamba_conv_dim(cfg)
    P = mamba_in_proj_out(cfg)
    nheads = cfg.mamba_num_heads
    K = cfg.conv_kernel
    NemotronHMambaLayer(
        cfg,
        layer_idx,
        zeros(Float32, P, H),
        zeros(Float32, d_conv, K),
        zeros(Float32, d_conv),
        zeros(Float32, nheads),
        zeros(Float32, nheads),
        zeros(Float32, d_inner),
        zeros(Float32, nheads),
        zeros(Float32, H, d_inner),
    )
end

function causal_conv1d_nemotron(
    x::AbstractArray{Float32,3},
    w::AbstractMatrix{Float32},
    b::AbstractVector{Float32},
    act,
)
    B, T, C = size(x)
    K = size(w, 2)
    out = similar(x)
    @inbounds for bi in 1:B, t in 1:T, c in 1:C
        acc = zero(Float32)
        for kk in 1:K
            tt = t - K + kk
            if tt >= 1
                acc += w[c, kk] * x[bi, tt, c]
            end
        end
        acc += b[c]
        out[bi, t, c] = act(acc)
    end
    out
end

function linear_last3(x::AbstractArray{Float32,3}, W::AbstractMatrix{Float32})
    B, T, fin = size(x)
    fout = size(W, 1)
    y = zeros(Float32, B, T, fout)
    @inbounds for b in 1:B, t in 1:T, o in 1:fout
        s = zero(Float32)
        for i in 1:fin
            s += W[o, i] * x[b, t, i]
        end
        y[b, t, o] = s
    end
    y
end

function (m::NemotronHMambaLayer)(hidden_states::AbstractArray{Ty,3}, attention_mask = nothing) where {Ty}
    cfg = m.cfg
    x = Float32.(apply_mask_to_padding_states(Float32.(hidden_states), attention_mask))
    B, seq_len, Hin = size(x)
    @assert Hin == cfg.hidden_size

    d_inner = mamba_intermediate_size(cfg)
    n_groups = cfg.n_groups
    d_state = cfg.ssm_state_size
    nheads = cfg.mamba_num_heads
    head_dim = cfg.mamba_head_dim
    conv_dim = mamba_conv_dim(cfg)
    chunk_sz = cfg.chunk_size
    projected = linear_last3(x, m.in_proj_w)
    P = size(projected, 3)
    d_mlp = (P - 2 * d_inner - 2 * n_groups * d_state - nheads) ÷ 2
    @assert 2 * d_mlp + 2 * d_inner + 2 * n_groups * d_state + nheads == P

    o = 2 * d_mlp
    gate = projected[:, :, o+1:o+d_inner]
    o += d_inner
    hidden_states_B_C = projected[:, :, o+1:o+conv_dim]
    o += conv_dim
    dt = projected[:, :, o+1:o+nheads]

    hsbc = causal_conv1d_nemotron(hidden_states_B_C, m.conv1d_w, m.conv1d_b, silu)
    hsbc = apply_mask_to_padding_states(hsbc, attention_mask)

    h_x = hsbc[:, :, 1:d_inner]
    Bpart = hsbc[:, :, d_inner+1:d_inner+n_groups*d_state]
    Cpart = hsbc[:, :, d_inner+n_groups*d_state+1:end]

    dt = softplus.(dt .+ reshape(m.dt_bias, 1, 1, :))
    # Match HF `torch_forward` naive path: clamp only a lower bound (no upper).
    dt = max.(dt, Float32(cfg.time_step_min))

    h_r = reshape(h_x, B, seq_len, nheads, head_dim)
    B_r = Float32.(reshape(Bpart, B, seq_len, n_groups, d_state))
    C_r = Float32.(reshape(Cpart, B, seq_len, n_groups, d_state))
    B_r = repeat(B_r; inner=(1, 1, nheads ÷ n_groups, 1))
    C_r = repeat(C_r; inner=(1, 1, nheads ÷ n_groups, 1))
    B_r = reshape(B_r, B, seq_len, nheads, d_state)
    C_r = reshape(C_r, B, seq_len, nheads, d_state)

    A = .-exp.(m.A_log)
    pad_size = (chunk_sz - seq_len % chunk_sz) % chunk_sz

    D_res = reshape(m.D, 1, 1, nheads, 1) .* pad_tensor_by_size(h_r, pad_size)

    h_dt = h_r .* reshape(dt, B, seq_len, nheads, 1)
    A_dt = reshape(A, 1, 1, nheads) .* dt
    A_dt = reshape(A_dt, B, seq_len, nheads)

    h_ch = reshape_into_chunks(h_dt, pad_size, chunk_sz)
    A_ch = reshape_into_chunks(A_dt, pad_size, chunk_sz)
    B_ch = reshape_into_chunks(B_r, pad_size, chunk_sz)
    C_ch = reshape_into_chunks(C_r, pad_size, chunk_sz)

    n_chunk = size(h_ch, 2)
    Lsz = chunk_sz
    A_perm = permutedims(A_ch, (1, 4, 2, 3))
    A_cumsum = cumsum(A_perm; dims=4)

    Lm = exp.(segment_sum(A_perm))
    # `Lm`: (B, H, C, L, L); align with `G` (B, C, L, L, H) like HF `L.permute(0, 2, 3, 4, 1)`.
    L_perm = permutedims(Lm, (1, 3, 4, 5, 2))

    G = zeros(Float32, B, n_chunk, Lsz, Lsz, nheads)
    @inbounds for bi in 1:B, c in 1:n_chunk, i in 1:Lsz, j in 1:Lsz, h in 1:nheads
        acc = zero(Float32)
        for n in 1:d_state
            acc += C_ch[bi, c, i, h, n] * B_ch[bi, c, j, h, n]
        end
        G[bi, c, i, j, h] = acc
    end

    M = G .* L_perm

    Y_diag = zeros(Float32, B, n_chunk, Lsz, nheads, head_dim)
    @inbounds for bi in 1:B, c in 1:n_chunk, i in 1:Lsz, h in 1:nheads, d in 1:head_dim
        acc = zero(Float32)
        for j in 1:Lsz
            acc += M[bi, c, i, j, h] * h_ch[bi, c, j, h, d]
        end
        Y_diag[bi, c, i, h, d] = acc
    end

    decay_st = exp.(A_cumsum[:, :, :, Lsz:Lsz] .- A_cumsum)
    decay_perm = permutedims(decay_st, (1, 4, 3, 2))
    B_decay = B_ch .* reshape(decay_perm, B, n_chunk, Lsz, nheads, 1)

    states_intra = zeros(Float32, B, n_chunk, nheads, head_dim, d_state)
    @inbounds for bi in 1:B, c in 1:n_chunk, h in 1:nheads, d in 1:head_dim, n in 1:d_state
        acc = zero(Float32)
        for l in 1:Lsz
            acc += B_decay[bi, c, l, h, n] * h_ch[bi, c, l, h, d]
        end
        states_intra[bi, c, h, d, n] = acc
    end

    prev = zeros(Float32, B, 1, nheads, head_dim, d_state)
    states_cat = cat(prev, states_intra; dims=2)

    A_end = A_cumsum[:, :, :, end]
    zpad = zeros(Float32, size(A_end, 1), size(A_end, 2), 1)
    A_pad = cat(zpad, A_end; dims=3)
    decay_chunk = exp.(segment_sum(A_pad))
    decay_chunk = permutedims(decay_chunk, (1, 3, 4, 2))

    nc1 = size(states_cat, 2)
    @assert size(decay_chunk, 2) == nc1 && size(decay_chunk, 3) == nc1
    new_states = zeros(Float32, B, nc1, nheads, head_dim, d_state)
    @inbounds for bi in 1:B, j in 1:nc1, h in 1:nheads, d in 1:head_dim, n in 1:d_state
        s = zero(Float32)
        for i in 1:nc1
            s += decay_chunk[bi, i, j, h]
        end
        new_states[bi, j, h, d, n] = states_cat[bi, j, h, d, n] * s
    end

    states_final = new_states[:, 1:end-1, :, :, :, :]

    Ctimes = zeros(Float32, B, n_chunk, Lsz, nheads, head_dim, d_state)
    @inbounds for bi in 1:B, c in 1:n_chunk, l in 1:Lsz, h in 1:nheads, d in 1:head_dim, n in 1:d_state
        Ctimes[bi, c, l, h, d, n] = C_ch[bi, c, l, h, n] * states_final[bi, c, h, d, n]
    end
    Csum = dropdims(sum(Ctimes; dims=6); dims=6)

    sdo = exp.(A_cumsum)
    sdo_p = permutedims(sdo, (1, 3, 4, 2))
    Y_off = Csum .* reshape(sdo_p, B, n_chunk, Lsz, nheads, 1)

    y = Y_diag .+ Y_off
    y = reshape(y, B, n_chunk * Lsz, nheads, head_dim)
    y = y .+ D_res
    if pad_size > 0
        y = y[:, 1:seq_len, :, :]
    end
    y = reshape(y, B, seq_len, d_inner)

    scan_output = mamba_rmsnorm_gated(y, m.norm_w, gate, cfg.layer_norm_epsilon, cfg.n_groups)
    return Ty.(linear_last3(Float32.(scan_output), m.out_proj_w))
end
