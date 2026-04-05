struct NemotronHAttentionLayer
    cfg::NemotronHConfig
    layer_idx::Int
    q_proj_w::Matrix{Float32}
    k_proj_w::Matrix{Float32}
    v_proj_w::Matrix{Float32}
    o_proj_w::Matrix{Float32}
end

Functors.@functor NemotronHAttentionLayer

function NemotronHAttentionLayer(cfg::NemotronHConfig, layer_idx::Int)
    H = cfg.hidden_size
    Dh = cfg.attention_head_dim
    nq = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    inner_q = nq * Dh
    inner_kv = nkv * Dh
    NemotronHAttentionLayer(
        cfg,
        layer_idx,
        zeros(Float32, inner_q, H),
        zeros(Float32, inner_kv, H),
        zeros(Float32, inner_kv, H),
        zeros(Float32, H, inner_q),
    )
end

function repeat_kv(x::AbstractArray{Float32,4}, n_rep::Int)
    B, nk, T, D = size(x)
    n_rep == 1 && return x
    y = zeros(Float32, B, nk * n_rep, T, D)
    for b in 1:B, t in 1:T, d in 1:D, h in 1:nk
        for r in 0:(n_rep-1)
            y[b, h + r * nk, t, d] = x[b, h, t, d]
        end
    end
    y
end

function (att::NemotronHAttentionLayer)(x::AbstractArray{Ty,3}, _mask = nothing) where {Ty}
    cfg = att.cfg
    B, seq, Hin = size(x)
    Dh = cfg.attention_head_dim
    nq = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    n_rep = nq ÷ nkv
    scale = Float32(1 / sqrt(Dh))

    x32 = Float32.(x)
    q = linear_last3(x32, att.q_proj_w)
    k = linear_last3(x32, att.k_proj_w)
    v = linear_last3(x32, att.v_proj_w)

    q = permutedims(reshape(q, B, seq, nq, Dh), (1, 3, 2, 4))
    k = permutedims(reshape(k, B, seq, nkv, Dh), (1, 3, 2, 4))
    v = permutedims(reshape(v, B, seq, nkv, Dh), (1, 3, 2, 4))

    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)

    attn_out = sdpa_causal(q, k, v, scale, attn_mask === nothing ? nothing : Float32.(attn_mask))
    attn_flat = permutedims(attn_out, (1, 3, 2, 4))
    attn_flat = reshape(attn_flat, B, seq, nq * Dh)
    y = linear_last3(attn_flat, att.o_proj_w)
    return Ty.(y)
end
