function embed_lookup(E::AbstractMatrix{T}, ids::AbstractMatrix{<:Integer}; one_based::Bool = false) where {T}
    B, TT = size(ids)
    H = size(E, 1)
    o = zeros(Float32, B, TT, H)
    @inbounds for b in 1:B, t in 1:TT
        j = one_based ? Int(ids[b, t]) : Int(ids[b, t]) + 1
        @assert 1 <= j <= size(E, 2)
        for d in 1:H
            o[b, t, d] = Float32(E[d, j])
        end
    end
    o
end

struct NemotronHModel
    cfg::NemotronHConfig
    tok_emb::Matrix{Float32}
    blocks::Vector{NemotronHBlock}
    norm_f_w::Vector{Float32}
end

Functors.@functor NemotronHModel

function NemotronHModel(cfg::NemotronHConfig)
    validate_pattern(cfg)
    H = cfg.hidden_size
    V = cfg.vocab_size
    NemotronHModel(
        cfg,
        zeros(Float32, H, V),
        [NemotronHBlock(cfg, i) for i in 0:(cfg.num_hidden_layers-1)],
        ones(Float32, H),
    )
end

function (model::NemotronHModel)(input_ids::AbstractMatrix{<:Integer}; attention_mask = nothing, one_based_tokens::Bool = false)
    B, T = size(input_ids)
    x = embed_lookup(model.tok_emb, input_ids; one_based = one_based_tokens)
    mamba_mask = attention_mask
    attn_mask = T > 1 ? repeat(causal_additive_mask(T, Float32); outer=(B, 1, 1, 1)) : nothing
    for blk in model.blocks
        mm = blk.block_type == :mamba ? mamba_mask : nothing
        am = blk.block_type == :attention ? attn_mask : nothing
        x = blk(x, mm, am)
    end
    x = nemotron_rmsnorm(x, model.norm_f_w, model.cfg.layer_norm_epsilon)
    return x
end

struct NemotronHCausalLM
    cfg::NemotronHConfig
    backbone::NemotronHModel
    lm_head_w::Matrix{Float32}
end

Functors.@functor NemotronHCausalLM

function NemotronHCausalLM(cfg::NemotronHConfig)
    NemotronHCausalLM(cfg, NemotronHModel(cfg), zeros(Float32, cfg.vocab_size, cfg.hidden_size))
end

function (lm::NemotronHCausalLM)(
    input_ids::AbstractMatrix{<:Integer};
    attention_mask = nothing,
    logits_to_keep = nothing,
    one_based_tokens::Bool = false,
)
    h = lm.backbone(input_ids; attention_mask, one_based_tokens)
    logits = linear_last3(h, lm.lm_head_w)
    k = logits_to_keep === nothing ? lm.cfg.num_logits_to_keep : logits_to_keep
    if k !== nothing && k < size(logits, 2)
        logits = logits[:, (end-k+1):end, :]
    end
    return logits
end
