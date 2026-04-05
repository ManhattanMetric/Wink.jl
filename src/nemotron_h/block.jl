struct NemotronHBlock
    cfg::NemotronHConfig
    layer_idx::Int
    block_type::Symbol
    norm_w::Vector{Float32}
    mixer::Any
end

Functors.@functor NemotronHBlock

function NemotronHBlock(cfg::NemotronHConfig, layer_idx::Int)
    kinds = layers_block_type(cfg)
    typ = kinds[layer_idx+1]
    mixer =
        typ == :mamba ? NemotronHMambaLayer(cfg, layer_idx) :
        typ == :attention ? NemotronHAttentionLayer(cfg, layer_idx) :
        NemotronHMLPLayer(cfg, layer_idx)
    NemotronHBlock(cfg, layer_idx, typ, ones(Float32, cfg.hidden_size), mixer)
end

function (blk::NemotronHBlock)(hidden::AbstractArray{Ty,3}, mamba_mask, attn_mask) where {Ty}
    residual = hidden
    h = nemotron_rmsnorm(Float32.(hidden), blk.norm_w, blk.cfg.layer_norm_epsilon)
    h = if blk.block_type == :mamba
        blk.mixer(h, mamba_mask)
    elseif blk.block_type == :attention
        blk.mixer(h, attn_mask)
    else
        blk.mixer(h)
    end
    return Ty.(Float32.(residual) .+ Float32.(h))
end
