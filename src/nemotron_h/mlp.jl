struct NemotronHMLPLayer
    cfg::NemotronHConfig
    layer_idx::Int
    intermediate::Int
    up_w::Matrix{Float32}
    down_w::Matrix{Float32}
end

Functors.@functor NemotronHMLPLayer

function NemotronHMLPLayer(cfg::NemotronHConfig, layer_idx::Int)
    H = cfg.hidden_size
    ffn = mlp_intermediate_size(cfg, layer_idx)
    NemotronHMLPLayer(cfg, layer_idx, ffn, zeros(Float32, ffn, H), zeros(Float32, H, ffn))
end

function (m::NemotronHMLPLayer)(x::AbstractArray{Ty,3}) where {Ty}
    h = relu2(linear_last3(Float32.(x), m.up_w))
    Ty.(linear_last3(h, m.down_w))
end
