"""
    NemotronHConfig

Mirrors Hugging Face `NemotronHConfig` / `configuration_nemotron_h.py` fields needed for inference.
Pattern: `M` = Mamba2, `*` = attention, `-` = MLP.
"""
Base.@kwdef struct NemotronHConfig
    vocab_size::Int = 131_072
    tie_word_embeddings::Bool = false
    hidden_size::Int = 3136
    intermediate_size::Union{Int,Vector{Int}} = 12_544
    num_hidden_layers::Int = 42
    hybrid_override_pattern::String = "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-"
    num_attention_heads::Int = 40
    attention_head_dim::Int = 128
    num_key_value_heads::Int = 8
    mlp_hidden_act::Symbol = :relu2
    attention_bias::Bool = false
    mlp_bias::Bool = false
    use_bias::Bool = false
    layer_norm_epsilon::Float32 = 1.0f-5
    residual_in_fp32::Bool = false
    use_cache::Bool = true
    num_logits_to_keep::Union{Int,Nothing} = 1
    max_position_embeddings::Int = 262_144
    attention_dropout::Float32 = 0.0f0
    hidden_dropout::Float32 = 0.0f0
    ssm_state_size::Int = 128
    mamba_num_heads::Int = 96
    n_groups::Int = 8
    mamba_head_dim::Int = 80
    conv_kernel::Int = 4
    expand::Int = 2
    mamba_hidden_act::Symbol = :silu
    time_step_min::Float32 = 1.0f-3
    time_step_max::Float32 = 1.0f-1
    time_step_limit::Tuple{Float32,Float32} = (0.0f0, Float32(Inf))
    time_step_floor::Float32 = 1.0f-4
    use_conv_bias::Bool = true
    mamba_proj_bias::Bool = false
    chunk_size::Int = 256
    rescale_prenorm_residual::Bool = true
end

function validate_pattern(cfg::NemotronHConfig)
    p = cfg.hybrid_override_pattern
    length(p) == cfg.num_hidden_layers ||
        error("hybrid_override_pattern length $(length(p)) != num_hidden_layers $(cfg.num_hidden_layers)")
    all(c -> c in ('M', '*', '-'), collect(p)) || error("pattern must only contain M, *, -")
    return nothing
end

function layers_block_type(cfg::NemotronHConfig)
    validate_pattern(cfg)
    map(collect(cfg.hybrid_override_pattern)) do c
        c == 'M' ? :mamba : c == '*' ? :attention : :mlp
    end
end

function mamba_intermediate_size(cfg::NemotronHConfig)
    cfg.mamba_num_heads * cfg.mamba_head_dim
end

function mamba_conv_dim(cfg::NemotronHConfig)
    mamba_intermediate_size(cfg) + 2 * cfg.n_groups * cfg.ssm_state_size
end

function mamba_in_proj_out(cfg::NemotronHConfig)
    d_inner = mamba_intermediate_size(cfg)
    d_conv = mamba_conv_dim(cfg)
    2 * d_inner + 2 * cfg.n_groups * cfg.ssm_state_size + cfg.mamba_num_heads
end

function mlp_index(cfg::NemotronHConfig, layer_idx::Int)
    count(c -> c == '-', cfg.hybrid_override_pattern[1:layer_idx+1]) - 1
end

function mlp_intermediate_size(cfg::NemotronHConfig, layer_idx::Int)
    if cfg.intermediate_size isa Int
        return cfg.intermediate_size
    end
    idx = mlp_index(cfg, layer_idx) + 1
    cfg.intermediate_size[idx]
end

function attention_inner_dim(cfg::NemotronHConfig)
    cfg.num_attention_heads * cfg.attention_head_dim
end

"""Default numeric spec for `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`."""
function nemotron_3_nano_4b_bf16()
    NemotronHConfig()
end
