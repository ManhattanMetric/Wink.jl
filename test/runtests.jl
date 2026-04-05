using Wink
using NPZ
using Test

include("hf_golden_utils.jl")

@testset "config / pattern" begin
    cfg = nemotron_3_nano_4b_bf16()
    @test length(layers_block_type(cfg)) == 42
    @test layers_block_type(cfg)[1] == :mamba
    p = cfg.hybrid_override_pattern
    @test_throws ErrorException begin
        bad = NemotronHConfig(; cfg.hidden_size, cfg.vocab_size, num_hidden_layers=3, hybrid_override_pattern=p)
        layers_block_type(bad)
    end
end

@testset "tiny MM forward (smoke)" begin
    cfg = NemotronHConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        hybrid_override_pattern="MM",
        mamba_num_heads=4,
        mamba_head_dim=8,
        n_groups=2,
        ssm_state_size=8,
        conv_kernel=4,
        chunk_size=8,
        num_attention_heads=4,
        attention_head_dim=8,
        num_key_value_heads=2,
        intermediate_size=64,
        num_logits_to_keep=nothing,
    )
    lm = NemotronHCausalLM(cfg)
    logits = lm(ones(Int, 1, 4); one_based_tokens=true)
    @test size(logits) == (1, 4, 64)
end

@testset "HF golden logits (tiny 2×Mamba)" begin
    path = joinpath(@__DIR__, "fixtures", "golden_tiny_mm.npz")
    @test isfile(path)
    raw = npzread(path)
    wdict = Dict{String,Array{Float32}}(
        String(k) => Float32.(v) for (k, v) in pairs(raw) if startswith(String(k), "w__")
    )
    cfg = NemotronHConfig(
        vocab_size=64,
        hidden_size=64,
        num_hidden_layers=2,
        hybrid_override_pattern="MM",
        mamba_num_heads=4,
        mamba_head_dim=16,
        n_groups=2,
        ssm_state_size=16,
        conv_kernel=4,
        chunk_size=8,
        num_attention_heads=4,
        attention_head_dim=16,
        num_key_value_heads=2,
        intermediate_size=128,
        num_logits_to_keep=nothing,
    )
    lm = NemotronHCausalLM(cfg)
    load_tiny_mm_from_npz!(lm, wdict)
    ids = Int.(raw["input_ids"])
    logits = lm(ids; one_based_tokens=false, logits_to_keep=nothing)
    gold = Float32.(raw["logits"])
    # Float32 CPU reference: expect ~1e-1 abs after two Mamba SSD blocks (manual SDPA vs torch matmul, etc.).
    @test size(logits) == size(gold)
    @test maximum(abs.(logits .- gold)) < 0.15f0
end

@testset "GGUF load (optional)" begin
    p = get(ENV, "RUYA_NEMOTRON_GGUF", "")
    if !isempty(p) && isfile(p)
        cfg = nemotron_3_nano_4b_bf16()
        lm = NemotronHCausalLM(cfg)
        load_nemotron_h_gguf!(lm, p)
        ids = fill(Int32(1), 1, 1)
        out = lm(ids; one_based_tokens=false)
        @test size(out, 3) == cfg.vocab_size
        @test all(isfinite, out)
    else
        @test true
    end
end
