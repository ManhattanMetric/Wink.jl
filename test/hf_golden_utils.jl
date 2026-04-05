# Helpers for loading `scripts/export_nemotron_golden.py` output into `NemotronHCausalLM`.

function _w(d::Dict{String,Array{Float32}}, parts::AbstractString...)
    k = "w__" * join(parts, "__")
    haskey(d, k) || error("missing npz key: $k")
    return d[k]
end

"""HF `nn.Linear` / `nn.Embedding` weight matrices are `(out, in)` like our `linear_last3`."""
function load_tiny_mm_from_npz!(lm::NemotronHCausalLM, d::Dict{String,Array{Float32}})
    emb = _w(d, "model", "embeddings", "weight")
    V, H = size(emb)
    @assert size(lm.backbone.tok_emb) == (H, V)
    # HF `Embedding`: row `p` (0-based vocab id `p`). With `embed_lookup(...; one_based=false)`,
    # token id `p` uses column `p+1`: `tok_emb[:, p+1] == emb[p+1, :]`.
    @inbounds for p in 0:(V-1), h in 1:H
        lm.backbone.tok_emb[h, p+1] = emb[p+1, h]
    end

    nf = vec(_w(d, "model", "norm_f", "weight"))
    lm.backbone.norm_f_w .= nf

    lh = _w(d, "lm_head", "weight")
    lm.lm_head_w .= lh

    for li in 0:(length(lm.backbone.blocks)-1)
        bid = string(li)
        blk = lm.backbone.blocks[li+1]
        blk.norm_w .= vec(_w(d, "model", "layers", bid, "norm", "weight"))
        blk.block_type == :mamba || error("expected Mamba block at layer $li")
        mix = blk.mixer
        mix.in_proj_w .= _w(d, "model", "layers", bid, "mixer", "in_proj", "weight")
        mix.conv1d_w .= _w(d, "model", "layers", bid, "mixer", "conv1d", "weight")
        mix.conv1d_b .= vec(_w(d, "model", "layers", bid, "mixer", "conv1d", "bias"))
        mix.dt_bias .= vec(_w(d, "model", "layers", bid, "mixer", "dt_bias"))
        mix.A_log .= vec(_w(d, "model", "layers", bid, "mixer", "A_log"))
        mix.D .= vec(_w(d, "model", "layers", bid, "mixer", "D"))
        mix.norm_w .= vec(_w(d, "model", "layers", bid, "mixer", "norm", "weight"))
        mix.out_proj_w .= _w(d, "model", "layers", bid, "mixer", "out_proj", "weight")
    end
    return lm
end
