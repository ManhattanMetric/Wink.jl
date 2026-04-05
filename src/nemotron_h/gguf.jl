using PythonCall

"""
    gguf_dequant_dict(path) -> Dict{String, Array{Float32}}

Read all tensors from a GGUF file and return dequantized `Float32` arrays (CPU).
Requires Python `gguf` (from [llama.cpp gguf-py](https://github.com/ggml-org/llama.cpp/tree/master/gguf-py)):

```bash
pip install gguf numpy
```
"""
function gguf_dequant_dict(path::AbstractString)
    gguf_mod = pyimport("gguf")
    quants = pyimport("gguf.quants")
    reader = gguf_mod.GGUFReader(path)
    out = Dict{String,Array{Float32}}()
    for t in reader.tensors
        name = pyconvert(String, t.name)
        raw = t.data
        qt = t.tensor_type
        arr = quants.dequantize(raw, qt)
        x = pyconvert(Array{Float32}, arr)
        out[name] = x
    end
    return out
end

function _get2(d::Dict, k::String)
    haskey(d, k) || error("missing GGUF tensor: $k")
    d[k]
end

"""Transpose last two dims if `a` is matrix (handles row-major numpy layout)."""
function np_matrix_to_julia(a::AbstractMatrix{Float32})
    Matrix{Float32}(collect(a'))
end

function np_matrix_to_julia_t(a::AbstractMatrix{Float32})
    Matrix{Float32}(collect(a))
end

"""
    load_nemotron_h_gguf!(lm::NemotronHCausalLM, path)

Fill `lm` with dequantized weights from an NVIDIA Nemotron-H GGUF (e.g. Q4_K_M).
Tensor names follow llama.cpp / gguf-py (`blk.N.*`).

**Validation**: BF16 Hugging Face logits are a tight reference; Q4_K_M dequant introduces
weight error, so compare against the same GGUF dequantized in Python (`gguf.quants.dequantize`)
or llama.cpp, not against BF16. For a Float32 CPU Nemotron-H forward, expect substantially
larger drift than the tiny `test/fixtures/golden_tiny_mm.npz` harness (~1e-1 abs on logits
for that toy model).
"""
function load_nemotron_h_gguf!(lm::NemotronHCausalLM, path::AbstractString)
    d = gguf_dequant_dict(path)
    cfg = lm.cfg
    kinds = layers_block_type(cfg)

    te = _get2(d, "token_embd.weight")
    lm.backbone.tok_emb .= np_matrix_to_julia(te)

    on = _get2(d, "output_norm.weight")
    lm.backbone.norm_f_w .= vec(on)

    if haskey(d, "output.weight")
        ow = _get2(d, "output.weight")
        lm.lm_head_w .= np_matrix_to_julia(ow)
    end

    for (li, blk) in enumerate(lm.backbone.blocks)
        i0 = li - 1
        bid = string(i0)
        an = _get2(d, "blk.$bid.attn_norm.weight")
        blk.norm_w .= vec(an)

        typ = kinds[li]
        if typ == :mamba
            mix = blk.mixer::NemotronHMambaLayer
            w = _get2(d, "blk.$bid.ssm_in.weight")
            mix.in_proj_w .= np_matrix_to_julia(w)
            cw = _get2(d, "blk.$bid.ssm_conv1d.weight")
            mix.conv1d_w .= np_matrix_to_julia_t(cw)
            if haskey(d, "blk.$bid.ssm_conv1d.bias")
                mix.conv1d_b .= vec(_get2(d, "blk.$bid.ssm_conv1d.bias"))
            end
            mix.dt_bias .= vec(_get2(d, "blk.$bid.ssm_dt.bias"))
            a = _get2(d, "blk.$bid.ssm_a")
            aflat = vec(a)
            mix.A_log .= log.(max.(abs.(aflat), 1f-10))
            dvec = vec(_get2(d, "blk.$bid.ssm_d"))
            mix.D .= dvec
            nw = _get2(d, "blk.$bid.ssm_norm.weight")
            if ndims(nw) == 2
                g1, g2 = size(nw)
                mix.norm_w .= vec(permutedims(nw, (2, 1)))
            else
                mix.norm_w .= vec(nw)
            end
            ow = _get2(d, "blk.$bid.ssm_out.weight")
            mix.out_proj_w .= np_matrix_to_julia(ow)
        elseif typ == :attention
            mix = blk.mixer::NemotronHAttentionLayer
            mix.q_proj_w .= np_matrix_to_julia(_get2(d, "blk.$bid.attn_q.weight"))
            mix.k_proj_w .= np_matrix_to_julia(_get2(d, "blk.$bid.attn_k.weight"))
            mix.v_proj_w .= np_matrix_to_julia(_get2(d, "blk.$bid.attn_v.weight"))
            mix.o_proj_w .= np_matrix_to_julia(_get2(d, "blk.$bid.attn_output.weight"))
        else
            mix = blk.mixer::NemotronHMLPLayer
            mix.up_w .= np_matrix_to_julia(_get2(d, "blk.$bid.ffn_up.weight"))
            mix.down_w .= np_matrix_to_julia(_get2(d, "blk.$bid.ffn_down.weight"))
        end
    end
    return lm
end
