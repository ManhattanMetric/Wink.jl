```@meta
CurrentModule = Wink
```

# Local models

Wink can run a model **inside your Julia process**: no server, no HTTP, and no
C library. A GGUF file is memory-mapped, and the forward pass runs in Julia
arrays you can introspect like any other value in your session.

```julia
Wink.local_model!("path/to/model.gguf")                    # CPU
Wink.local_model!("path/to/model.gguf"; array = MtlArray)  # GPU prefill (with `using Metal`)
Wink.local_model!(nothing)                                 # unload, back to the configured provider
```

Loading a model routes every `ai>` turn through it and lowers
`CONFIG.context_budget` to 75% of the model's context window, so compaction
fires before the window fills.

## What it supports

| | |
|---|---|
| Architectures | gemma-3, gemma-4 (mixture-of-experts), OLMoE |
| Weights | F32, F16, BF16, and q4_0 / q4_1 / q8_0 / q6_K quantization |
| Tokenizers | SentencePiece (score-driven and gemma-4's merge-ranked rule), GPT-2 byte-level BPE |
| Tool calls | constrained at the logits on families with a canonical call syntax (gemma-4) |

Quantized weights are read straight from the mmap and dequantized inside the
matrix multiply, so a 4GB model loads in about a second and full precision is
never materialized. Every piece of this is validated against llama.cpp: the
test suite carries recorded llama.cpp outputs for synthetic models covering
each architecture, quantization, and tokenizer.

## Choosing a model

Wink is a tool-driven agent, so pick models for tool-calling ability first — a
model that chats beautifully but cannot drive tools is decorative here. Model
families without a canonical tool-call syntax run text-only.

## CPU and GPU

On Apple silicon, generation runs fastest on the CPU, which reaches parity
with llama.cpp through int8 dot-product kernels; passing `array = MtlArray`
additionally keeps a device copy of the weights for prefill, where the GPU is
about 3× faster. Any GPUArrays-compatible vendor package (Metal.jl, CUDA.jl,
AMDGPU.jl, oneAPI.jl) works the same way — Wink itself depends on none of
them.

## Reference

```@autodocs
Modules = [Wink.GGUF, Wink.Quant, Wink.SPMTokenizer, Wink.BPETokenizer,
    Wink.Gemma3, Wink.Gemma4, Wink.OLMoE]
```
