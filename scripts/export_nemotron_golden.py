#!/usr/bin/env python3
"""
Golden trace for Nemotron-H parity (Hugging Face `transformers` reference).

Builds a *tiny* 2×Mamba `NemotronHForCausalLM` with stock `transformers` (no NVIDIA
remote code, no `mamba_ssm` GPU kernels). Uses `torch_forward` / naive SSD path.

Requires: pip install transformers torch numpy

Usage:
  python3 scripts/export_nemotron_golden.py [--out PATH]

Default output: test/fixtures/golden_tiny_mm.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers.models.nemotron_h import NemotronHConfig, NemotronHForCausalLM


def tiny_mm_config() -> NemotronHConfig:
    return NemotronHConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        layers_block_type=["mamba", "mamba"],
        mamba_num_heads=4,
        mamba_head_dim=16,
        n_groups=2,
        ssm_state_size=16,
        conv_kernel=4,
        chunk_size=8,
        num_attention_heads=4,
        head_dim=16,
        num_key_value_heads=2,
        max_position_embeddings=512,
        use_cache=False,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "test/fixtures/golden_tiny_mm.npz",
    )
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    cfg = tiny_mm_config()
    torch.manual_seed(0)
    model = NemotronHForCausalLM(cfg).float().eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids, use_cache=False)
    logits = out.logits.numpy().astype(np.float32)

    payload: dict[str, np.ndarray] = {
        "input_ids": input_ids.numpy().astype(np.int64),
        "logits": logits,
    }
    sd = model.state_dict()
    for k, v in sd.items():
        arr = v.detach().cpu().numpy().astype(np.float32)
        if k.endswith("conv1d.weight") and arr.ndim == 3:
            assert arr.shape[1] == 1
            arr = arr[:, 0, :]
        key = "w__" + k.replace(".", "__")
        payload[key] = arr

    np.savez_compressed(args.out, **payload)
    print(f"Wrote {args.out} (logits {logits.shape})")


if __name__ == "__main__":
    main()
