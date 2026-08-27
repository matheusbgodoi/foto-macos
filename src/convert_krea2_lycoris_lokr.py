#!/usr/bin/env python3
"""Converte LoKr Krea 2 do SimpleTuner para MFLUX.

O conversor aceita tanto Q/K/V separados quanto o alvo fundido
``lycoris_*_attn_to_qkv``. O Krea 2 do MFLUX usa Q, K e V separados. Como a
decomposicao LoKr usada aqui e ``kron(w1, w2)``, e as fronteiras Q/K/V caem
exatamente entre linhas de ``w2``, a conversao reutiliza ``w1`` e fatia ``w2``
sem aproximacao.

No modo ``full_matrix`` usado nos testes, o LyCORIS força ``scale=1`` quando
``w1`` e ``w2`` sao matrizes diretas. O ``alpha`` salvo pode aparecer como
9984 em BF16 embora o rank configurado seja 10000; ele e preservado como
metadado do adapter, mas nao representa uma atenuacao que deva ser aplicada
novamente pelo conversor.
"""
from __future__ import annotations

import argparse
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BASE_RE = re.compile(
    r"^lycoris_(text_fusion_layerwise_blocks|text_fusion_refiner_blocks|"
    r"transformer_blocks)_(\d+)_attn_to_(qkv|q|k|v|gate|out_0)$"
)
GROUPS = {
    "text_fusion_layerwise_blocks": "text_fusion.layerwise_blocks",
    "text_fusion_refiner_blocks": "text_fusion.refiner_blocks",
    "transformer_blocks": "transformer_blocks",
}
DIRECT = ("gate", "out_0")
FACTORS = ("lokr_w1", "lokr_w2", "alpha")


def _target(group: str, block: str, projection: str) -> str:
    suffix = "out.0" if projection == "out_0" else projection
    return f"{GROUPS[group]}.{block}.attn.to_{suffix}"


def convert_tensors(source: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    bases = {key.rsplit(".", 1)[0] for key in source}
    output: dict[str, torch.Tensor] = {}
    effective_w1 = []

    for base in sorted(bases):
        match = BASE_RE.match(base)
        if not match:
            continue
        group, block, projection = match.groups()

        if projection in DIRECT:
            target = _target(group, block, projection)
            for factor in FACTORS:
                key = f"{base}.{factor}"
                if key in source:
                    output[f"{target}.{factor}"] = source[key]
            effective_w1.append(source[f"{base}.lokr_w1"])
            continue

        if projection in ("q", "k", "v"):
            # Quando QKV esta fundido, os alvos individuais permanecem no
            # arquivo como modulos originais mas nao dirigem o forward. Sem a
            # fusao, eles sao os fatores treinados e podem ser copiados direto.
            qkv_base = base.rsplit("_", 1)[0] + "_qkv"
            if qkv_base in bases:
                continue
            target = _target(group, block, projection)
            for factor in FACTORS:
                key = f"{base}.{factor}"
                if key in source:
                    output[f"{target}.{factor}"] = source[key]
            effective_w1.append(source[f"{base}.lokr_w1"])
            continue

        if projection != "qkv":
            continue

        w1 = source[f"{base}.lokr_w1"]
        w2 = source[f"{base}.lokr_w2"]
        alpha = source.get(f"{base}.alpha")
        sibling_prefix = base.removesuffix("qkv")
        row_sizes = [
            source[f"{sibling_prefix}{part}.lokr_w2"].shape[0]
            for part in ("q", "k", "v")
        ]
        if sum(row_sizes) != w2.shape[0]:
            raise ValueError(
                f"QKV incompativel em {base}: {w2.shape[0]} != {row_sizes}"
            )
        start = 0
        for part, size in zip(("q", "k", "v"), row_sizes, strict=True):
            target = _target(group, block, part)
            output[f"{target}.lokr_w1"] = w1
            output[f"{target}.lokr_w2"] = w2[start:start + size].contiguous()
            if alpha is not None:
                output[f"{target}.alpha"] = alpha
            start += size
        effective_w1.append(w1)

    if not output:
        raise ValueError("nenhum alvo LyCORIS Krea 2 reconhecido")
    if not effective_w1 or all(float(t.float().norm()) == 0 for t in effective_w1):
        raise ValueError(
            "adapter sem efeito: todos os lokr_w1 usados pelo forward estao zerados"
        )
    return output


def convert_file(source_path: str, destination_path: str) -> None:
    tensors: dict[str, torch.Tensor] = {}
    metadata = {}
    with safe_open(source_path, framework="pt", device="cpu") as handle:
        metadata.update(handle.metadata() or {})
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
    converted = convert_tensors(tensors)
    metadata.update({
        "foto_macos_conversion": "simpletuner_krea2_lycoris_lokr_to_mflux_v1",
        "source_file": os.path.basename(source_path),
    })
    save_file(converted, destination_path, metadata=metadata)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    convert_file(
        os.path.abspath(os.path.expanduser(args.source)),
        os.path.abspath(os.path.expanduser(args.destination)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
