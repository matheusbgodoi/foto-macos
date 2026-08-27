#!/usr/bin/env python3
"""Converte LoRA Krea 2 PEFT com ``to_qkv`` fundido para Q/K/V separados.

O SimpleTuner recomenda ``fuse_qkv_projections=true`` e, nesse formato, salva
uma matriz A compartilhada e uma matriz B contendo Q, K e V concatenados. O
MFLUX representa o Krea 2 com ``wq``, ``wk`` e ``wv`` separados. A conversao e
exata: cada destino reutiliza A e recebe a fatia correspondente de B.

Nenhuma imagem, caption ou peso-base e lido. Somente o adapter e reempacotado.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
from safetensors import safe_open


QKV_A = re.compile(r"^(.*\.attn\.)to_qkv\.lora_A\.weight$")
QKV_B = re.compile(r"^(.*\.attn\.)to_qkv\.lora_B\.weight$")


def tensor_copy(value):
    """Copia arrays NumPy em testes e tensores Torch bfloat16 em arquivos reais."""
    return value.clone() if hasattr(value, "clone") else value.copy()


def tensor_norm(value) -> float:
    """Norma em float32 para detectar adapters ainda na inicializacao zero."""
    if hasattr(value, "float"):
        return float(value.float().norm())
    return float(np.linalg.norm(value.astype(np.float32, copy=False)))


def qkv_sizes(prefix: str, total: int) -> tuple[int, int, int]:
    """Retorna as dimensoes de saida Q/K/V da arquitetura Krea 2."""
    if ".transformer_blocks." in prefix:
        # 48 query heads e 12 KV heads, head_dim=128.
        expected = (6144, 1536, 1536)
    elif ".text_fusion." in prefix:
        # 20 query e 20 KV heads, head_dim=128.
        expected = (2560, 2560, 2560)
    else:
        raise ValueError(f"to_qkv fora de um modulo Krea 2 conhecido: {prefix}")
    if sum(expected) != total:
        raise ValueError(
            f"shape to_qkv inesperado em {prefix}: saida={total}, "
            f"esperado={sum(expected)}"
        )
    return expected


def convert_tensors(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Converte um dicionario de tensores e rejeita adapters incompletos."""
    converted: dict[str, np.ndarray] = {}
    fused_prefixes: set[str] = set()
    b_norms: list[float] = []

    for key, value in tensors.items():
        if ".lora_B." in key:
            b_norms.append(tensor_norm(value))
        match_a = QKV_A.match(key)
        match_b = QKV_B.match(key)
        if match_a:
            prefix = match_a.group(1)
            fused_prefixes.add(prefix)
            for name in ("q", "k", "v"):
                converted[f"{prefix}to_{name}.lora_A.weight"] = tensor_copy(value)
        elif match_b:
            prefix = match_b.group(1)
            fused_prefixes.add(prefix)
            sizes = qkv_sizes(prefix, value.shape[0])
            start = 0
            for name, size in zip(("q", "k", "v"), sizes):
                converted[f"{prefix}to_{name}.lora_B.weight"] = tensor_copy(value[start : start + size])
                start += size
        else:
            converted[key] = value

    if not fused_prefixes:
        raise ValueError("o adapter nao contem nenhuma camada attn.to_qkv fundida")
    if not b_norms or not any(norm > 0 for norm in b_norms):
        raise ValueError(
            "todos os tensores lora_B estao zerados; o checkpoint nao aprendeu"
        )
    for prefix in fused_prefixes:
        for name in ("q", "k", "v"):
            for matrix in ("A", "B"):
                key = f"{prefix}to_{name}.lora_{matrix}.weight"
                if key not in converted:
                    raise ValueError(f"adapter QKV incompleto; falta {key}")
    return converted


def convert_file(source: str, destination: str) -> tuple[int, int]:
    # Torch e intencional: o adapter Krea e bfloat16, dtype que a ponte NumPy
    # do safetensors nao representa sem uma conversao destrutiva para float32.
    from safetensors.torch import save_file

    tensors = {}
    metadata = None
    with safe_open(source, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)

    converted = convert_tensors(tensors)
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    save_file(converted, destination, metadata=metadata)
    return len(tensors), len(converted)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="LoRA PEFT/SimpleTuner com to_qkv")
    parser.add_argument("destination", help="novo .safetensors compativel com MFLUX")
    args = parser.parse_args()
    source = os.path.abspath(os.path.expanduser(args.source))
    destination = os.path.abspath(os.path.expanduser(args.destination))
    if source == destination:
        print("erro: origem e destino devem ser arquivos diferentes", file=sys.stderr)
        return 2
    if not os.path.isfile(source):
        print(f"erro: arquivo ausente: {source}", file=sys.stderr)
        return 2
    try:
        before, after = convert_file(source, destination)
    except (OSError, ValueError) as error:
        print(f"erro: {error}", file=sys.stderr)
        return 1
    print(f"convertido: {before} -> {after} tensores\n{destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
