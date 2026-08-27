#!/usr/bin/env python3
"""Diagnostico compartilhado pelo CLI e pelo MCP."""
from __future__ import annotations

import os

from comfy_service import COMFY, comfy_ok
from krea2 import identity_registry, model_path as krea_model_path


def report() -> str:
    models = os.path.expanduser(
        os.environ.get("COMFYUI_DIR", "~/comfyui") + "/models")
    required = {
        "editar": f"{models}/diffusion_models/mage_flow_edit_turbo_bf16.safetensors",
        "encoder": f"{models}/text_encoders/qwen3vl_4b_bf16.safetensors",
        "vae": f"{models}/vae/mage_flow_vae_bf16.safetensors",
        "polir": f"{models}/checkpoints/RealVisXL_V5.0_fp16.safetensors",
        "pele": f"{models}/upscale_models/1x-ITF-SkinDiffDetail-Lite-v1.pth",
        "gerar/referencias": f"{models}/diffusion_models/flux-2-klein-4b.safetensors",
    }
    lines = [f"ComfyUI em {COMFY}: {'no ar' if comfy_ok() else 'FORA DO AR'}"]
    for role, path in required.items():
        lines.append(
            f"  {'ok   ' if os.path.exists(path) else 'FALTA'} {role}: {os.path.basename(path)}")

    draw_cli = os.path.expanduser(os.environ.get(
        "DRAWTHINGS_BIN", "/opt/homebrew/bin/draw-things-cli"))
    draw_model = os.path.expanduser(os.path.join(os.environ.get(
        "DRAWTHINGS_MODELS_DIR",
        "~/Library/Application Support/local-photo-ai-m5/models"),
        "z_image_turbo_1.0_i8x.ckpt"))
    draw_ok = os.path.isfile(draw_cli) and os.path.isfile(draw_model)
    lines.append(
        f"  {'ok   ' if draw_ok else 'opcional ausente'} gerar rapido: Draw Things + Z-Image i8x")

    famegrid = os.path.expanduser(
        "~/Library/Application Support/foto-macos/loras/krea2/"
        "Famegrid-Natural-V1-Krea-2.safetensors")
    krea_ok = os.path.isfile(famegrid) and bool(krea_model_path())
    lines.append(
        f"  {'ok   ' if krea_ok else 'opcional ausente'} fotorrealismo: Krea 2 Q4 + Famegrid")
    for name, config in identity_registry().items():
        path = os.path.abspath(os.path.expanduser(str(config.get("lora", ""))))
        lines.append(
            f"  {'ok   ' if os.path.isfile(path) else 'PENDENTE'} identidade {name}: "
            f"{os.path.basename(path) or 'LoRA sem caminho'}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
