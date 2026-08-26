"""Nos ComfyUI que chamam os runtimes Apple do foto-macos.

O Krea 2 roda em MLX/MFLUX (mais eficiente no Mac) e devolve uma IMAGE normal
ao grafo. Portanto localhost, CLI e MCP compartilham o mesmo peso e LoRA.
"""
from __future__ import annotations

import os
import gc
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = os.path.abspath(os.path.expanduser(os.environ.get(
    "FOTO_MACOS_ROOT", str(Path(__file__).resolve().parents[2]))))
PYTHON = os.path.expanduser(os.environ.get(
    "FOTO_PYTHON", "~/comfyui/.venv/bin/python"))
COMFY = os.path.expanduser(os.environ.get("COMFYUI_DIR", "~/comfyui"))


def _release_comfy_models():
    """Libera pesos MPS antes de abrir um segundo runtime Apple.

    Estes nos nao dependem de um MODEL do grafo: Draw Things e MFLUX rodam em
    processos separados. Manter um Mage/SDXL antigo residente ao mesmo tempo
    faria os 24 GB entrarem em swap e pode provocar o encerramento do ComfyUI.
    """
    try:
        import comfy.model_management as model_management
        model_management.unload_all_models()
        model_management.soft_empty_cache()
    except Exception:
        # Versoes do ComfyUI variam; o subprocesso ainda pode rodar em low-ram.
        pass
    gc.collect()


class FotoMacosKrea2:
    """Gera com Krea 2/Famegrid por MLX dentro de um workflow ComfyUI."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompt": ("STRING", {"multiline": True, "default":
                "A candid natural photograph of a person in an ordinary real place"}),
            "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
            "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
            "style": (["natural", "iphone", "profissional"],),
            "famegrid_strength": ("FLOAT", {"default": 0.7, "min": 0.0,
                                               "max": 1.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "file")
    FUNCTION = "generate"
    CATEGORY = "foto-macos/Apple MLX"

    def generate(self, prompt, width, height, seed, style, famegrid_strength):
        _release_comfy_models()
        target = os.path.join(COMFY, "temp", f"foto_krea2_{time.time_ns()}.png")
        command = [
            PYTHON, os.path.join(ROOT, "src", "krea2.py"), prompt,
            "--saida", target, "--tamanho", f"{width}x{height}",
            "--seed", str(seed), "--estilo", style,
            "--peso", str(famegrid_strength),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=3600)
        if result.returncode or not os.path.isfile(target):
            detail = ((result.stdout or "") + (result.stderr or ""))[-2000:]
            raise RuntimeError(f"foto-macos Krea 2 falhou:\n{detail}")
        pixels = np.asarray(Image.open(target).convert("RGB"), dtype=np.float32) / 255.0
        return (torch.from_numpy(pixels)[None, ...], target)


class FotoMacosGerar:
    """Roteador Apple externo: Draw Things ou MLX, sem reentrar na fila."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompt": ("STRING", {"multiline": True, "default":
                "A candid natural photograph in an ordinary real place"}),
            "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
            "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
            "style": (["auto", "foto-natural", "iphone", "profissional", "produto",
                       "cartoon", "pixel-art", "ilustracao", "anime", "famegrid"],),
            # SDXL e FLUX.2 usam os workflows ComfyUI nativos 1--3. Chamar a
            # API do proprio ComfyUI deste no bloquearia a fila atual.
            "engine": (["auto", "drawthings", "krea2"],),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "file")
    FUNCTION = "generate"
    CATEGORY = "foto-macos/Router"

    def generate(self, prompt, width, height, seed, style, engine):
        _release_comfy_models()
        target = os.path.join(COMFY, "temp", f"foto_router_{time.time_ns()}.png")
        command = [
            PYTHON, os.path.join(ROOT, "src", "gerar_coringa.py"), prompt,
            "--saida", target, "--tamanho", f"{width}x{height}",
            "--seed", str(seed), "--estilo", style, "--motor", engine,
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=3600)
        if result.returncode or not os.path.isfile(target):
            detail = ((result.stdout or "") + (result.stderr or ""))[-2000:]
            raise RuntimeError(f"foto-macos roteador falhou:\n{detail}")
        pixels = np.asarray(Image.open(target).convert("RGB"), dtype=np.float32) / 255.0
        return (torch.from_numpy(pixels)[None, ...], target)


NODE_CLASS_MAPPINGS = {
    "FotoMacosKrea2": FotoMacosKrea2,
    "FotoMacosGerar": FotoMacosGerar,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FotoMacosKrea2": "Foto Macos · Krea 2 + Famegrid (MLX)",
    "FotoMacosGerar": "Foto Macos · Gerar rapido (Apple externo)",
}
