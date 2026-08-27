#!/usr/bin/env python3
"""Estado e inicio automatico do ComfyUI, sem depender do SDK MCP."""
from __future__ import annotations

import fcntl
import os
import subprocess
import time
import urllib.request
from urllib.parse import urlparse

COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
COMFY_DIR = os.path.abspath(os.path.expanduser(
    os.environ.get("COMFYUI_DIR", "~/comfyui")))
PY = os.environ.get("FOTO_PYTHON") or os.path.join(
    COMFY_DIR, ".venv", "bin", "python")


def comfy_ok() -> bool:
    try:
        with urllib.request.urlopen(COMFY + "/system_stats", timeout=3):
            pass
        return True
    except Exception:
        return False


def ensure_comfy() -> str | None:
    """Garante o servidor local ativo; devolve uma mensagem apenas ao falhar."""
    if comfy_ok():
        return None
    parsed = urlparse(COMFY)
    if parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        return f"ComfyUI remoto nao responde em {COMFY}; inicio automatico so vale para localhost"

    state_dir = os.path.expanduser("~/Library/Application Support/foto-macos")
    os.makedirs(state_dir, exist_ok=True)
    lock_path = os.path.join(state_dir, "comfy-start.lock")
    log_path = os.path.join(state_dir, "comfy-autostart.log")

    with open(lock_path, "a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if comfy_ok():
            return None
        if not os.path.isfile(PY):
            return f"Python do ComfyUI ausente: {PY}"
        command = [
            PY, os.path.join(COMFY_DIR, "main.py"),
            # Mage-Flow declara bf16/float32. --force-fp16 converte o modelo
            # antes de consultar essa declaracao e, no MPS, pode produzir NaNs
            # que o ComfyUI salva como um PNG preto. Dois GB reservados tambem
            # evitam o ciclo de offload causado pela antiga reserva de 6 GB.
            "--use-pytorch-cross-attention", "--reserve-vram", "2",
            "--port", str(parsed.port or 8188),
            "--listen", parsed.hostname or "127.0.0.1",
        ]
        try:
            with open(log_path, "ab", buffering=0) as log:
                process = subprocess.Popen(
                    command, cwd=COMFY_DIR, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True,
                    env=os.environ.copy(),
                )
        except OSError as exc:
            return f"nao foi possivel iniciar o ComfyUI: {exc}"
        deadline = time.time() + 120
        while time.time() < deadline:
            if comfy_ok():
                return None
            if process.poll() is not None:
                break
            time.sleep(1)
    return (f"ComfyUI nao iniciou em {COMFY}. Consulte {log_path} "
            f"(processo encerrou com {process.poll()}).")
