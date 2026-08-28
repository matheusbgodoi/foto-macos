#!/usr/bin/env python3
"""Upscale. SeedVR2 (MLX) por padrao, Lanczos deterministico como fallback.

Medido nesta maquina, 896x1216 -> 2x:
  SeedVR2 (mflux/MLX)      25,8 s   pico 11,3 GB   detalhe generativo real
SeedVR2 reconstroi microtextura, portanto pode inventar detalhes. Em edicoes,
o pipeline o executa antes de recolar a cabeca original. Se ele falhar, o
fallback Lanczos apenas redimensiona a imagem e nao inventa pixels.
Ele NAO roda no ComfyUI/MPS (issues 15053/15785 abertas, 73-88 GB de consumo);
so no mflux.
"""
import argparse, os, subprocess, sys, uuid
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.environ.get("FOTO_PYTHON") or os.path.expanduser(
    os.environ.get("COMFYUI_DIR", "~/comfyui") + "/.venv/bin/python")
MFLUX = os.environ.get("MFLUX_SEEDVR2") or os.path.expanduser(
    "~/.local/bin/mflux-upscale-seedvr2")


def seedvr2(image, out, escala, softness):
    env = dict(os.environ, HF_HUB_DISABLE_XET="1")
    root, extension = os.path.splitext(out)
    temporary = f"{root}.partial-{uuid.uuid4().hex[:8]}{extension or '.png'}"
    cmd = [MFLUX, "--image-path", image, "--resolution", f"{escala:g}x",
           "--low-ram", "--softness", str(softness), "--output", temporary]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    except OSError as error:
        print(f"[ampliar] SeedVR2 indisponivel: {error}")
        return 1
    if r.returncode == 0 and os.path.exists(temporary):
        os.replace(temporary, out)
        print(f"[ampliar] {out}  (SeedVR2)")
        return 0
    if os.path.exists(temporary):
        os.unlink(temporary)
    print("[ampliar] SeedVR2 falhou:", r.stderr.strip().splitlines()[-1:] or r.stdout[-300:])
    return 1


def lanczos(image, out, escala):
    with Image.open(image) as source:
        target = (max(1, round(source.width * escala)),
                  max(1, round(source.height * escala)))
        source.convert("RGB").resize(target, Image.Resampling.LANCZOS).save(out)
    print(f"[ampliar] {out}  (Lanczos deterministico)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("imagem")
    ap.add_argument("--out", required=True, help="caminho do arquivo de saida")
    ap.add_argument("--escala", type=float, default=2.0)
    ap.add_argument(
        "--modo", choices=("fiel", "equilibrado", "criativo"),
        default="equilibrado",
        help=("fiel: sem pre-downsampling (softness 0); equilibrado: 0.5; "
              "criativo: 0.75, reconstrucao mais livre e maior risco de mudar tracos"),
    )
    ap.add_argument(
        "--softness", type=float, default=None,
        help=("ajuste avancado que substitui --modo. 0 preserva mais a entrada; "
              "valores maiores pre-reduzem a imagem e dao mais liberdade ao modelo"),
    )
    a = ap.parse_args()

    img = os.path.abspath(os.path.expanduser(a.imagem))
    out = os.path.abspath(os.path.expanduser(a.out))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    presets = {"fiel": 0.0, "equilibrado": 0.5, "criativo": 0.75}
    softness = a.softness if a.softness is not None else presets[a.modo]
    if not 0.0 <= softness <= 1.0:
        ap.error("--softness precisa estar entre 0 e 1")
    print(f"[ampliar] modo={a.modo} softness={softness:g}")
    if seedvr2(img, out, a.escala, softness) == 0:
        return 0
    print("[ampliar] caindo para Lanczos; nenhum detalhe sera inventado")
    return lanczos(img, out, a.escala)


if __name__ == "__main__":
    sys.exit(main())
