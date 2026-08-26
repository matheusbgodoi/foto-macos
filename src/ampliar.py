#!/usr/bin/env python3
"""Upscale. SeedVR2 (MLX) por padrao, ESRGAN como alternativa.

Medido nesta maquina, 896x1216 -> 2x:
  SeedVR2 (mflux/MLX)      25,8 s   pico 11,3 GB   detalhe generativo real
  4x-UltraSharpV2 (ComfyUI) 183 s                  interpolacao esperta

SeedVR2 ganha nas duas pontas: 7x mais rapido e reconstroi microtextura
(fio de cabelo, poro, barba) em vez de so aumentar a nitidez de borda.
Ele NAO roda no ComfyUI/MPS (issues 15053/15785 abertas, 73-88 GB de consumo);
so no mflux.
"""
import argparse, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.environ.get("FOTO_PYTHON") or os.path.expanduser(
    os.environ.get("COMFYUI_DIR", "~/comfyui") + "/.venv/bin/python")
MFLUX = os.environ.get("MFLUX_SEEDVR2") or os.path.expanduser(
    "~/.local/bin/mflux-upscale-seedvr2")


def seedvr2(image, out, escala, softness):
    env = dict(os.environ, HF_HUB_DISABLE_XET="1")
    cmd = [MFLUX, "--image-path", image, "--resolution", f"{int(escala)}x",
           "--low-ram", "--softness", str(softness), "--output", out]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if os.path.exists(out):
        print(f"[ampliar] {out}  (SeedVR2)")
        return 0
    print("[ampliar] SeedVR2 falhou:", r.stderr.strip().splitlines()[-1:] or r.stdout[-300:])
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("imagem")
    ap.add_argument("--out", required=True, help="caminho do arquivo de saida")
    ap.add_argument("--escala", type=float, default=2.0)
    ap.add_argument("--softness", type=float, default=0.5,
                    help="0 = mais detalhe e mais risco de artefato; 1 = mais suave")
    ap.add_argument("--esrgan", action="store_true",
                    help="usa 4x-UltraSharpV2 no ComfyUI em vez do SeedVR2")
    a = ap.parse_args()

    img = os.path.abspath(os.path.expanduser(a.imagem))
    out = os.path.abspath(os.path.expanduser(a.out))

    if not a.esrgan:
        if seedvr2(img, out, a.escala, a.softness) == 0:
            return 0
        print("[ampliar] caindo para o ESRGAN")
    return subprocess.call([PY, os.path.join(HERE, "upscale.py"), img,
                            "--scale", str(a.escala),
                            "--out", os.path.splitext(os.path.basename(out))[0]])


if __name__ == "__main__":
    sys.exit(main())
