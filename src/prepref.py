#!/usr/bin/env python3
"""Prepara fotos de referencia de IDENTIDADE.

Duas coisas que arruinam identidade zero-shot e que sao culpa do preparo, nao
do modelo:
  1. orientacao errada (EXIF perdido na conversao de HEIC/RAW) — o modelo
     simplesmente descarta a referencia;
  2. rosto pequeno demais no quadro — o encoder de visao reduz a referencia
     (tipicamente lado maior de 384-512 px), entao um rosto de 200 px numa foto
     de corpo inteiro chega ao modelo com ~30 px de rosto: nao ha identidade ali.

Este script rotaciona automaticamente (testa as 4 orientacoes e fica com a que
o detector de rosto aprova) e emite um crop de rosto grande, que e o que deve
ser mandado como referencia de identidade.
"""
import argparse, os
from PIL import Image
import facedet


def auto_orient(path, tmpdir):
    """Testa as 4 rotacoes e devolve a imagem na orientacao em que o detector
    encontra o maior rosto."""
    im = Image.open(path).convert("RGB")
    best, best_area, best_det = None, 0, None
    for deg in (0, 90, 180, 270):
        cand = im.rotate(deg, expand=True) if deg else im
        p = os.path.join(tmpdir, f"_orient_{deg}.png")
        # detector trabalha em arquivo; downscale so para o teste, e barato
        probe = cand.copy(); probe.thumbnail((1400, 1400))
        probe.save(p)
        d = facedet.detect(p)
        os.remove(p)
        if not d:
            continue
        _, _, w, h = d["bbox"]
        if w * h > best_area:
            best, best_area, best_det = cand, w * h, deg
    return (best, best_det) if best is not None else (im, None)


def face_crop(im, out_path, target=1024, margin=1.9, tmpdir="/tmp"):
    p = os.path.join(tmpdir, "_fc.png")
    probe = im.copy(); probe.thumbnail((1400, 1400))
    probe.save(p)
    d = facedet.detect(p)
    os.remove(p)
    if not d:
        return None
    sx = im.width / probe.width
    x, y, w, h = [v * sx for v in d["bbox"]]
    cx, cy = x + w / 2, y + h / 2
    r = max(w, h) * margin / 2
    box = (max(0, cx - r), max(0, cy - r * 1.15),
           min(im.width, cx + r), min(im.height, cy + r * 1.15))
    c = im.crop([int(v) for v in box])
    if c.width < target:
        return None
    c = c.resize((target, int(target * c.height / c.width)), Image.LANCZOS)
    c.save(out_path, quality=98)
    return c.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--outdir", default=os.path.expanduser("~/Pictures/refs/id"))
    ap.add_argument("--target", type=int, default=1024)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    tmp = "/private/tmp/claude-501"
    os.makedirs(tmp, exist_ok=True)

    for p in a.images:
        p = os.path.expanduser(p)
        base = os.path.splitext(os.path.basename(p))[0]
        im, deg = auto_orient(p, tmp)
        full = os.path.join(a.outdir, f"{base}_full.png")
        im.save(full)
        fc = os.path.join(a.outdir, f"{base}_face.png")
        size = face_crop(im, fc, a.target, tmpdir=tmp)
        print(f"{base}: rotacao={deg}  full={im.size}  face={size or 'rosto pequeno demais'}")


if __name__ == "__main__":
    main()
