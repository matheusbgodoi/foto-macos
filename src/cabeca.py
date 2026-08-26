#!/usr/bin/env python3
"""Recola a CABECA da foto original quando a pose nao mudou.

Quando a edicao e sobre a propria foto (trocar roupa, trocar objeto), a pessoa
nao se move: a cabeca esta no mesmo lugar na entrada e na saida. Entao nao ha
transformacao para estimar — basta compor de volta a regiao da cabeca com os
pixels originais. Isso preserva rosto E cabelo exatamente, que e onde o modelo
mais estraga (achata o cabelo, suaviza a pele, muda a expressao).

Diferente de face1to1.py: aqui NAO ha warp, e a mascara cobre cabelo e queixo,
nao so o oval do rosto. So use quando a pose for a mesma.
"""
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import facedet


def head_mask(det, shape, largura=1.9, altura_cima=2.1, altura_baixo=1.35):
    """Elipse em volta do bbox do rosto, alargada para conter cabelo e queixo."""
    h, w = shape[:2]
    x, y, bw, bh = det["bbox"]
    cx, cy = x + bw / 2, y + bh / 2
    rx = bw / 2 * largura
    ry_up, ry_dn = bh / 2 * altura_cima, bh / 2 * altura_baixo
    cy_e = cy - (ry_up - ry_dn) / 2
    ry = (ry_up + ry_dn) / 2
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).ellipse([cx - rx, cy_e - ry, cx + rx, cy_e + ry], fill=255)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original")
    ap.add_argument("editada")
    ap.add_argument("saida")
    ap.add_argument("--feather", type=int, default=22)
    ap.add_argument("--largura", type=float, default=1.9)
    ap.add_argument("--cima", type=float, default=2.1)
    ap.add_argument("--baixo", type=float, default=1.35)
    ap.add_argument("--debug-mask", default=None)
    a = ap.parse_args()

    o = Image.open(a.original).convert("RGB")
    e = Image.open(a.editada).convert("RGB")

    det = facedet.detect(a.original)
    if det is None:
        raise SystemExit("nenhum rosto detectado na foto original")

    # Trabalha na resolucao da EDITADA (que ja passou pelo upscale e costuma ser
    # maior). Antes o script forcava tudo para o tamanho da original e jogava
    # fora a resolucao ganha no upscale.
    if o.size != e.size:
        sx, sy = e.width / o.width, e.height / o.height
        o = o.resize(e.size, Image.LANCZOS)
        x, y, bw, bh = det["bbox"]
        det = dict(det, bbox=(x * sx, y * sy, bw * sx, bh * sy))

    m = head_mask(det, (e.height, e.width), a.largura, a.cima, a.baixo)
    m = m.filter(ImageFilter.GaussianBlur(a.feather))
    mask = np.asarray(m).astype(np.float32)[..., None] / 255.0

    out = np.asarray(e).astype(np.float32) * (1 - mask) + np.asarray(o).astype(np.float32) * mask
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(a.saida)
    if a.debug_mask:
        m.save(a.debug_mask)
    print(f"[cabeca] {a.saida} — cabeca original recolada ({mask.mean()*100:.1f}% do quadro)")


if __name__ == "__main__":
    main()
