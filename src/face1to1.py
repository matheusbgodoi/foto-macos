#!/usr/bin/env python3
"""Camada de IDENTIDADE — devolve o rosto ORIGINAL para dentro da imagem gerada.

Modelo de edicao/geracao nenhum preserva identidade: ele RECRIA uma pessoa
parecida. Esta camada resolve pelo outro lado — pega o rosto real da foto de
referencia e o recoloca na imagem de saida, alinhado por landmarks faciais:

  1. detecta 478 landmarks (MediaPipe FaceMesh) na original e na gerada
  2. estima a transformacao de similaridade (escala + rotacao + translacao) que
     leva o rosto original ao lugar/tamanho/angulo do rosto gerado
  3. deforma a original por essa transformacao
  4. monta a mascara pelo contorno facial detectado NA GERADA, com feather
  5. casa a cor/luz do rosto original com a iluminacao da cena gerada
  6. compoe

Limite honesto: so funciona quando os dois rostos estao em pose parecida
(similaridade e 4 graus de liberdade, nao reconstroi 3D). Se a cabeca gerada
estiver num angulo muito diferente da referencia, o resultado fica colado.
"""
import argparse
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

import facedet


def detect(path):
    d = facedet.detect(path)
    if d is None:
        return None, None, np.asarray(Image.open(path).convert("RGB"))
    return d, facedet.anchors(d), np.asarray(Image.open(path).convert("RGB"))


def similarity_transform(src, dst):
    """Umeyama: escala + rotacao + translacao que melhor leva src em dst."""
    src_m, dst_m = src.mean(0), dst.mean(0)
    s, d = src - src_m, dst - dst_m
    cov = d.T @ s / len(src)
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    scale = S.sum() / (s ** 2).sum() * len(src)
    t = dst_m - scale * R @ src_m
    return scale * R, t


def oval_mask(det, shape, grow=1.05, up=1.18):
    """Elipse sobre o bbox do rosto. O faceContour do Vision vai so do queixo as
    temporas — nao cobre testa nem sobrancelha — entao uma elipse alargada em
    torno do bbox cobre melhor a area que define identidade. 'up' estica para
    cima para pegar a testa."""
    h, w = shape[:2]
    x, y, bw, bh = det["bbox"]
    cx, cy = x + bw / 2, y + bh / 2
    rx, ry = bw / 2 * grow, bh / 2 * grow * up
    cy -= bh * (up - 1) * 0.18
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    return np.asarray(im).astype(np.float32) / 255.0


def match_color(src, dst, mask):
    """Casa media/desvio do rosto colado com o rosto da cena, dentro da mascara."""
    out = src.copy()
    sel = mask > 0.5
    if sel.sum() < 50:
        return out
    for c in range(3):
        s, d = src[..., c][sel], dst[..., c][sel]
        ss = s.std() + 1e-3
        g = float(np.clip(d.std() / ss, 0.7, 1.4))
        out[..., c] = np.clip((src[..., c] - s.mean()) * g + d.mean(), 0, 255)
    return out


def restore(identity_path, generated_path, out_path, feather=12, grow=1.05,
            debug=None):
    src_det, src_pts, src = detect(identity_path)
    dst_det, dst_pts, dst = detect(generated_path)
    if src_pts is None:
        raise SystemExit("nenhum rosto detectado na foto de identidade")
    if dst_pts is None:
        raise SystemExit("nenhum rosto detectado na imagem gerada")

    M, t = similarity_transform(src_pts, dst_pts)
    # PIL.transform usa o mapeamento INVERSO (destino -> origem)
    Minv = np.linalg.inv(M)
    tinv = -Minv @ t
    coeffs = (Minv[0, 0], Minv[0, 1], tinv[0], Minv[1, 0], Minv[1, 1], tinv[1])
    h, w = dst.shape[:2]
    warped = np.asarray(Image.fromarray(src).transform(
        (w, h), Image.AFFINE, coeffs, resample=Image.BICUBIC)).astype(np.float32)

    # Mascara = elipse do rosto NA CENA  ∩  elipse do rosto NA REFERENCIA apos o
    # warp. Sem a segunda, quando a referencia e um crop apertado a elipse do
    # destino cai fora da cabeca de origem e o composite arrasta o FUNDO da foto
    # de referencia (madeira, parede) para dentro da cena.
    m_dst = oval_mask(dst_det, dst.shape, grow)
    m_src = oval_mask(src_det, src.shape, grow)
    m_src_w = np.asarray(Image.fromarray((m_src * 255).astype(np.uint8)).transform(
        (w, h), Image.AFFINE, coeffs, resample=Image.BILINEAR)).astype(np.float32) / 255.0
    m = np.minimum(m_dst, m_src_w)
    m = np.asarray(Image.fromarray((m * 255).astype(np.uint8))
                   .filter(ImageFilter.GaussianBlur(feather))).astype(np.float32) / 255.0

    dstf = dst.astype(np.float32)
    warped = match_color(warped, dstf, m)
    out = dstf * (1 - m[..., None]) + warped * m[..., None]
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(out_path)
    if debug:
        Image.fromarray((m * 255).astype(np.uint8)).save(debug)
    scale = float(np.sqrt(abs(np.linalg.det(M))))
    return scale, m.mean()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("identity", help="foto real da pessoa (rosto nitido)")
    ap.add_argument("generated", help="imagem gerada/editada")
    ap.add_argument("out")
    ap.add_argument("--feather", type=int, default=12)
    ap.add_argument("--grow", type=float, default=1.05)
    ap.add_argument("--debug-mask", default=None)
    a = ap.parse_args()
    s, frac = restore(a.identity, a.generated, a.out, a.feather, a.grow, a.debug_mask)
    print(f"[face] {a.out} — rosto reescalado {s:.2f}x, cobre {frac*100:.1f}% do quadro")
