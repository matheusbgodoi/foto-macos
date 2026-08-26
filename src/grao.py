#!/usr/bin/env python3
"""Devolve o GRAO da foto original para dentro da regiao editada.

Toda foto real tem ruido: sensor, ISO, compressao JPEG. O decoder do VAE nao
reproduz isso — a area que o modelo redesenha sai lisa. Medido numa foto de
WhatsApp: a regiao original tinha desvio de alta frequencia 19,45 e a mesma
regiao depois da edicao, 2,04. Quase 10x menos. E esse contraste entre "area
granulada" e "area limpa" dentro da MESMA foto que denuncia a edicao — mais do
que a forma ou a cor do que foi gerado.

Metodo, sem mascara e sem parametro de regiao:
  1. mapa LOCAL de ruido (desvio do residual de alta frequencia, em janelas) na
     original e na editada
  2. deficit por janela = sqrt(max(0, sigma_orig^2 - sigma_edit^2))
  3. injeta ruido modulado por esse mapa

Assim o grao entra exatamente onde faltou, na intensidade que faltou, e nao
toca no que ja estava granulado. Ruido com componente de luminancia e uma
fracao de crominancia, que e como sensor real se comporta.
"""
import argparse
import numpy as np
from PIL import Image, ImageFilter


def _blur_f(a, r):
    """Blur gaussiano aproximado para arrays float. O PIL recusa GaussianBlur em
    modo 'F', entao aproxima por reducao/ampliacao (box + bilinear), que e
    suficiente para suavizar mapa de ruido e para dar correlacao espacial."""
    if r <= 0:
        return a
    h, w = a.shape
    sh, sw = max(1, int(h / max(1.0, r * 2))), max(1, int(w / max(1.0, r * 2)))
    im = Image.fromarray(a.astype(np.float32), mode="F")
    return np.asarray(im.resize((sw, sh), Image.BOX).resize((w, h), Image.BILINEAR))


def _hf_residual(arr, r=1.2):
    """Residual de alta frequencia por canal (o 'grão' e a microtextura)."""
    im = Image.fromarray(arr.astype(np.uint8))
    lp = np.asarray(im.filter(ImageFilter.GaussianBlur(r))).astype(np.float32)
    return arr.astype(np.float32) - lp


def _local_sigma(res, win=24):
    """Desvio local do residual, em blocos, devolvido no tamanho original."""
    g = res.mean(axis=2) if res.ndim == 3 else res
    h, w = g.shape
    bh, bw = max(1, h // win), max(1, w // win)
    # media de g^2 por bloco via redimensionamento (box filter barato e estavel)
    sq = Image.fromarray(np.clip(g ** 2, 0, 65535).astype(np.float32), mode="F")
    small = sq.resize((bw, bh), Image.BOX)
    back = small.resize((w, h), Image.BILINEAR)
    return np.sqrt(np.maximum(np.asarray(back), 0.0))


def match_grain(orig_path, edit_path, out_path, win=24, chroma=0.12,
                max_add=10.0, debug=None):
    o = Image.open(orig_path).convert("RGB")
    e = Image.open(edit_path).convert("RGB")
    if o.size != e.size:
        o = o.resize(e.size, Image.LANCZOS)
    O = np.asarray(o).astype(np.float32)
    E = np.asarray(e).astype(np.float32)

    so = _local_sigma(_hf_residual(O), win)
    se = _local_sigma(_hf_residual(E), win)

    # ALVO = nivel de ruido BASE da foto, nao o sigma local ponto a ponto.
    # O sigma local mede tambem TEXTURA (um logo estampado, folhagem), e usa-lo
    # como alvo injeta ruido enorme justamente onde a original tinha detalhe —
    # foi o que produziu uma nuvem de pixels coloridos no peito do moletom.
    # O percentil baixo dos sigmas locais representa as areas lisas, ou seja,
    # o grao puro do sensor/JPEG.
    base = float(np.percentile(so, 25))
    deficit = np.clip(base - se, 0.0, max_add)
    # suaviza o mapa para nao criar fronteira visivel entre bloco com e sem grao
    deficit = _blur_f(deficit, win / 4)

    rng = np.random.default_rng(0)
    luma = rng.standard_normal(E.shape[:2]).astype(np.float32)
    # grao de foto nao e ruido branco puro: tem correlacao espacial de ~1 px
    luma = luma * 0.7 + _blur_f(luma, 0.6) * 0.3
    luma /= (luma.std() + 1e-6)

    noise = np.repeat(luma[..., None], 3, axis=2)
    if chroma > 0:
        c = rng.standard_normal(E.shape).astype(np.float32)
        c = np.stack([_blur_f(c[..., k], 0.9) for k in range(3)], axis=2)
        c /= (c.std() + 1e-6)
        noise = noise * (1 - chroma) + c * chroma
        noise /= (noise.std() + 1e-6)

    out = np.clip(E + noise * deficit[..., None], 0, 255).astype(np.uint8)
    Image.fromarray(out).save(out_path)

    if debug:
        d = deficit / (deficit.max() + 1e-6) * 255
        Image.fromarray(d.astype(np.uint8)).save(debug)
    return float(deficit.mean()), float(deficit.max())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("original"); ap.add_argument("editada"); ap.add_argument("saida")
    ap.add_argument("--janela", type=int, default=24)
    ap.add_argument("--chroma", type=float, default=0.12)
    ap.add_argument("--max", type=float, default=10.0)
    ap.add_argument("--debug-map", default=None)
    a = ap.parse_args()
    m, mx = match_grain(a.original, a.editada, a.saida, a.janela, a.chroma,
                        a.max, a.debug_map)
    print(f"[grao] {a.saida} — grao injetado: medio {m:.2f}, maximo {mx:.2f}")
