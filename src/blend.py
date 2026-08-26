#!/usr/bin/env python3
"""Blend multi-banda (piramide Laplaciana) — compoe sem deixar emenda visivel.

O composite alfa simples falha quando as duas fontes tem estatisticas
diferentes. Aqui elas tem: a foto original e crua; a editada passou pelo VAE do
modelo e por um polimento SDXL, entao mudou tom, nitidez e nivel de grao.
Misturar as duas com uma unica mascara borrada deixa uma assinatura — no realce
high-pass da saida, a borda oval da mascara aparece como uma linha clara em
volta da cabeca.

A piramide Laplaciana resolve porque mistura cada banda de frequencia com uma
largura de transicao proporcional a sua escala:

  * baixas frequencias (cor, iluminacao) -> transicao LARGA, entao nao ha
    degrau de tom;
  * altas frequencias (detalhe, grao)    -> transicao ESTREITA, entao o
    detalhe nao vira media borrada dos dois lados.

E o mesmo metodo usado para costurar panoramas (Burt & Adelson, 1983).
"""
import numpy as np
from PIL import Image, ImageFilter


def _down(a):
    """Reduz pela metade com pre-filtro gaussiano (um nivel da piramide)."""
    h, w = a.shape[:2]
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    im = im.filter(ImageFilter.GaussianBlur(1.0))
    return np.asarray(im.resize((max(1, w // 2), max(1, h // 2)), Image.BILINEAR)).astype(np.float32)


def _up(a, size):
    """Amplia para o tamanho dado (w, h)."""
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return np.asarray(im.resize(size, Image.BILINEAR)).astype(np.float32)


def _down_mask(m):
    h, w = m.shape
    im = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8))
    im = im.filter(ImageFilter.GaussianBlur(1.0))
    return np.asarray(im.resize((max(1, w // 2), max(1, h // 2)), Image.BILINEAR)).astype(np.float32) / 255.0


def niveis_para(mask, teto=7):
    """Numero de niveis: o suficiente para que a menor dimensao da regiao da
    mascara chegue a poucos pixels no topo da piramide. Poucos niveis deixam
    halo de cor; muitos custam tempo sem ganho."""
    ys, xs = np.where(mask > 0.01)
    if len(ys) == 0:
        return 4
    lado = max(ys.max() - ys.min(), xs.max() - xs.min(), 1)
    n = int(np.floor(np.log2(max(lado, 2)))) - 2
    return int(np.clip(n, 3, teto))


def blend(base, sobre, mask, levels=None):
    """Compoe `sobre` sobre `base` na regiao `mask` (float 0..1, HxW).

    base, sobre: arrays HxWx3 float32 (0..255), mesmo tamanho.
    """
    base = base.astype(np.float32)
    sobre = sobre.astype(np.float32)
    if levels is None:
        levels = niveis_para(mask)

    # piramides gaussianas
    gb, gs, gm = [base], [sobre], [mask.astype(np.float32)]
    for _ in range(levels):
        gb.append(_down(gb[-1]))
        gs.append(_down(gs[-1]))
        gm.append(_down_mask(gm[-1]))

    # piramides laplacianas (detalhe de cada banda)
    lb, ls = [], []
    for i in range(levels):
        tam = (gb[i].shape[1], gb[i].shape[0])
        lb.append(gb[i] - _up(gb[i + 1], tam))
        ls.append(gs[i] - _up(gs[i + 1], tam))
    lb.append(gb[-1])
    ls.append(gs[-1])

    # mistura banda a banda com a mascara REDUZIDA daquele nivel — e isso que
    # da a largura de transicao proporcional a escala
    saida = None
    for i in range(levels, -1, -1):
        m = gm[i][..., None]
        camada = lb[i] * (1 - m) + ls[i] * m
        if saida is None:
            saida = camada
        else:
            saida = _up(saida, (camada.shape[1], camada.shape[0])) + camada
    return np.clip(saida, 0, 255)


def casar_cor(src, dst, mask, margem=0.25):
    """Aproxima o tom de `src` ao de `dst` medindo os dois num ANEL em volta da
    mascara — regiao onde ambos deveriam ser iguais. Usa mediana e MAD, que
    aguentam contaminacao melhor que media e desvio."""
    nucleo = mask > 0.9
    fora = mask < 0.02
    if nucleo.sum() < 50 or fora.sum() < 50:
        return src
    # anel: perto da borda, dos dois lados
    m8 = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8))
    largo = np.asarray(m8.filter(ImageFilter.GaussianBlur(
        max(4, int(min(mask.shape) * margem * 0.1))))).astype(np.float32) / 255.0
    anel = (largo > 0.05) & (largo < 0.95)
    if anel.sum() < 200:
        anel = fora
    out = src.copy()
    for c in range(3):
        s, d = src[..., c][anel], dst[..., c][anel]
        ms, md = np.median(s), np.median(d)
        ss = np.median(np.abs(s - ms)) + 1e-3
        sd = np.median(np.abs(d - md)) + 1e-3
        g = float(np.clip(sd / ss, 0.92, 1.09))
        out[..., c] = np.clip((src[..., c] - ms) * g + md, 0, 255)
    return out
