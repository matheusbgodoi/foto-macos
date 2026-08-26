#!/usr/bin/env python3
"""Mede a fidelidade 1:1 de uma edicao: quanto do que NAO foi pedido mudou.

Alinha os tamanhos, calcula o mapa de diferenca, separa a regiao realmente
editada (top X% de diff) do resto, e reporta PSNR/erro medio no RESTO.
Quanto maior o PSNR-fora, mais a foto ficou identica onde deveria.
"""
import sys
import numpy as np
from PIL import Image


def load(p, size=None):
    im = Image.open(p).convert("RGB")
    if size and im.size != size:
        im = im.resize(size, Image.LANCZOS)
    return np.asarray(im).astype(np.float32)


def report(orig_p, edit_p, label=""):
    o = Image.open(orig_p).convert("RGB")
    e = Image.open(edit_p).convert("RGB")
    if o.size != e.size:
        o = o.resize(e.size, Image.LANCZOS)
    a, b = np.asarray(o).astype(np.float32), np.asarray(e).astype(np.float32)

    diff = np.abs(a - b).mean(axis=2)
    # regiao editada = pixels acima do percentil 88 do erro (heuristica: a edicao
    # costuma ocupar 5-15% do quadro). O resto e o que deveria ter ficado igual.
    thr = np.percentile(diff, 88)
    edited = diff > thr
    outside = ~edited

    def psnr(mask):
        mse = ((a - b) ** 2).mean(axis=2)[mask].mean()
        return 99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse)

    print(f"[{label or edit_p}]")
    print(f"  erro medio global      : {diff.mean():6.2f} /255")
    print(f"  erro medio FORA da edicao: {diff[outside].mean():6.2f} /255   <- quanto menor, melhor")
    print(f"  PSNR FORA da edicao    : {psnr(outside):6.2f} dB          <- quanto maior, melhor")
    print(f"  PSNR dentro da edicao  : {psnr(edited):6.2f} dB          (baixo = a edicao aconteceu)")
    print(f"  pixels identicos (dif<1): {(diff < 1).mean()*100:5.1f}%")
    return diff[outside].mean()


if __name__ == "__main__":
    for pair in zip(sys.argv[1::2], sys.argv[2::2]):
        report(pair[0], pair[1])
