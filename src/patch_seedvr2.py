#!/usr/bin/env python3
"""Corrige o SeedVR2 do mflux para o MLX 0.32.x.

O codigo chama mx.repeat(arr, mx.array(counts), axis=...) em 4 pontos — repeticao
com contagem VARIAVEL por elemento. O MLX 0.32.2 so aceita repeats:int, entao a
chamada estoura com:

    TypeError: repeat(): incompatible function arguments.

Este patch injeta um helper que repete fatia a fatia e concatena (mesma semantica
de np.repeat com lista) e troca as 4 chamadas. Ver docs/BUGS.md.
Idempotente: nao faz nada se o helper ja estiver la.
"""
import pathlib
import sys

HELPER = '''

def _repeat_var(arr, counts, axis=0):
    """mx.repeat com contagens VARIAVEIS por elemento.

    Patch local: o MLX 0.32.x so aceita repeats:int. Um upgrade do mflux reverte
    este arquivo — o original esta em attention.py.bak.
    """
    reps = counts.tolist() if hasattr(counts, "tolist") else list(counts)
    parts = []
    for i, c in enumerate(reps):
        c = int(c)
        if c <= 0:
            continue
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(i, i + 1)
        parts.append(mx.repeat(arr[tuple(sl)], c, axis=axis))
    return mx.concatenate(parts, axis=axis)

'''

TROCAS = [
    ("mx.repeat(txt, mx.array(counts), axis=0)", "_repeat_var(txt, counts, axis=0)"),
    ("mx.repeat(txt_shape, mx.array(counts), axis=0)", "_repeat_var(txt_shape, counts, axis=0)"),
    ("mx.repeat(mx.arange(len(counts)), mx.array(counts))", "_repeat_var(mx.arange(len(counts)), counts, axis=0)"),
    ("mx.repeat(mx.arange(len(txt_len)), mx.array(counts))", "_repeat_var(mx.arange(len(txt_len)), counts, axis=0)"),
]


def main(path):
    p = pathlib.Path(path)
    s = p.read_text()
    if "_repeat_var" in s:
        print("patch ja aplicado")
        return 0

    import re
    fins = [m.end() for m in re.finditer(r"^(?:import |from ).*$", s, re.M)]
    if not fins:
        print("nao achei a secao de imports", file=sys.stderr)
        return 1
    s = s[:max(fins)] + "\n" + HELPER + s[max(fins):]

    n = 0
    for velho, novo in TROCAS:
        if velho in s:
            s = s.replace(velho, novo)
            n += 1
    p.write_text(s)
    resto = s.count("mx.array(counts)")
    print(f"patch aplicado: {n} chamadas trocadas, {resto} restantes")
    return 0 if resto == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
