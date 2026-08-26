#!/usr/bin/env python3
"""Recola a CABECA da foto original quando a pose nao mudou.

Quando a edicao e sobre a propria foto (trocar roupa, trocar objeto), a pessoa
nao se move: a cabeca esta no mesmo lugar na entrada e na saida. Nao ha
transformacao para estimar — e composicao direta. Isso preserva rosto E cabelo,
que e onde o modelo mais estraga (achata o cabelo, alisa a pele, muda a
expressao).

O defeito da versao anterior era o alpha blend de uma ELIPSE com feather de
22 px. Duas fontes com estatisticas diferentes (a foto crua e a imagem que
passou por VAE + polimento SDXL) misturadas com UMA unica largura de transicao
deixam duas assinaturas: uma rampa de tom cujas pontas o realce transforma em
linha, e uma faixa 15-20% mais LISA que os dois lados (var = s^2(a^2+(1-a)^2)),
que e o "fade esquisito". Ver blend.py.

Ordem do que este script faz — e a ordem importa:

  1. PARIDADE DE GRADE — por padrao compoe na resolucao NATIVA da original.
     Ampliar a foto de 640 px para 1278 px com LANCZOS destroi ~54% da energia
     da oitava mais fina do rosto antes de qualquer blend. Quando o pipeline
     pede upscale, o SeedVR2 toca somente a imagem editada e esta etapa recebe
     a grade ampliada; a cabeca original entra por reamostragem deterministica.
  2. MASCARA SEMANTICA — contorno real de cabelo+rosto+pescoco pelo Vision
     (headseg.py), com fio de cabelo. A elipse inclui fundo perto das orelhas e
     do ombro, e o multi-banda faz esse fundo ORIGINAL reaparecer sobre o fundo
     EDITADO. Cai para a elipse sozinho se o Vision nao responder.
  3. CASAR TOM da fonte com uma afim global robusta — o degrau de cor tem de
     morrer NA FONTE, nao ser espalhado pelo blend.
  4. CASAR NITIDEZ da fonte, SO quando ela foi ampliada, e SO para cima —
     nunca alisar o rosto para imitar a textura do SDXL.
  5. COMPOR por piramide Laplaciana float32 com feather base de 2 px.

Diferente de face1to1.py: aqui NAO ha warp, e a mascara cobre cabelo e queixo,
nao so o oval do rosto. So use quando a pose for a mesma.
"""
import argparse
import numpy as np
from PIL import Image, ImageDraw

import blend as B
import facedet

try:
    import headseg
except Exception:                                   # pyobjc/Vision indisponivel
    headseg = None


def head_mask_elipse(det, shape, largura=1.9, altura_cima=2.1, altura_baixo=1.35):
    """Fallback: elipse em volta do bbox do rosto, alargada para conter cabelo e
    queixo. So entra quando o Vision nao devolve os mattes semanticos."""
    h, w = shape[:2]
    x, y, bw, bh = det["bbox"]
    cx, cy = x + bw / 2, y + bh / 2
    rx = bw / 2 * largura
    ry_up, ry_dn = bh / 2 * altura_cima, bh / 2 * altura_baixo
    cy_e = cy - (ry_up - ry_dn) / 2
    ry = (ry_up + ry_dn) / 2
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).ellipse([cx - rx, cy_e - ry, cx + rx, cy_e + ry], fill=255)
    return np.asarray(im, np.float32) / 255.0


def mascara(original, det, shape, a):
    """Alpha 0..1 da cabeca na grade de trabalho, com cadeia de fallback.
    O bbox passado ao headseg tem de estar na escala do ARQUIVO original."""
    h, w = shape[:2]
    if a.mascara != "elipse" and headseg is not None:
        try:
            alpha = headseg.head_alpha(original, face_bbox=a.bbox_nativo,
                                       tamanho=(w, h), feather_frac=a.mask_feather)
            if alpha is not None:
                return alpha, "semantica"
            print("[cabeca] Vision nao achou pessoa; usando elipse")
        except Exception as exc:
            print(f"[cabeca] Vision falhou ({exc}); usando elipse")
    return head_mask_elipse(det, shape, a.largura, a.cima, a.baixo), "elipse"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original")
    ap.add_argument("editada")
    ap.add_argument("saida")
    ap.add_argument("--grade", choices=("auto", "original", "editada"), default="auto",
                    help="resolucao da composicao. auto = a menor das duas (quase "
                         "sempre a original), que e onde o rosto e real")
    ap.add_argument("--devolver", action="store_true",
                    help="depois de compor na grade da original, devolve o "
                         "tamanho da editada com LANCZOS. Os dois lados passam "
                         "pelo MESMO reamostrador, entao a costura continua "
                         "invisivel — so nao recupera detalhe")
    ap.add_argument("--mascara", choices=("auto", "elipse"), default="auto")
    ap.add_argument("--feather", type=float, default=2.0,
                    help="raio da mascara BASE. 2 px, so para matar serrilhado. "
                         "22 px era o defeito: recria a faixa mole de 70 px")
    ap.add_argument("--mask-feather", type=float, default=0.003,
                    help="fracao do lado usada no feather interno da mascara "
                         "semantica (pescoco/gola). O multi-banda ja da a "
                         "transicao; aqui so o necessario para nao serrilhar")
    ap.add_argument("--niveis", type=int, default=0, help="0 = automatico. Nunca < 5")
    ap.add_argument("--tom", choices=("editada", "foto", "lf", "nada"), default="foto",
                    help="foto (padrao): o quadro TODO volta ao tom da foto — "
                         "desfaz a deriva do VAE em vez de propaga-la para o "
                         "rosto, e o rosto sai com a textura intacta. "
                         "editada: o inverso, a cabeca adota o tom do quadro "
                         "(conservador: nao mexe no resto da imagem). "
                         "lf: campo local (perigoso perto de roupa nova)")
    ap.add_argument("--nitidez", choices=("auto", "sim", "nao"), default="auto",
                    help="auto = so quando a original teve de ser AMPLIADA")
    ap.add_argument("--ganho-max", type=float, default=1.5)
    ap.add_argument("--largura", type=float, default=1.9)
    ap.add_argument("--cima", type=float, default=2.1)
    ap.add_argument("--baixo", type=float, default=1.35)
    ap.add_argument("--debug-mask", default=None)
    ap.add_argument("--qa", action="store_true", help="mede a costura na saida")
    ap.add_argument("--alfa", action="store_true",
                    help="composicao alfa simples com feather 22 (a antiga); "
                         "so para comparar lado a lado")
    a = ap.parse_args()

    o = Image.open(a.original).convert("RGB")
    e = Image.open(a.editada).convert("RGB")
    tam_editada = e.size

    det = facedet.detect(a.original)
    if det is None:
        raise SystemExit("nenhum rosto detectado na foto original")
    a.bbox_nativo = det["bbox"]

    # ---- 1. paridade de grade
    esc, ampliou = 1.0, False
    if o.size != e.size:
        if a.grade == "editada" or (a.grade == "auto" and o.width > e.width):
            esc = e.width / o.width
            o = o.resize(e.size, Image.LANCZOS)
            ampliou = esc > 1.0
            x, y, bw, bh = det["bbox"]
            det = dict(det, bbox=(x * esc, y * esc, bw * esc, bh * esc))
        else:
            e = e.resize(o.size, Image.LANCZOS)

    O = np.asarray(o, np.float32)
    E = np.asarray(e, np.float32)
    alpha, tipo = mascara(a.original, det, (e.height, e.width), a)
    Ocru = O

    if a.alfa:                                       # caminho antigo, para A/B
        m = B._g(alpha, 22.0)[..., None]
        out = E * (1 - m) + O * m
        info = "alfa feather 22 (modo antigo)"
    else:
        # ---- 3. tom
        nota, par = "", None
        if a.tom == "editada":
            O, par = B.casar_tom(O, E)
            nota = f"tom da cabeca -> quadro {par}"
        elif a.tom == "foto":
            E, par = B.casar_tom(E, O)
            nota = f"tom do quadro -> foto {par}"
        elif a.tom == "lf":
            O, d = B.casar_lf(O, E)
            nota = f"tom local {d:.2f} LSB"
        if par is not None and min(p[2] for p in par) < 0.6:
            # peso medio de inlier baixo = a edicao mudou boa parte do quadro,
            # entao a afim esta sendo ajustada em cima do que MUDOU
            print("[cabeca] AVISO: pouca area em comum entre foto e edicao "
                  f"(inliers {min(p[2] for p in par):.2f}); o casamento de tom "
                  "esta menos confiavel — confira, ou use --tom nada")

        # ---- 4. nitidez (so se a fonte foi ampliada)
        ganhos = defs = None
        if a.nitidez == "sim" or (a.nitidez == "auto" and ampliou):
            nucleo = (B._g((alpha > 0.9).astype(np.float32),
                           max(2.0, 0.01 * min(O.shape[:2]))) > 0.98).astype(np.float32)
            if nucleo.sum() < 500:
                nucleo = (alpha > 0.9).astype(np.float32)
            O, ganhos, defs = B.casar_hf(O, E, nucleo, gmax=a.ganho_max)

        # ---- 5. compor
        L = a.niveis if a.niveis >= 5 else B.n_levels(O.shape)
        out = B.compor(E, O, alpha, levels=L, feather=a.feather)
        info = f"multi-banda L={L}, feather {a.feather:g}px, mascara {tipo}, {nota}"
        if ganhos:
            info += f", ganhos {ganhos}"
            if max(defs) > 0.05:
                info += f", grao sintetico {defs}"

    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    if a.devolver and img.size != tam_editada:
        img = img.resize(tam_editada, Image.LANCZOS)
    img.save(a.saida)
    if a.debug_mask:
        Image.fromarray((np.clip(alpha, 0, 1) * 255).astype(np.uint8)).save(a.debug_mask)
    print(f"[cabeca] {a.saida} {img.width}x{img.height} — cabeca original "
          f"recolada ({(alpha > 0.5).mean()*100:.1f}% do quadro, {info})")
    if a.qa:
        dip = B.dip_costura(out, E, Ocru, alpha)
        pior = min(dip.values()) if dip else float("nan")
        print(f"[cabeca] QA textura na costura: pior razao {pior:.2f} "
              f"(1.0 = perfeito; alfa22 chega a 0.81)  {dip}")


if __name__ == "__main__":
    main()
