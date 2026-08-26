#!/usr/bin/env python3
"""Harmoniza a foto ORIGINAL com a EDITADA antes de recolar a cabeca.

O anel oval que aparece num high-pass da imagem final nao e um defeito do
"fade": e a assinatura de duas fontes com estatisticas diferentes coladas por
media ponderada. Tres discrepancias, medidas no par real do pipeline
(WhatsApp 640px -> Mage -> polir SDXL -> 1278px), num patch de bochecha:

    banda (DoG)        b0(~1px)  b1     b2     b3
    ORIGINAL nativa      5.21    6.13   8.89   8.37
    ORIGINAL 2x lanczos  2.39    3.74   5.92   8.70   <- o que cabeca.py cola
    EDITADA (1278)       3.60    4.65   6.59   9.21   <- o que esta em volta

  cor (Lab, bochecha):  ORIGup L=34.98 a=+5.34 b=+4.11
                        EDIT   L=31.72 a=+3.60 b=+5.14   (dL = -3.3)

Ou seja: a cabeca colada e 1.5x menos nitida na banda mais fina e 3.3 unidades
de L mais clara que a pele ao redor. Nenhum feather conserta isso.

Este modulo corrige a ORIGINAL *antes* da composicao:
  1. COR      campo de baixa frequencia ajustado num ANEL em volta da mascara,
              com estatistica robusta (mediana + Huber) e extrapolacao suave
              push-pull. So mexe em frequencias baixas => nao toca no grao.
  2. NITIDEZ  ganho por banda de oitava, medido com MAD colocada pixel a pixel
              (as duas imagens tem a MESMA cena na regiao da cabeca), clipado.
  3. GRAO     o deficit que o clip de nitidez nao pode cobrir vira ruido
              sintetico (amplificar banda vazia de um lanczos so amplifica
              ringing e bloco de JPEG, nao detalhe).
E compoe com blend que PRESERVA VARIANCIA na alta frequencia (ver blend_hf).
"""
import numpy as np
import cv2

# ---------------------------------------------------------------- utilitarios

def _g(a, s):
    if s <= 0:
        return a
    return cv2.GaussianBlur(a, (0, 0), s, borderType=cv2.BORDER_REPLICATE)


def mad_sigma(x, w=None):
    """Desvio robusto (MAD*1.4826). Imune a fio de cabelo / borda de gola, que
    dominam o std e fazem a medida de nitidez mentir."""
    v = x[w > 0.5] if w is not None else x.ravel()
    if v.size < 32:
        return 0.0
    med = np.median(v)
    return float(np.median(np.abs(v - med)) * 1.4826)


def dog_stack(img, steps=(0.8, 1.3, 2.6, 5.2)):
    """Pilha DoG a resolucao plena. reconstrucao EXATA: sum(bands)+base == img."""
    bands, cur = [], img.astype(np.float32)
    for s in steps:
        lp = _g(cur, s)
        bands.append(cur - lp)
        cur = lp
    return bands, cur


def push_pull(values, weights, levels=6):
    """Convolucao normalizada multiescala: espalha 'values' (validos onde
    weights>0) por todo o quadro de forma suave. Substituto barato de um solve
    de Laplace; e o que permite extrapolar a correcao de cor medida no ANEL
    para dentro da cabeca sem criar degrau."""
    v = values.astype(np.float32) * weights[..., None]
    w = weights.astype(np.float32)
    vs, ws = [v], [w]
    for _ in range(levels):
        if min(vs[-1].shape[:2]) < 4:
            break
        vs.append(cv2.pyrDown(vs[-1]))
        ws.append(cv2.pyrDown(ws[-1]))
    out = vs[-1] / np.maximum(ws[-1], 1e-6)[..., None]
    for i in range(len(vs) - 2, -1, -1):
        up = cv2.resize(out, (vs[i].shape[1], vs[i].shape[0]), interpolation=cv2.INTER_LINEAR)
        conf = np.clip(ws[i], 0.0, 1.0)[..., None]
        cur = vs[i] / np.maximum(ws[i], 1e-6)[..., None]
        out = cur * conf + up * (1.0 - conf)
    return out


def ring_weights(mask, inner=0.55, outer=0.98, feather_px=24):
    """Peso de amostragem: um ANEL centrado na borda da mascara.

    Usa distancia com sinal, entao funciona pra qualquer formato (elipse, SAM3,
    matte de cabelo). Largura do anel ~ 2x o feather: e exatamente a faixa onde
    as duas imagens vao se misturar, logo e onde as estatisticas TEM que bater.
    """
    m = (mask > 0.5).astype(np.uint8)
    din = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    sd = dout - din                      # >0 fora, <0 dentro
    r = float(feather_px) * 2.0
    w = np.exp(-0.5 * (sd / (r * 0.6)) ** 2).astype(np.float32)
    w[np.abs(sd) > r * 1.8] = 0.0
    return w


def change_guard(O, E, w, sigma=2.0, k=2.5):
    """Zera o peso onde o modelo REALMENTE mudou o conteudo (gola nova, fundo
    reescrito). Sem isso a amostra do anel fica contaminada e a correcao de cor
    deriva — foi o que mediu +5.4 de a* no pescoco, que e a gola, nao a pele."""
    d = np.abs(_g(O, sigma) - _g(E, sigma)).mean(2)
    s = mad_sigma(d, w)
    if s <= 1e-6:
        return w
    keep = (d < np.median(d[w > 0.5]) + k * s).astype(np.float32)
    return w * keep


# ------------------------------------------------------------------- (1) COR

def match_color_lf(O, E, mask, feather_px=24, order=1, sigma_lf=None,
                   clamp=(6.0, 4.0, 4.0), min_inliers=0.25):
    """Casa a COR da original com a editada usando so baixa frequencia.

    O -> corrigida.  O e E em RGB float32 0..255, mesmo tamanho.
    order: 0 = offset constante (mais seguro), 1 = plano, 2 = quadratica.
    Retorna (O_corrigida, info).

    Por que nao Reinhard (media+desvio) nem histogram matching:
      * o std do anel e dominado por ESTRUTURA (cabelo, borda de gola), nao por
        tom; casar std aplica um ganho de contraste no rosto inteiro por causa
        de uma borda. Aqui casa-se so a media, de forma espacialmente variavel.
      * histogram matching estima uma CDF: com anel fino (poucos milhares de
        px) a CDF e ruidosa, o LUT resultante e nao-linear e ALTERA o sigma do
        grao — justamente o que se quer preservar.
      * a correcao vive num campo passa-baixa; por construcao nao pode mexer em
        nitidez nem em grao. Essa e a propriedade que torna a ordem
        cor -> nitidez -> grao bem definida.
    """
    h, w = O.shape[:2]
    if sigma_lf is None:
        sigma_lf = max(8.0, feather_px * 1.2)

    wr = ring_weights(mask, feather_px=feather_px)
    wr = change_guard(O, E, wr)

    lo = cv2.cvtColor(np.clip(_g(O, sigma_lf), 0, 255) / 255.0, cv2.COLOR_RGB2LAB)
    le = cv2.cvtColor(np.clip(_g(E, sigma_lf), 0, 255) / 255.0, cv2.COLOR_RGB2LAB)
    d = (le - lo).astype(np.float32)          # o quanto a original precisa andar

    sel = wr > 0.05
    n = int(sel.sum())
    info = {"ring_px": n, "inliers": 0.0, "delta": (0.0, 0.0, 0.0)}
    if n < 500:
        return O, info

    # rejeicao robusta por canal (mediana + MAD), depois IRLS de Huber
    keep = np.ones_like(wr, dtype=bool)
    for c in range(3):
        v = d[..., c]
        med = np.median(v[sel])
        s = max(mad_sigma(v, wr), 1e-3)
        keep &= np.abs(v - med) < 3.0 * s
    wk = wr * keep
    frac = float((wk > 0.05).sum()) / max(n, 1)
    info["inliers"] = frac
    if frac < min_inliers:
        order = 0                              # amostra suja: cai pro mais seguro

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx = (xx / w) * 2 - 1
    yy = (yy / h) * 2 - 1
    if order == 0:
        basis = [np.ones_like(xx)]
    elif order == 1:
        basis = [np.ones_like(xx), xx, yy]
    else:
        basis = [np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy]
    A = np.stack([b[wk > 0.05] for b in basis], 1)
    sw = np.sqrt(wk[wk > 0.05])[:, None]
    field = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        y = d[..., c][wk > 0.05]
        coef = np.linalg.lstsq(A * sw, y * sw[:, 0], rcond=None)[0]
        for _ in range(3):                     # IRLS Huber
            r = y - A @ coef
            s = max(np.median(np.abs(r)) * 1.4826, 1e-3)
            hw = np.clip(1.345 * s / np.maximum(np.abs(r), 1e-6), 0, 1)
            ww = np.sqrt(wk[wk > 0.05] * hw)[:, None]
            coef = np.linalg.lstsq(A * ww, y * ww[:, 0], rcond=None)[0]
        field[..., c] = sum(co * b for co, b in zip(coef, basis))
        field[..., c] = np.clip(field[..., c], -clamp[c], clamp[c])

    lab_o = cv2.cvtColor(np.clip(O, 0, 255) / 255.0, cv2.COLOR_RGB2LAB)
    out = cv2.cvtColor(lab_o + field, cv2.COLOR_LAB2RGB) * 255.0
    info["delta"] = tuple(float(field[..., c][mask > 0.5].mean()) for c in range(3))
    return np.clip(out, 0, 255).astype(np.float32), info


def match_color_lf_field(O, E, mask, feather_px=24, sigma_lf=None, clamp=(6., 4., 4.)):
    """Variante NAO-parametrica: mede a diferenca LF pixel a pixel no anel e
    extrapola pra dentro com push-pull. Segue gradiente de luz real (janela de
    um lado), coisa que um plano de ordem 1 so aproxima. Use quando o anel for
    limpo e grande; a versao parametrica e mais segura quando ele e sujo."""
    if sigma_lf is None:
        sigma_lf = max(8.0, feather_px * 1.2)
    wr = change_guard(O, E, ring_weights(mask, feather_px=feather_px))
    lo = cv2.cvtColor(np.clip(_g(O, sigma_lf), 0, 255) / 255.0, cv2.COLOR_RGB2LAB)
    le = cv2.cvtColor(np.clip(_g(E, sigma_lf), 0, 255) / 255.0, cv2.COLOR_RGB2LAB)
    d = (le - lo).astype(np.float32)
    for c in range(3):
        s = max(mad_sigma(d[..., c], wr), 1e-3)
        med = np.median(d[..., c][wr > 0.05]) if (wr > 0.05).any() else 0.0
        wr = wr * (np.abs(d[..., c] - med) < 3.0 * s)
    field = push_pull(d, wr)
    field = _g(field, sigma_lf)
    for c in range(3):
        field[..., c] = np.clip(field[..., c], -clamp[c], clamp[c])
    lab_o = cv2.cvtColor(np.clip(O, 0, 255) / 255.0, cv2.COLOR_RGB2LAB)
    return np.clip(cv2.cvtColor(lab_o + field, cv2.COLOR_LAB2RGB) * 255.0, 0, 255).astype(np.float32)


# --------------------------------------------------------------- (2) NITIDEZ

def band_report(O, E, region, steps=(0.8, 1.3, 2.6, 5.2)):
    """sigma robusto por banda dentro de 'region', nas duas imagens."""
    bo, _ = dog_stack(O.mean(2), steps)
    be, _ = dog_stack(E.mean(2), steps)
    return ([mad_sigma(b, region) for b in bo],
            [mad_sigma(b, region) for b in be])


def match_sharpness(O, E, region, steps=(0.8, 1.3, 2.6, 5.2),
                    gmin=0.6, gmax=1.5):
    """Iguala a resposta de alta frequencia da ORIGINAL a da EDITADA, banda a
    banda, dentro de 'region'. Devolve (O_corrigida, ganhos, deficit_por_banda).

    Medida: as duas imagens tem a MESMA cena na regiao da cabeca (a pose nao
    mudou), entao da pra comparar pixel-colocado, sem escolher patch. Usa MAD
    pra que cabelo e cilios nao dominem.

    Clip: ganho > ~1.5 numa banda que nasceu de um lanczos 2x nao restaura
    detalhe, amplifica ringing e bloco de JPEG. O que sobra do clip sai como
    'deficit' e vai virar RUIDO SINTETICO no estagio 3 — sintetizar e correto
    porque nao existe sinal ali pra amplificar.
    """
    bo, base = dog_stack(O.astype(np.float32), steps)
    gains, deficit = [], []
    for k, b in enumerate(bo):
        so = mad_sigma(b.mean(2), region)
        se = mad_sigma(dog_stack(E.astype(np.float32), steps)[0][k].mean(2), region)
        if so < 1e-3:
            g = 1.0
        else:
            g = se / so
        gc = float(np.clip(g, gmin, gmax))
        gains.append((round(float(g), 3), gc))
        # variancia que falta depois do clip (em unidades de sigma)
        deficit.append(float(max(0.0, se ** 2 - (gc * so) ** 2) ** 0.5))
        bo[k] = b * gc
    out = base + sum(bo)
    return np.clip(out, 0, 255).astype(np.float32), gains, deficit


def match_sharpness_fast(O, E, region, steps=(0.8, 1.3, 2.6, 5.2), gmin=0.6, gmax=1.5):
    """Igual, mas decompoe E uma vez so (a versao acima recomputa por banda)."""
    bo, base = dog_stack(O.astype(np.float32), steps)
    be, _ = dog_stack(E.astype(np.float32), steps)
    gains, deficit = [], []
    for k in range(len(bo)):
        so = mad_sigma(bo[k].mean(2), region)
        se = mad_sigma(be[k].mean(2), region)
        g = 1.0 if so < 1e-3 else se / so
        gc = float(np.clip(g, gmin, gmax))
        gains.append((round(float(g), 3), round(gc, 3)))
        deficit.append(float(max(0.0, se ** 2 - (gc * so) ** 2) ** 0.5))
        bo[k] = bo[k] * gc
    return np.clip(base + sum(bo), 0, 255).astype(np.float32), gains, deficit


# ------------------------------------------------------------------ (3) GRAO

def _correlated_noise(shape, sigma_px, rng):
    n = rng.standard_normal(shape[:2]).astype(np.float32)
    if sigma_px > 0:
        n = _g(n, sigma_px)
    n /= (n.std() + 1e-6)
    return n


def inject_band_noise(img, region, deficit, steps=(0.8, 1.3, 2.6, 5.2),
                      chroma=0.15, seed=0):
    """Injeta ruido com o espectro certo: uma banda de cada vez, com o sigma
    que faltou apos o clip de nitidez. Diferente de somar ruido branco, isto
    reproduz o formato do espectro do grao alvo."""
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32).copy()
    reg = region[..., None] if region.ndim == 2 else region
    cum = 0.0
    for k, s in enumerate(steps):
        if deficit[k] <= 1e-3:
            cum += s
            continue
        cum_sig = max(0.4, (cum ** 2 + s ** 2) ** 0.5 * 0.5)
        luma = _correlated_noise(img.shape, cum_sig, rng)
        n = np.repeat(luma[..., None], 3, 2)
        if chroma > 0:
            c = np.stack([_correlated_noise(img.shape, cum_sig * 1.4, rng) for _ in range(3)], 2)
            n = n * (1 - chroma) + c * chroma
            n /= (n.std() + 1e-6)
        # mantem so a energia DENTRO da banda k
        n = n - _g(n, s)
        sn = n.mean(2).std() + 1e-6
        out += n * (deficit[k] / sn) * reg
        cum += s
    return np.clip(out, 0, 255).astype(np.float32)


# --------------------------------------------------------------- (4) COMPOSE

def blend_hf(hf_o, hf_e, a, rho=0.0):
    """Blend que PRESERVA VARIANCIA na alta frequencia.

    Um alpha-blend comum entre dois campos de ruido INDEPENDENTES de mesmo
    sigma da var = sigma^2 (a^2 + (1-a)^2), que cai a 0.5 (sigma*0.707) em
    a=0.5. Resultado: uma faixa de ~2x feather em volta da cabeca com 30% menos
    grao que os dois lados — isso e literalmente o anel que aparece no
    high-pass x6. Dividir pelo fator restaura o sigma.
    """
    denom = np.sqrt(np.maximum(a * a + (1 - a) * (1 - a) + 2 * a * (1 - a) * rho, 1e-6))
    return (hf_o * a + hf_e * (1 - a)) / denom


def composite(O, E, mask_soft, hf_sigma=1.4, preserve_var=True, rho=0.15):
    """Compoe separando LF (alpha normal) de HF (alpha preservando variancia)."""
    a = mask_soft[..., None].astype(np.float32)
    lo, le = _g(O, hf_sigma), _g(E, hf_sigma)
    ho, he = O - lo, E - le
    lf = lo * a + le * (1 - a)
    hf = blend_hf(ho, he, a, rho) if preserve_var else (ho * a + he * (1 - a))
    return np.clip(lf + hf, 0, 255).astype(np.float32)


# --------------------------------------------------------------- diagnostico

def seam_profile(img, mask, hf_sigma=1.2, bins=None):
    """Perfil de energia de alta frequencia em funcao da distancia com sinal ate
    a borda da mascara. Se houver anel, aparece como um vale (ou pico) em d~0.
    E a metrica que prova se o conserto funcionou."""
    m = (mask > 0.5).astype(np.uint8)
    sd = (cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
          - cv2.distanceTransform(m, cv2.DIST_L2, 5))
    hf = (img.astype(np.float32) - _g(img.astype(np.float32), hf_sigma)).mean(2)
    if bins is None:
        bins = np.arange(-70, 71, 10)
    prof = []
    for i in range(len(bins) - 1):
        s = (sd >= bins[i]) & (sd < bins[i + 1])
        prof.append((int(bins[i]), round(mad_sigma(hf, s.astype(np.float32)), 3), int(s.sum())))
    return prof
