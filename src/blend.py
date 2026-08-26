#!/usr/bin/env python3
"""Composicao invisivel de duas fontes com estatisticas diferentes.

O problema: a foto ORIGINAL e crua; a EDITADA passou pelo VAE do modelo de
edicao e por um polimento SDXL. Tom, nitidez e nivel de grao sao diferentes.
Um alpha blend com mascara borrada usa UMA unica largura de transicao para
todas as frequencias, e isso deixa duas assinaturas mensuraveis:

  * o degrau de TOM vira uma rampa cuja derivada e descontinua nas duas pontas;
    qualquer realce (que e um Laplaciano) transforma cada ponta numa LINHA;
  * dentro da rampa, out = a*A + (1-a)*B com A e B tendo graos INDEPENDENTES,
    entao var = s^2*(a^2+(1-a)^2), minimo 0.5*s^2 em a=0.5 — um ANEL 30% mais
    liso que os dois lados. E o "fade esquisito".

A resposta e a piramide Laplaciana (Burt & Adelson, 1983): a banda k e cruzada
com a mascara borrada NO NIVEL k, o que da uma largura de transicao por oitava
(~2^k * 4 px). O grao troca em 3 px (sem faixa mole) e o tom troca em ~2^L*4 px
(sem descontinuidade detectavel).

Tres cuidados que nao sao opcionais:
  * float32 do inicio ao fim. Quantizar a piramide em uint8 (ou usar o
    cv2.detail_MultiBandBlender, que so aceita 8U/16S) suja o grao do rosto —
    justamente o que o pipeline inteiro existe para preservar.
  * dstsize explicito em TODO pyrUp: pyrDown gera ((cols+1)//2,(rows+1)//2),
    entao com dimensao impar o pyrUp cego devolve 1 px a mais.
  * NUNCA menos de 5 niveis: com L=4 a transicao do residual tem ~64 px e o
    degrau de tom fica quase duro — pior que o alpha blend que se quer trocar.
"""
import numpy as np
import cv2

# ---------------------------------------------------------------- piramides

def _gpyr(img, L):
    p = [img]
    for _ in range(L):
        # BORDER_REFLECT_101 (default). pyrDown REJEITA BORDER_CONSTANT, e o
        # reflect evita que uma moldura escura contamine as bandas grossas
        # quando a cabeca encosta na borda do quadro.
        p.append(cv2.pyrDown(p[-1]))
    return p


def _lpyr(img, L):
    g = _gpyr(img, L)
    lp = [g[i] - cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0]))
          for i in range(L)]
    lp.append(g[-1])          # residual passa-baixa = o TOM
    return lp


def _collapse(lp):
    o = lp[-1]
    for i in range(len(lp) - 2, -1, -1):
        o = cv2.pyrUp(o, dstsize=(lp[i].shape[1], lp[i].shape[0])) + lp[i]
    return o


def n_levels(shape):
    """Quantas oitavas. O que precisa ser espalhado e o degrau de TOM em pixels
    ABSOLUTOS (~2^L*4 px), nao uma fracao da mascara — por isso a regra depende
    da resolucao e nao do tamanho da cabeca. Da 5 em 0.4 MP, 6 em 1.5 MP,
    7 em 6 MP."""
    h, w = shape[:2]
    L = int(round(6 + np.log2(np.sqrt(float(h) * w) / 1254.0)))
    teto = int(np.floor(np.log2(max(8, min(h, w))))) - 2
    return int(np.clip(L, 5, max(5, teto)))


# ------------------------------------------------------------------- helpers

def _g(a, s):
    if s <= 0:
        return a
    return cv2.GaussianBlur(a, (0, 0), float(s), borderType=cv2.BORDER_REPLICATE)


def mad_sigma(x, w=None, minimo=64):
    """Desvio robusto (MAD*1.4826). Media/desvio nao servem aqui: um fio de
    cabelo ou a borda de uma gola domina o std e faz a medida mentir."""
    v = x[w > 0.5] if w is not None else x.ravel()
    if v.size < minimo:
        return 0.0
    med = np.median(v)
    return float(np.median(np.abs(v - med)) * 1.4826)


def dog_stack(img, steps):
    """Pilha de diferencas de gaussianas. Reconstrucao exata: soma+base == img."""
    bands, cur = [], img.astype(np.float32)
    for s in steps:
        lp = _g(cur, s)
        bands.append(cur - lp)
        cur = lp
    return bands, cur


def _ruido_correlacionado(shape, sigma, rng):
    n = rng.standard_normal(shape[:2]).astype(np.float32)
    n = _g(n, sigma)
    return n / (n.std() + 1e-6)


# -------------------------------------------------------- (1) TOM (baixa freq)

def casar_tom(src, dst, iters=6, gclamp=(0.85, 1.18), oclamp=25.0, alvo=400):
    """Casa o TOM de `src` ao de `dst` com uma afim GLOBAL por canal
    (ganho + offset), ajustada por IRLS de Huber sobre o quadro subamostrado.

    E o degrau de tom que o realce transforma numa linha: o VAE do modelo de
    edicao aplica uma deriva no quadro INTEIRO (medido: o fundo intocado caiu de
    L=47,4 para L=41,7), entao a cabeca original colada aparece varias unidades
    mais clara que tudo em volta. Nenhum feather esconde isso.

    Por que uma afim GLOBAL e nao um campo de baixa frequencia: o campo puxa a
    cor da roupa NOVA (que fica a poucos pixels do queixo) para dentro do rosto.
    A afim global e monotona e nao tem resolucao espacial — nao pode inventar
    mancha local — e o Huber joga a roupa trocada para fora do ajuste como
    outlier (peso medio medido: 0,91-0,94 de inlier).

    Devolve (src_corrigida, [(ganho, offset, inliers) por canal]).
    """
    s = np.asarray(src, np.float32)
    d = np.asarray(dst, np.float32)
    st = max(1, int(min(s.shape[:2]) / alvo))
    xs = s[::st, ::st].reshape(-1, 3)
    ys = d[::st, ::st].reshape(-1, 3)
    out = s.copy()
    par = []
    for c in range(3):
        x, y = xs[:, c], ys[:, c]
        wt = np.ones_like(x)
        g, off = 1.0, 0.0
        for _ in range(iters):
            A = np.stack([x, np.ones_like(x)], 1) * np.sqrt(wt)[:, None]
            g, off = np.linalg.lstsq(A, y * np.sqrt(wt), rcond=None)[0]
            r = y - (g * x + off)
            sg = max(float(np.median(np.abs(r)) * 1.4826), 1e-3)
            wt = np.clip(1.345 * sg / np.maximum(np.abs(r), 1e-6), 0, 1)
        g = float(np.clip(g, *gclamp))
        off = float(np.clip(off, -oclamp, oclamp))
        par.append((round(g, 4), round(off, 2), round(float(wt.mean()), 2)))
        out[..., c] = s[..., c] * g + off
    return np.clip(out, 0, 255), par


def casar_lf(src, dst, sigma=None, clamp=10.0):
    """Casa a BAIXA frequencia de `src` com a de `dst`, no quadro inteiro.

    Alternativa LOCAL a casar_tom, para quando a deriva varia dentro do quadro
    (luz mista, vinheta). Mais poderosa e mais perigosa: com sigma pequeno ela
    importa a cor da roupa nova para o queixo. Use com clamp apertado.

    Elimina o degrau de tom NA FONTE, antes de compor, preservando 100% do
    detalhe (a correcao e aditiva e passa-baixa, logo tem derivada 1 na banda do
    grao — nao toca em nitidez nem em textura). Com isso o numero de niveis da
    piramide deixa de ser critico: sobra pouco degrau para espalhar.

    sigma : padrao 5% da diagonal (so deriva GLOBAL de tom, nunca recolorizacao
            local). clamp : teto por canal, em LSB — protege contra o caso em
            que o modelo mudou o conteudo de verdade.
    """
    s = src.astype(np.float32)
    d = dst.astype(np.float32)
    if sigma is None:
        sigma = 0.05 * float(np.hypot(*s.shape[:2]))
    delta = np.clip(_g(d, sigma) - _g(s, sigma), -clamp, clamp)
    return np.clip(s + delta, 0, 255), float(np.abs(delta).mean())


# ------------------------------------------------- (2) NITIDEZ / GRAO da fonte

def casar_hf(src, dst, regiao, steps=(0.8, 1.3, 2.6, 5.2),
             gmax=1.5, injetar=True, chroma=0.15, seed=0):
    """Iguala a alta frequencia de `src` a de `dst` DENTRO de `regiao`, banda a
    banda — mas SO PARA CIMA (ganho >= 1).

    Regra dura: nunca reduzir a textura do rosto. Se a fonte tem MAIS grao que
    o destino (caso normal, quando as duas nascem no mesmo tamanho), o ganho e
    1.0 e esta funcao nao faz nada — quem tem de subir e o destino, e isso e
    trabalho do grao.py, depois da composicao, no quadro inteiro.
    Se a fonte tem MENOS (caso de uma original de 640 px ampliada para 1278),
    a banda e amplificada com teto, e o que o teto nao cobriu e SINTETIZADO com
    o espectro daquela banda — amplificar o vazio e inventar o vazio sao coisas
    diferentes, e a segunda e a correta quando nao ha sinal para amplificar.

    Devolve (src_corrigida, ganhos, deficits).
    """
    bo, base = dog_stack(src.astype(np.float32), steps)
    be, _ = dog_stack(dst.astype(np.float32), steps)
    reg = regiao.astype(np.float32)
    ganhos, deficits = [], []
    for k in range(len(bo)):
        so = mad_sigma(bo[k].mean(2), reg)
        se = mad_sigma(be[k].mean(2), reg)
        g = 1.0 if so < 1e-3 else se / so
        gc = float(np.clip(g, 1.0, gmax))
        ganhos.append(round(float(g), 3))
        deficits.append(float(max(0.0, se ** 2 - (gc * so) ** 2) ** 0.5))
        if gc != 1.0:
            bo[k] = bo[k] * gc
    out = base + sum(bo)
    if injetar and max(deficits) > 1e-3:
        rng = np.random.default_rng(seed)
        m = reg[..., None]
        cum = 0.0
        for k, s in enumerate(steps):
            if deficits[k] > 1e-3:
                sig = max(0.4, (cum ** 2 + s ** 2) ** 0.5 * 0.5)
                luma = _ruido_correlacionado(out.shape, sig, rng)
                n = np.repeat(luma[..., None], 3, 2)
                if chroma > 0:
                    c = np.stack([_ruido_correlacionado(out.shape, sig * 1.4, rng)
                                  for _ in range(3)], 2)
                    n = n * (1 - chroma) + c * chroma
                n = n - _g(n, s)          # so a energia DA banda k
                out = out + n * (deficits[k] / (n.mean(2).std() + 1e-6)) * m
            cum = (cum ** 2 + s ** 2) ** 0.5
    return np.clip(out, 0, 255), ganhos, [round(v, 3) for v in deficits]


# ------------------------------------------------------------- (3) COMPOSICAO

def compor(fundo, frente, mask, levels=None, feather=2.0):
    """Piramide Laplaciana. mask: HxW em 0..1 (ou 0..255); 1 = frente.

    feather e o raio da mascara BASE — 2 px, so para matar serrilhado. NAO 22:
    o feather grande so controla a banda mais fina, e e exatamente ele que
    recria a faixa mole de 70 px que se quer eliminar. As bandas grossas ja tem
    a sua propria largura de transicao, vinda da piramide.
    """
    A = np.asarray(fundo, np.float32)
    B = np.asarray(frente, np.float32)
    m = np.asarray(mask, np.float32)
    if m.max() > 1.5:
        m = m / 255.0
    if levels is None:
        levels = n_levels(A.shape)
    if feather > 0:
        m = cv2.GaussianBlur(m, (0, 0), float(feather))
    lA, lB, gm = _lpyr(A, levels), _lpyr(B, levels), _gpyr(m, levels)
    out = [a * (1 - g[..., None]) + b * g[..., None] for a, b, g in zip(lA, lB, gm)]
    return np.clip(_collapse(out), 0, 255)


# ---------------------------------------------------------------- (4) METRICA

def ridge_costura(img_rgb, mask_bin, faixa=(10, 23)):
    """Razao pico/fundo do realce (|Laplaciano| da luminancia passa-baixa) numa
    faixa de +-80 px do contorno. Piso de uma imagem SEM costura: ~1.1-1.2.
    Referencia medida: alpha feather 22 = 2.6 | alpha feather 8 = 11.3 |
    multibanda L=4 = 5.8 | L=5 = 1.7 | L=6 = 1.13."""
    M = (np.asarray(mask_bin) > (127 if np.asarray(mask_bin).max() > 1.5 else 0.5))
    M = M.astype(np.uint8) * 255
    if M.min() == M.max():
        return float("nan")
    sd = (cv2.distanceTransform(M, cv2.DIST_L2, 5)
          - cv2.distanceTransform(255 - M, cv2.DIST_L2, 5))
    g = cv2.cvtColor(np.clip(img_rgb, 0, 255).astype(np.uint8),
                     cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(_g(g, 8.0), cv2.CV_32F, ksize=5))
    p = []
    for r in range(-80, 81, 5):
        sel = (sd >= r) & (sd < r + 5)
        p.append(lap[sel].mean() if sel.any() else np.nan)
    p = np.array(p, np.float32)
    med = np.nanmedian(p)
    if not np.isfinite(med) or med <= 0:
        return float("nan")
    return float(np.nanmax(p[faixa[0]:faixa[1]]) / med)


def dip_costura(out, fundo, frente, alpha, passo=8, alcance=48):
    """A METRICA QUE PEGA O DEFEITO. Para cada faixa de distancia ate a borda,
    compara a energia de alta frequencia da SAIDA com a que as duas fontes
    teriam ali, misturadas na mesma proporcao:

        razao(d) = sigma_HF(saida) / (a*sigma_HF(frente) + (1-a)*sigma_HF(fundo))

    1.0 = a saida tem a textura que devia ter. Abaixo de 1 = a faixa esta mais
    LISA que os dois lados — o anel de variancia perdida do alpha blend, que e
    literalmente o "fade esquisito". Medido no par real (foto 640 + polida
    1278, mascara semantica): alpha feather 22 chega a 0.81 e o dip se espalha
    por +-40 px; multi-banda fica >= 0.91 e so em +-8 px.

    Nao confunda com ridge_costura: aquela mede o realce absoluto na borda e num
    retrato real e dominada por CONTEUDO (cabelo, gola), nao pela emenda.
    """
    a = np.clip(np.asarray(alpha, np.float32), 0, 1)
    M = (a > 0.5).astype(np.uint8)
    sd = (cv2.distanceTransform(1 - M, cv2.DIST_L2, 5)
          - cv2.distanceTransform(M, cv2.DIST_L2, 5))
    hf = lambda x: (np.asarray(x, np.float32) - _g(np.asarray(x, np.float32), 1.2)).mean(2)
    hb, hf_, hx = hf(fundo), hf(frente), hf(out)
    prof = {}
    for d in range(-alcance, alcance + 1, passo):
        sel = (sd >= d) & (sd < d + passo)
        if sel.sum() < 200:
            continue
        w = sel.astype(np.float32)
        am = float(a[sel].mean())
        ref = mad_sigma(hf_, w) * am + mad_sigma(hb, w) * (1 - am)
        prof[d] = round(mad_sigma(hx, w) / max(ref, 1e-6), 3)
    return prof


def perfil_hf(img_rgb, mask_bin, sigma=1.5, passo=6, alcance=60):
    """Energia de alta frequencia por distancia COM SINAL ate a borda
    (negativo = dentro). Plano => sem degrau de grao."""
    M = (np.asarray(mask_bin) > (127 if np.asarray(mask_bin).max() > 1.5 else 0.5))
    M = M.astype(np.uint8)
    sd = (cv2.distanceTransform(1 - M, cv2.DIST_L2, 5)
          - cv2.distanceTransform(M, cv2.DIST_L2, 5))
    a = np.asarray(img_rgb, np.float32)
    hf = (a - _g(a, sigma)).mean(2)
    out = {}
    for r in range(-alcance, alcance + 1, passo):
        sel = ((sd >= r) & (sd < r + passo)).astype(np.float32)
        out[r] = round(mad_sigma(hf, sel), 3)
    return out


# --------------------------------------------- compatibilidade com o antigo
def blend(base, sobre, mask, levels=None):
    """Assinatura antiga (blend.blend). Agora em float32, com feather base."""
    return compor(base, sobre, mask, levels=levels, feather=0.0)


def casar_cor(src, dst, mask, margem=0.25):
    """Assinatura antiga (blend.casar_cor). Agora e o casamento de BAIXA
    frequencia no quadro inteiro, que e estritamente melhor que medir num anel:
    a amostra deixa de ser uma faixa fina contaminada pela roupa nova."""
    return casar_tom(src, dst)[0]
