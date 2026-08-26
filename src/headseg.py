#!/usr/bin/env python3
"""Mascara SEMANTICA de cabeca (cabelo + rosto + pescoco) pelo Vision do macOS.

Por que nao a elipse: ela corta pescoco e ombro em linha reta e engole pedaco
de fundo. Depois do blend multi-banda esse fundo ORIGINAL reaparece por cima do
fundo EDITADO, e no realce high-pass a borda aparece como uma linha em volta da
cabeca. Medido: a banda de transicao da elipse ocupa 4,5% do quadro; a
semantica, 1,2%.

Por que nao o SAM3 do ComfyUI: o node devolve mascara BINARIA (hard-coded
`(mask > 0).float()`), custa 2,9-5,6 s a quente e o checkpoint de 1,75 GB
despeja o Mage/SDXL da memoria unificada. O Vision devolve mattes SOFT na
resolucao EXATA da entrada em 30-160 ms, e o de cabelo tem fio solto de verdade.

AVISO: VNGenerateHumanAttributesSegmentationRequest e SPI (nao documentada pela
Apple). Existe e funciona no macOS 26/27. Toda chamada aqui tem fallback: se a
request sumir num update, head_alpha() devolve None e o chamador usa a elipse.
"""
import numpy as np
import cv2
import Quartz
import Vision
from Foundation import NSURL

_ONE8 = 1278226488     # kCVPixelFormatType_OneComponent8
_ONE32F = 1278226534   # kCVPixelFormatType_OneComponent32Float
_ONE16H = 1278226536   # kCVPixelFormatType_OneComponent16Half


def _cgimage(path):
    src = Quartz.CGImageSourceCreateWithURL(NSURL.fileURLWithPath_(str(path)), None)
    if src is None:
        raise IOError(str(path))
    return Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)


def _pb_to_np(pb):
    """CVPixelBuffer de 1 canal -> numpy float32 0..1."""
    Quartz.CVPixelBufferLockBaseAddress(pb, 1)
    try:
        w = Quartz.CVPixelBufferGetWidth(pb)
        h = Quartz.CVPixelBufferGetHeight(pb)
        bpr = Quartz.CVPixelBufferGetBytesPerRow(pb)
        fmt = Quartz.CVPixelBufferGetPixelFormatType(pb)
        buf = Quartz.CVPixelBufferGetBaseAddress(pb).as_buffer(bpr * h)
        if fmt == _ONE8:
            a = np.frombuffer(buf, np.uint8).reshape(h, bpr)[:, :w].astype(np.float32) / 255.0
        elif fmt == _ONE32F:
            a = np.frombuffer(buf, np.float32).reshape(h, bpr // 4)[:, :w]
        elif fmt == _ONE16H:
            a = np.frombuffer(buf, np.float16).reshape(h, bpr // 2)[:, :w].astype(np.float32)
        else:
            raise ValueError("pixel format inesperado: %s" % hex(fmt))
        return np.ascontiguousarray(a, np.float32)
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(pb, 1)


def attribute_mattes(path):
    """[pele, cabelo, dentes, oculos] float32 0..1 no tamanho da entrada.
    Lista vazia se a request nao existir nesta versao do macOS."""
    cls = getattr(Vision, "VNGenerateHumanAttributesSegmentationRequest", None)
    if cls is None:
        return []
    # .init() e NS_UNAVAILABLE nestas requests: tem de ser o completion handler.
    req = cls.alloc().initWithCompletionHandler_(None)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
        _cgimage(path), None)
    ok, _err = handler.performRequests_error_([req], None)
    if not ok:
        return []
    return [_pb_to_np(o.pixelBuffer()) for o in (req.results() or [])]


def _idx_cabelo(ms, bbox):
    """Qual dos mattes e o CABELO. A ordem observada e (pele, cabelo, dentes,
    oculos), mas por ser SPI nao ha contrato: identifica pelo matte com maior
    presenca na faixa ACIMA do bbox do rosto, onde so ha cabelo."""
    if bbox is None or not ms:
        return 1 if len(ms) > 1 else 0
    H, W = ms[0].shape
    x, y, bw, bh = [float(v) for v in bbox]
    y0 = int(max(0, y - 0.6 * bh)); y1 = int(max(1, y + 0.15 * bh))
    x0 = int(max(0, x - 0.2 * bw)); x1 = int(min(W, x + 1.2 * bw))
    if y1 <= y0 or x1 <= x0:
        return 1 if len(ms) > 1 else 0
    return int(np.argmax([float(m[y0:y1, x0:x1].mean()) for m in ms]))


def _envelope(shape, bbox, largura, cima, baixo, suave=0.12):
    """Elipse GENEROSA em volta do rosto, com borda suave, usada so para cortar
    pele distante. O matte de PELE cobre toda a pele exposta e o fechamento
    morfologico chega a colar a MAO no queixo quando ela passa perto do rosto
    (medido numa foto de espelho segurando o celular). O envelope resolve isso
    sem voltar a ser uma mascara eliptica: ele e largo demais para tocar no
    contorno do cabelo, so mata o que esta longe."""
    h, w = shape
    x, y, bw, bh = [float(v) for v in bbox]
    cx, cy = x + bw / 2, y + bh / 2
    rx = bw / 2 * largura
    ru, rd = bh / 2 * cima, bh / 2 * baixo
    cy_e = cy - (ru - rd) / 2
    ry = (ru + rd) / 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy_e) / ry) ** 2)
    return np.clip((1.0 + suave - r) / suave, 0, 1).astype(np.float32)


def head_alpha(path, face_bbox=None, tamanho=None, close=0.030,
               neck=(0.35, 1.30), erode_frac=0.006, feather_frac=0.006,
               envelope=(2.2, 2.6, 2.4)):
    """Alpha 0..1 da CABECA. None quando nao ha pessoa (cai no fallback).

    face_bbox : (x, y, w, h) em px, origem topo-esquerda — de facedet.detect().
                Sem ele, braco/mao/colo entram: o matte de PELE cobre toda a
                pele exposta, nao so a cabeca.
    tamanho   : (w, h) de saida; redimensiona se a imagem de trabalho ja foi
                reescalada. face_bbox deve estar NA ESCALA DA IMAGEM ORIGINAL.
    envelope  : (largura, cima, baixo) em bboxes de rosto — corta pele distante
                (mao, braco, colo). None desliga.
    neck      : rampa smoothstep 1->0 entre queixo+a*bh e queixo+b*bh, para nao
                colar o peito/gola ORIGINAL por cima da roupa nova.
    """
    ms = attribute_mattes(path)
    if not ms or max(float(m.max()) for m in ms) < 0.2:
        return None

    soft = np.maximum.reduce(ms)
    H, W = soft.shape
    s = max(H, W)
    hair = ms[_idx_cabelo(ms, face_bbox)]

    def ker(f):
        k = max(3, int(s * f) | 1)
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # miolo solido: olho, sobrancelha e bigode ficam FORA do matte de pele,
    # entao sem fechar + tapar buracos a mascara sai furada.
    core = (soft > 0.5).astype(np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, ker(close))
    ff = core.copy()
    cv2.floodFill(ff, np.zeros((H + 2, W + 2), np.uint8), (0, 0), 1)
    core |= (1 - ff)

    vizinhanca = None
    if face_bbox is not None:
        x, y, bw, bh = [int(v) for v in face_bbox]
        n, lab = cv2.connectedComponents(core)
        roi = lab[max(0, y):max(1, y + bh), max(0, x):max(1, x + bw)]
        ids, cnt = np.unique(roi[roi > 0], return_counts=True)
        if len(ids):
            core = (lab == ids[int(cnt.argmax())]).astype(np.uint8)
            # o matte SOFT vale so na vizinhanca do componente escolhido: ele
            # cobre TODA a pele exposta, e sem este corte a mao (que o
            # componente conexo ja tinha descartado) volta pelo lado soft.
            vizinhanca = cv2.dilate(core, ker(0.020)).astype(np.float32)

    alpha = np.maximum(soft if vizinhanca is None else soft * vizinhanca,
                       core.astype(np.float32))

    if envelope is not None and face_bbox is not None:
        alpha = alpha * _envelope((H, W), face_bbox, *envelope)

    if neck is not None and face_bbox is not None:
        x, y, bw, bh = [float(v) for v in face_bbox]
        y0, y1 = y + bh + neck[0] * bh, y + bh + neck[1] * bh
        t = np.clip((np.arange(H, dtype=np.float32)[:, None] - y0) / max(1.0, y1 - y0), 0, 1)
        alpha = alpha * (1.0 - (t * t * (3 - 2 * t)))

    # feather LARGO no pescoco/gola (onde a mascara corta carne viva) e ZERO
    # onde o matte de cabelo ja e soft (onde estao os fios).
    ep = max(3, int(s * erode_frac)) | 1
    fe = max(2.0, s * feather_frac)
    er = cv2.erode((alpha > 0.5).astype(np.uint8),
                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ep, ep)))
    seguro = np.minimum(alpha, cv2.GaussianBlur(er.astype(np.float32), (0, 0), fe))
    fio = (hair > 0.02) & (hair < 0.98)
    alpha = np.where(fio, alpha, seguro).astype(np.float32)

    if tamanho is not None and (tamanho[0] != W or tamanho[1] != H):
        alpha = cv2.resize(alpha, (int(tamanho[0]), int(tamanho[1])),
                           interpolation=cv2.INTER_LINEAR)
    return np.clip(alpha, 0, 1)


if __name__ == "__main__":
    import sys, time
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import facedet
    for f in sys.argv[1:]:
        t = time.time()
        d = facedet.detect(f)
        a = head_alpha(f, face_bbox=None if d is None else d["bbox"])
        if a is None:
            print(f, "-> sem pessoa (fallback elipse)")
        else:
            print(f, "->", a.shape, "area %.2f%%" % ((a > 0.5).mean() * 100),
                  "%.0f ms" % ((time.time() - t) * 1000))
