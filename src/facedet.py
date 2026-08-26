#!/usr/bin/env python3
"""Deteccao de rosto e landmarks pelo Vision framework NATIVO do macOS.

Por que nao MediaPipe: no macOS o FaceLandmarker do MediaPipe aborta dentro do
face detector (DrishtiMetalHelper -> "Check failed: service_ Service is
unavailable") e o pacote 1.x nem expoe mais o caminho legado mp.solutions.
O Vision ja esta no sistema, roda no Neural Engine e nao pede download.

Devolve, em pixels e com origem no topo-esquerda (Vision usa origem embaixo):
  bbox      : (x, y, w, h)
  pontos    : dict de regiao -> array Nx2
"""
import numpy as np
import Quartz
import Vision
from Foundation import NSURL

REGIONS = ("faceContour", "leftEye", "rightEye", "nose", "noseCrest",
           "medianLine", "outerLips", "innerLips", "leftEyebrow",
           "rightEyebrow", "leftPupil", "rightPupil")


def detect(path):
    url = NSURL.fileURLWithPath_(str(path))
    src = Quartz.CGImageSourceCreateWithURL(url, None)
    if src is None:
        return None
    cg = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    W, H = Quartz.CGImageGetWidth(cg), Quartz.CGImageGetHeight(cg)

    req = Vision.VNDetectFaceLandmarksRequest.alloc().init()
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        return None
    obs = req.results()
    if not obs:
        return None
    # maior rosto do quadro
    face = max(obs, key=lambda o: o.boundingBox().size.width * o.boundingBox().size.height)

    bb = face.boundingBox()
    bx, by = bb.origin.x * W, bb.origin.y * H
    bw, bh = bb.size.width * W, bb.size.height * H
    # Vision: origem embaixo-esquerda -> converte para topo-esquerda
    bbox = (bx, H - by - bh, bw, bh)

    lms = face.landmarks()
    pts = {}
    if lms is not None:
        for name in REGIONS:
            region = getattr(lms, name)()
            if region is None:
                continue
            n = region.pointCount()
            if not n:
                continue
            # normalizedPoints devolve um objc.varlist (ponteiro C sem tamanho):
            # precisa ser fatiado com o pointCount para virar tuplas Python.
            raw = region.normalizedPoints()[:n]
            arr = []
            for p in raw:
                # pontos vem normalizados DENTRO do boundingBox, origem embaixo
                x = bx + p.x * bw
                y = by + p.y * bh
                arr.append((x, H - y))
            pts[name] = np.array(arr, dtype=np.float64)
    return {"bbox": bbox, "points": pts, "size": (W, H)}


def anchors(det):
    """Pontos estaveis para alinhar dois rostos: centro de cada olho, ponta do
    nariz e centro da boca. Evita o contorno externo, que varia com cabelo e
    mandibula, e evita pontos que o modelo generativo costuma deslocar."""
    p = det["points"]
    out = []
    for eye in ("leftPupil", "rightPupil"):
        if eye in p:
            out.append(p[eye].mean(axis=0))
    if len(out) < 2:
        for eye in ("leftEye", "rightEye"):
            if eye in p:
                out.append(p[eye].mean(axis=0))
    if "nose" in p:
        out.append(p["nose"].mean(axis=0))
    if "outerLips" in p:
        out.append(p["outerLips"].mean(axis=0))
    if "medianLine" in p and len(p["medianLine"]) >= 2:
        out.append(p["medianLine"][0])
        out.append(p["medianLine"][-1])
    return np.array(out) if len(out) >= 3 else None


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        d = detect(f)
        if not d:
            print(f, "-> nenhum rosto")
            continue
        a = anchors(d)
        print(f, "-> bbox", tuple(round(v) for v in d["bbox"]),
              "| regioes:", len(d["points"]), "| ancoras:", 0 if a is None else len(a))
