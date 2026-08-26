#!/usr/bin/env python3
"""Cria uma folha de aprovacao centrada nos rostos de um dataset privado.

Execute com o Python do ComfyUI, que possui o bridge do Vision.framework. O
script nao altera as imagens de treino; apenas gera thumbnails para revisao.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from facedet import detect


def face_crop(image: Image.Image, bbox: tuple[float, float, float, float]) -> Image.Image:
    x, y, width, height = bbox
    # Inclui cabelo, mandibula e parte dos ombros, mas evita que o contexto
    # esconda filtros/oclusoes que precisam ser vistos na aprovacao.
    side = max(width, height) * 3.1
    center_x = x + width / 2
    center_y = y + height * 0.55
    left = max(0, min(image.width - side, center_x - side / 2))
    top = max(0, min(image.height - side, center_y - side * 0.42))
    right = min(image.width, left + side)
    bottom = min(image.height, top + side)
    return image.crop((round(left), round(top), round(right), round(bottom)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--columns", type=int, default=6)
    args = parser.parse_args()

    root = Path(args.dataset).expanduser().resolve()
    paths = sorted(path for path in root.iterdir()
                   if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    if not paths:
        raise SystemExit(f"nenhuma imagem em {root}")

    cell, label_height = 260, 38
    rows = math.ceil(len(paths) / args.columns)
    sheet = Image.new("RGB", (args.columns * cell, rows * (cell + label_height)), "#111111")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=17)

    for position, path in enumerate(paths):
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        result = detect(path)
        crop = face_crop(image, result["bbox"]) if result else image
        thumb = ImageOps.fit(crop, (cell, cell), method=Image.Resampling.LANCZOS)
        x = (position % args.columns) * cell
        y = (position // args.columns) * (cell + label_height)
        sheet.paste(thumb, (x, y))
        label = f"{position + 1:02d}  {path.stem}" + ("" if result else "  SEM ROSTO")
        draw.text((x + 8, y + cell + 8), label, fill="white", font=font)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=95)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
