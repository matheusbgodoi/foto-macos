#!/usr/bin/env python3
"""Prepara um dataset privado de identidade a partir de People no Apple Photos.

O script nao altera a biblioteca. Ele consulta metadados via osxphotos, escolhe
retratos diversos onde so existe um rosto detectado, exporta copias e cria
captions locais para treino de LoRA. Fotos e UUIDs ficam apenas no diretorio de
saida informado.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont, ImageOps

# As fontes sao originais locais e confiaveis do Apple Photos. Algumas fotos
# ProRAW/panoramas passam do limite conservador do Pillow antes de serem
# reduzidas para 2048 px; desativar o aviso aqui evita falso positivo sem mudar
# o tamanho final do dataset.
Image.MAX_IMAGE_PIXELS = None


def query(library: str, person: str, since: str, until: str) -> list[dict]:
    command = [
        "uvx", "osxphotos", "query", "--library", library,
        "--person", person, "--from-date", since, "--to-date", until,
        "--not-hidden", "--not-missing", "--only-photos", "--json", "--mute",
    ]
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(process.stderr[-2000:])
    return json.loads(process.stdout)


def target_face(photo: dict, person: str) -> dict | None:
    faces = [face for face in photo.get("face_info", []) if face.get("name") == person]
    if not faces:
        return None
    return max(faces, key=lambda face: (face.get("quality") or 0, face.get("size") or 0))


def face_pixels(face: dict) -> float:
    rect = face.get("face_rect")
    if not rect or len(rect) != 2:
        return 0
    return max(0.0, float(rect[1][0]) - float(rect[0][0]))


def bucket(px: float) -> str:
    if px >= 400:
        return "close"
    if px >= 200:
        return "medium"
    return "wide"


def score(photo: dict, face: dict) -> float:
    px = face_pixels(face)
    quality = float(face.get("quality") or 0)
    year = int(str(photo.get("date", "2021"))[:4])
    recent = max(0, year - 2021) * 0.08
    favorite = 0.35 if photo.get("favorite") else 0
    frontal = max(0.0, 0.3 - abs(float(face.get("yaw") or 0)))
    return quality * 4 + min(px / 400, 2.0) + recent + favorite + frontal


def choose(photos: list[dict], person: str, count: int, rejected: set[str]) -> list[dict]:
    candidates = []
    for photo in photos:
        if photo.get("uuid") in rejected:
            continue
        face = target_face(photo, person)
        if not face:
            continue
        # Evita ensinar outra identidade. O nome em Persons e a contagem de
        # faces precisam apontar somente para o sujeito escolhido.
        if photo.get("persons") != [person] or len(photo.get("face_info", [])) != 1:
            continue
        px = face_pixels(face)
        if px < 100 or min(photo.get("width") or 0, photo.get("height") or 0) < 720:
            continue
        item = dict(photo)
        item["_target_face"] = face
        item["_face_pixels"] = px
        item["_bucket"] = bucket(px)
        item["_score"] = score(photo, face)
        candidates.append(item)

    quotas = {
        "close": round(count * 0.42),
        "medium": round(count * 0.38),
    }
    quotas["wide"] = count - quotas["close"] - quotas["medium"]
    selected: list[dict] = []
    used_days: dict[str, int] = {}
    used_months: dict[str, int] = {}

    for kind in ("close", "medium", "wide"):
        pool = sorted(
            (item for item in candidates if item["_bucket"] == kind),
            key=lambda item: item["_score"], reverse=True,
        )
        for item in pool:
            date = str(item.get("date", ""))[:10]
            month = date[:7]
            if used_days.get(date, 0) >= 2 or used_months.get(month, 0) >= 5:
                continue
            selected.append(item)
            used_days[date] = used_days.get(date, 0) + 1
            used_months[month] = used_months.get(month, 0) + 1
            if sum(x["_bucket"] == kind for x in selected) >= quotas[kind]:
                break

    # Se a diversidade temporal impediu preencher uma cota, completa com os
    # melhores restantes, ainda mantendo um unico rosto.
    if len(selected) < count:
        chosen = {item["uuid"] for item in selected}
        rest = sorted((item for item in candidates if item["uuid"] not in chosen),
                      key=lambda item: item["_score"], reverse=True)
        selected.extend(rest[: count - len(selected)])
    chosen = {item["uuid"] for item in selected}
    extras = sorted((item for item in candidates if item["uuid"] not in chosen),
                    key=lambda item: item["_score"], reverse=True)
    # Exporta uma pequena reserva. Duplicatas perceptuais so podem ser
    # detectadas depois que os pixels forem copiados da biblioteca.
    return (selected[:count] + extras)[: count + 24]


def export(library: str, uuids: Path, raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    command = [
        "uvx", "osxphotos", "export", str(raw), "--library", library,
        "--uuid-from-file", str(uuids), "--filename", "{uuid}",
        "--skip-live", "--skip-raw", "--skip-bursts", "--skip-original-if-edited",
        "--edited-suffix", "", "--convert-to-jpeg", "--jpeg-quality", "0.98",
        "--not-hidden", "--not-missing", "--only-photos", "--touch-file",
        "--update",
    ]
    subprocess.run(command, check=True)


def caption(item: dict, trigger: str) -> str:
    face = item["_target_face"]
    framing = {
        "close": "close-up portrait",
        "medium": "medium shot",
        "wide": "full-body or environmental photograph",
    }[item["_bucket"]]
    details = []
    if face.get("has_smile"):
        details.append("natural smile")
    if int(face.get("glasses_type") or 0) not in (0, 3):
        details.append("wearing glasses")
    suffix = ", " + ", ".join(details) if details else ""
    return (f"A real candid photograph of {trigger}, an adult man, {framing}{suffix}, "
            "natural skin texture, real hair, ordinary lighting and a believable environment.")


def difference_hash(image: Image.Image) -> int:
    small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(small.get_flattened_data())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return value


def normalize(selected: list[dict], raw: Path, data: Path, trigger: str,
              count: int) -> tuple[list[Path], list[dict]]:
    data.mkdir(parents=True, exist_ok=True)
    for old in data.glob("*.*"):
        old.unlink()
    output: list[Path] = []
    kept: list[dict] = []
    hashes: list[int] = []
    for item in selected:
        matches = sorted(raw.glob(item["uuid"] + ".*"))
        matches = [path for path in matches if path.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if not matches:
            continue
        with Image.open(matches[0]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            digest = difference_hash(image)
            if any((digest ^ previous).bit_count() <= 5 for previous in hashes):
                continue
            # Evita arquivos gigantes sem cortar a composicao. O trainer ainda
            # faz seu proprio bucket/max_resolution.
            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            index = len(output) + 1
            path = data / f"{index:03d}.jpg"
            image.save(path, quality=96, subsampling=0, optimize=True)
        path.with_suffix(".txt").write_text(caption(item, trigger) + "\n", encoding="utf-8")
        hashes.append(digest)
        output.append(path)
        kept.append(item)
        if len(output) >= count:
            break
    return output, kept


def contact_sheet(images: list[Path], selected: list[dict], output: Path) -> None:
    thumb_w, thumb_h, label_h = 220, 220, 34
    columns = 6
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), "#171717")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=15)
    for index, path in enumerate(images):
        with Image.open(path) as source:
            thumb = ImageOps.fit(source.convert("RGB"), (thumb_w, thumb_h),
                                 method=Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_w
        y = (index // columns) * (thumb_h + label_h)
        sheet.paste(thumb, (x, y))
        item = selected[index]
        label = f"{index + 1:02d}  {str(item.get('date'))[:10]}  {item['_bucket']}"
        draw.text((x + 7, y + thumb_h + 7), label, fill="white", font=font)
    sheet.save(output, quality=94)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", default="~/Pictures/Photos Library.photoslibrary")
    parser.add_argument("--person", required=True)
    parser.add_argument("--trigger", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--since", default="2021-01-01")
    parser.add_argument("--until", default="2027-01-01")
    parser.add_argument("--reject-uuid-file", default="")
    args = parser.parse_args()

    library = os.path.abspath(os.path.expanduser(args.library))
    root = Path(os.path.abspath(os.path.expanduser(args.output)))
    root.mkdir(parents=True, exist_ok=True)
    photos = query(library, args.person, args.since, args.until)
    rejected: set[str] = set()
    if args.reject_uuid_file:
        reject_path = Path(os.path.abspath(os.path.expanduser(args.reject_uuid_file)))
        rejected = {line.strip() for line in reject_path.read_text().splitlines()
                    if line.strip() and not line.startswith("#")}
    selected = choose(photos, args.person, args.count, rejected)
    if len(selected) < max(12, args.count // 2):
        print(f"erro: somente {len(selected)} candidatos seguros", file=sys.stderr)
        return 2

    uuids = root / "selected-uuids.txt"
    uuids.write_text("".join(item["uuid"] + "\n" for item in selected), encoding="utf-8")

    raw = root / "raw"
    export(library, uuids, raw)
    images, kept = normalize(selected, raw, root / "data", args.trigger, args.count)
    manifest = root / "manifest.json"
    safe_manifest = [{
        "index": index, "uuid": item["uuid"], "date": item.get("date"),
        "width": item.get("width"), "height": item.get("height"),
        "bucket": item["_bucket"], "face_pixels": item["_face_pixels"],
        "score": item["_score"],
    } for index, item in enumerate(kept, 1)]
    manifest.write_text(json.dumps(safe_manifest, indent=2) + "\n", encoding="utf-8")
    contact_sheet(images, kept, root / "contact-sheet.jpg")
    print(json.dumps({
        "photos_queried": len(photos), "selected": len(images),
        "dataset": str(root / "data"), "contact_sheet": str(root / "contact-sheet.jpg"),
        "manifest": str(manifest),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
