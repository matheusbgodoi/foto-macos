#!/usr/bin/env python3
"""Krea 2 Turbo + Famegrid em MLX, calibrado para Mac com 24 GB.

Usa um snapshot Q4 pronto do MFLUX Community. O modelo continua sujeito a
Krea 2 Community License; a quantizacao nao altera a licenca original.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
import uuid

MFLUX = os.path.expanduser(os.environ.get(
    "MFLUX_KREA2_BIN", "~/.local/bin/mflux-generate-krea2"))
MODEL_REPO = "mflux-community/krea-2-turbo-mflux-q4"


def model_path() -> str:
    override = os.environ.get("KREA2_MODEL")
    if override:
        return os.path.expanduser(override)
    snapshots = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--mflux-community--"
        "krea-2-turbo-mflux-q4/snapshots/*"))
    for snapshot in snapshots:
        index_path = os.path.join(snapshot, "model.safetensors.index.json")
        if not os.path.isfile(index_path):
            continue
        try:
            with open(index_path, encoding="utf-8") as handle:
                files = set(json.load(handle).get("weight_map", {}).values())
            if files and all(os.path.isfile(os.path.join(snapshot, name)) for name in files):
                return snapshot
        except (OSError, ValueError):
            continue
    return ""
LORA = os.path.expanduser(os.environ.get(
    "FAMEGRID_LORA",
    "~/Library/Application Support/foto-macos/loras/krea2/"
    "Famegrid-Natural-V1-Krea-2.safetensors"))
IDENTITIES_FILE = os.path.expanduser(os.environ.get(
    "FOTO_IDENTITIES_FILE",
    "~/Library/Application Support/foto-macos/identities.json"))

STYLE = {
    "natural": (
        "Natural unstaged photograph, physically plausible available light, "
        "real material texture, subtle sensor noise, ordinary imperfections, "
        "realistic dynamic range, no beauty retouching, no CGI gloss"),
    "iphone": (
        "Casual iPhone photograph, computational smartphone exposure, deep "
        "phone-camera depth of field, slight sensor grain, imperfect framing, "
        "available light, unretouched and not cinematic"),
    "profissional": (
        "Professional editorial photograph, controlled but plausible lighting, "
        "real skin and fabric microtexture, optical depth of field, restrained "
        "color grading, photographed rather than rendered"),
}


def identity_registry() -> dict:
    """Carrega o registro privado sem exigir que as LoRAs ja existam."""
    if not os.path.isfile(IDENTITIES_FILE):
        return {}
    try:
        with open(IDENTITIES_FILE, encoding="utf-8") as handle:
            registry = json.load(handle)
    except (OSError, ValueError):
        return {}
    return registry if isinstance(registry, dict) else {}


def identity_matches(prompt: str) -> list[dict]:
    """Resolve nomes publicos para tokens/LoRAs privados configurados localmente."""
    registry = identity_registry()
    matches = []
    for name, config in registry.items():
        if not isinstance(config, dict):
            continue
        if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", prompt,
                     flags=re.IGNORECASE):
            item = dict(config)
            item["name"] = name
            matches.append(item)
    return matches


def apply_identities(prompt: str, matches: list[dict]) -> str:
    for item in matches:
        token = str(item.get("token") or item["name"])
        prompt = re.sub(rf"(?<!\w){re.escape(item['name'])}(?!\w)", token,
                        prompt, flags=re.IGNORECASE)
    return prompt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--saida", required=True)
    parser.add_argument("--tamanho", default="1024x1024")
    parser.add_argument("--seed", type=int, default=int(time.time()) % 1_000_000_000)
    parser.add_argument("--estilo", choices=tuple(STYLE), default="natural")
    parser.add_argument(
        "--peso", type=float, default=None,
        help=("peso da Famegrid; padrao 0.7 sem identidade e 0.3 quando uma "
              "identidade privada e reconhecida"),
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--sem-lora", action="store_true")
    parser.add_argument(
        "--sem-famegrid", action="store_true",
        help="desliga apenas a Famegrid, mantendo LoRAs privadas de identidade",
    )
    args = parser.parse_args()

    if not os.path.isfile(MFLUX):
        print(f"erro: mflux-generate-krea2 ausente: {MFLUX}", file=sys.stderr)
        return 2
    model = model_path()
    if not model:
        print("erro: snapshot Krea 2 Q4 incompleto. Rode: "
              f"hf download {MODEL_REPO}", file=sys.stderr)
        return 2
    if not args.sem_lora and not args.sem_famegrid and not os.path.isfile(LORA):
        print(f"erro: LoRA Famegrid ausente: {LORA}", file=sys.stderr)
        return 2
    width, height = (int(value) for value in args.tamanho.lower().split("x", 1))
    output = os.path.abspath(os.path.expanduser(args.saida))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    stem, extension = os.path.splitext(output)
    temporary = f"{stem}.foto-macos-{uuid.uuid4().hex}{extension or '.png'}"
    user_prompt = args.prompt.strip()
    identities = [] if args.sem_lora else identity_matches(user_prompt)
    for identity in identities:
        identity["lora"] = os.path.abspath(os.path.expanduser(str(identity.get("lora", ""))))
        if not os.path.isfile(identity["lora"]):
            print(f"erro: a identidade {identity['name']!r} foi reconhecida, mas a "
                  f"LoRA ainda nao existe: {identity['lora']}", file=sys.stderr)
            return 2
    user_prompt = apply_identities(user_prompt, identities)
    if args.peso is not None:
        famegrid_scale = args.peso
    elif identities:
        # Famegrid forte melhora o aspecto natural, mas compete com a LoRA de
        # identidade. Nos testes controlados no M5, 0.3 preservou muito mais o
        # rosto que 0.7 sem perder o look fotografico. Cada identidade pode
        # calibrar esse valor no registro privado.
        famegrid_scale = min(
            float(identity.get("famegrid_scale", 0.3))
            for identity in identities
        )
    else:
        famegrid_scale = 0.7
    # Peso zero precisa desativar tambem a trigger textual. Antes, o arquivo da
    # LoRA era aplicado com peso 0 mas "Famegrid" continuava mudando o prompt.
    famegrid_enabled = (
        not args.sem_lora
        and not args.sem_famegrid
        and famegrid_scale > 0
    )
    trigger = ""
    if famegrid_enabled and not user_prompt.lower().startswith("famegrid"):
        trigger = "Famegrid, "
    prompt = f"{trigger}{user_prompt}. {STYLE[args.estilo]}"
    command = [
        MFLUX, "--model", model,
        "--low-ram", "--mlx-cache-limit-gb", "8",
        "--prompt", prompt, "--width", str(width), "--height", str(height),
        "--steps", str(args.steps), "--guidance", str(args.guidance),
        "--seed", str(args.seed), "--output", temporary, "--metadata",
    ]
    if not args.sem_lora:
        if famegrid_enabled:
            command += ["--lora", LORA, str(famegrid_scale)]
        for identity in identities:
            command += ["--lora", identity["lora"], str(identity.get("scale", 0.85))]
        command += ["--no-bake-lora"]
    print(f"[krea2] {width}x{height} steps={args.steps} guidance={args.guidance} "
          f"Famegrid={famegrid_scale if famegrid_enabled else 'off'} identidades="
          f"{','.join(item['name'] for item in identities) or '-'}",
          file=sys.stderr)
    result = subprocess.run(command)
    if result.returncode:
        return result.returncode
    if not os.path.isfile(temporary):
        print(f"erro: MFLUX nao produziu {temporary}", file=sys.stderr)
        return 1
    # O MFLUX cria sufixos _1/_2 quando o destino ja existe. Uma saida unica e
    # a troca atomica impedem o MCP de devolver silenciosamente um PNG antigo.
    os.replace(temporary, output)
    temporary_metadata = os.path.splitext(temporary)[0] + ".metadata.json"
    if os.path.isfile(temporary_metadata):
        os.replace(temporary_metadata, stem + ".metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
