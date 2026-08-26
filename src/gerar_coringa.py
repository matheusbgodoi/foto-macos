#!/usr/bin/env python3
"""Roteador de geracao para Apple Silicon.

Um comando, quatro backends especializados:

* Draw Things + Z-Image i8x: padrao, rapido, fotografia e estilos por prompt;
* MLX + Krea 2 Turbo + Famegrid: teto de fotorrealismo, mais lento;
* ComfyUI + SDXL: somente quando ha LoRA/checkpoint SDXL explicito;
* ComfyUI + FLUX.2 Klein: caminho local alternativo para prompt complexo.

O Draw Things continua instalado como motor. O usuario e os agentes nao
precisam escolher o aplicativo: falam apenas com ``foto gerar``.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.environ.get("FOTO_PYTHON") or os.path.expanduser(
    os.environ.get("COMFYUI_DIR", "~/comfyui") + "/.venv/bin/python")
LOCAL_PHOTO = os.environ.get("LOCAL_PHOTO_BIN", os.path.expanduser(
    "~/.local/node22/bin/local-photo"))
DRAWTHINGS = os.environ.get("DRAWTHINGS_BIN", "/opt/homebrew/bin/draw-things-cli")
DRAW_MODELS = os.path.expanduser(os.environ.get(
    "DRAWTHINGS_MODELS_DIR", "~/Library/Application Support/local-photo-ai-m5/models"))
DRAW_MODEL = "z_image_turbo_1.0_i8x.ckpt"

STYLE_PROMPTS = {
    "cartoon": "clean expressive 2D cartoon illustration, coherent anatomy, bold readable shapes",
    "pixel-art": "authentic 8-bit pixel art, limited color palette, crisp hard pixel edges, no antialiasing",
    "ilustracao": "polished editorial illustration, deliberate shapes, coherent composition",
    "anime": "high quality anime illustration, clean line art, consistent character design",
}

PHOTO_PRESETS = {
    "foto-natural": "natural",
    "iphone": "smartphone",
    "profissional": "professional",
    "produto": "product",
}

PHOTO_PROMPTS = {
    "foto-natural": (
        "ordinary candid photograph, available light, unretouched natural skin, "
        "subtle sensor grain, realistic dynamic range, imperfect real-world framing"),
    "iphone": (
        "casual iPhone photograph, computational smartphone exposure, ordinary "
        "available light, slight sensor noise, deep phone-camera depth of field, "
        "unretouched and not cinematic"),
    "profissional": (
        "professional editorial photograph, soft controlled light that matches the "
        "room, natural skin texture, clean but not glossy commercial photography"),
    "produto": (
        "real product photograph on a physical surface, accurate material texture, "
        "plausible reflections and contact shadow, photographed rather than CGI"),
}


def detect_style(prompt):
    # Uma identidade cadastrada localmente implica Krea 2: o gerador rapido
    # nao sabe carregar a LoRA de identidade e produziria outra pessoa.
    try:
        from krea2 import identity_matches
        if identity_matches(prompt):
            return "famegrid"
    except Exception:
        pass
    text = prompt.lower()
    tests = (
        ("famegrid", ("famegrid", "krea 2", "qualidade maxima",
                      "qualidade máxima", "teto de realismo",
                      "indistinguivel de real", "indistinguível de real")),
        ("pixel-art", ("pixel art", "pixel-art", "8-bit", "8 bit", "pixelado")),
        ("cartoon", ("cartoon", "cartum", "desenho animado")),
        ("anime", ("anime", "manga", "mangá")),
        ("ilustracao", ("illustration", "ilustração", "ilustracao", "desenho 2d")),
        ("iphone", ("iphone", "smartphone", "foto de celular", "phone photo")),
    )
    for style, words in tests:
        if any(word in text for word in words):
            return style
    return "foto-natural"


def run(command):
    process = subprocess.run(command, text=True, capture_output=True)
    if process.stdout:
        print(process.stdout.strip())
    if process.returncode and process.stderr:
        print(process.stderr.strip(), file=sys.stderr)
    return process.returncode


def drawthings_available():
    """O backend rapido so existe quando binario e pesos estao presentes."""
    if os.path.isfile(LOCAL_PHOTO) and os.access(LOCAL_PHOTO, os.X_OK):
        return True
    return (os.path.isfile(DRAWTHINGS) and os.access(DRAWTHINGS, os.X_OK)
            and os.path.isfile(os.path.join(DRAW_MODELS, DRAW_MODEL)))


def drawthings(args, style, output):
    if os.path.isfile(LOCAL_PHOTO) and os.access(LOCAL_PHOTO, os.X_OK):
        command = [LOCAL_PHOTO, "generate", args.prompt, "--output", output,
                   "--seed", str(args.seed), "--quiet", "--json"]
        if style in PHOTO_PRESETS:
            command += ["--preset", PHOTO_PRESETS[style]]
        else:
            prompt = f"{STYLE_PROMPTS.get(style, style)}, {args.prompt}"
            command[2] = prompt
            command += ["--raw"]
            if style == "pixel-art":
                # Lanczos cria cores intermediarias e amolece a grade; pixel
                # art deve sair exatamente na resolucao renderizada.
                command += ["--upscale", "off"]
        if args.tamanho:
            width, height = args.tamanho.lower().split("x", 1)
            command += ["--width", width, "--height", height]
        return run(command)

    # Fallback autocontido para uma instalacao publica sem o projeto privado
    # local-photo-ai-m5. Perde o analisador semantico sofisticado, nao o motor.
    width, height = (int(value) for value in args.tamanho.lower().split("x", 1))
    width, height = round(width / 64) * 64, round(height / 64) * 64
    prefix = PHOTO_PROMPTS.get(style) or STYLE_PROMPTS.get(style, style)
    prompt = f"{prefix}, {args.prompt}"
    command = [DRAWTHINGS, "generate", "--models-dir", DRAW_MODELS,
               "--model", DRAW_MODEL, "--prompt", prompt,
               "--width", str(width), "--height", str(height),
               "--steps", "8", "--cfg", "1", "--seed", str(args.seed),
               "--output", output, "--disable-preview",
               "--no-download-missing", "--offline"]
    return run(command)


def comfy_sdxl(args, output):
    command = [PY, os.path.join(HERE, "gerar.py"), args.prompt,
               "--saida", output, "--tamanho", args.tamanho,
               "--seed", str(args.seed)]
    for lora in args.lora:
        command += ["--lora", lora]
    return run(command)


def comfy_flux2(args, output):
    return run([PY, os.path.join(HERE, "flux2.py"), args.prompt,
                "--saida", output, "--tamanho", args.tamanho,
                "--seed", str(args.seed)])


def mlx_krea2(args, output, style):
    krea_style = "iphone" if style == "iphone" else (
        "profissional" if style in ("profissional", "produto") else "natural")
    return run([PY, os.path.join(HERE, "krea2.py"), args.prompt,
                "--saida", output, "--tamanho", args.tamanho,
                "--seed", str(args.seed), "--estilo", krea_style])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--saida", default=None)
    parser.add_argument("--tamanho", default="1024x1024")
    parser.add_argument("--seed", type=int, default=int(time.time()) % 1_000_000_000)
    parser.add_argument("--estilo", default="auto", choices=(
        "auto", "foto-natural", "iphone", "profissional", "produto",
        "cartoon", "pixel-art", "ilustracao", "anime", "famegrid", "livre"))
    parser.add_argument("--motor", default="auto",
                        choices=("auto", "drawthings", "krea2", "sdxl", "flux2"))
    parser.add_argument("--lora", action="append", default=[])
    args = parser.parse_args()

    style = detect_style(args.prompt) if args.estilo == "auto" else args.estilo
    engine = args.motor
    if engine == "auto":
        if style == "famegrid":
            engine = "krea2"
        elif args.lora:
            engine = "sdxl"
        else:
            # Uma instalacao publica sem Draw Things continua funcional:
            # FLUX.2 e o fallback mantido pelo proprio repo/ComfyUI.
            engine = "drawthings" if drawthings_available() else "flux2"
    extension = ".png"
    output = os.path.abspath(os.path.expanduser(args.saida or os.path.join(
        "~/Downloads", f"foto_{int(time.time())}{extension}")))
    os.makedirs(os.path.dirname(output), exist_ok=True)

    print(f"[roteador] estilo={style} motor={engine} saida={output}", file=sys.stderr)
    if engine == "drawthings":
        if not drawthings_available():
            print("erro: Draw Things CLI/modelo i8x nao estao instalados; "
                  "use --motor flux2 ou deixe --motor auto", file=sys.stderr)
            return 2
        return drawthings(args, style, output)
    if engine == "sdxl":
        return comfy_sdxl(args, output)
    if engine == "krea2":
        return mlx_krea2(args, output, style)
    return comfy_flux2(args, output)


if __name__ == "__main__":
    sys.exit(main())
