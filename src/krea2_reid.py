#!/usr/bin/env python3
"""Krea 2 ReID: runner experimental para CUDA, não para o Apple MPS.

Segue os parâmetros oficiais do adapter ``yijunwang2/krea2-reid``: uma única
referência preparada, Krea 2 Turbo INT8 ConvRot, LoRA 1.0, KV cache ligado,
8 passos, CFG 1, Euler/simple. O fluxo é diferente da LoRA pessoal: aceita uma
pessoa arbitrária sem treino, mas não deve ser usado para edição pixel a pixel.
O INT8 ConvRot usa ``aten::_int_mm``: o operador não existe no MPS e o fallback
para CPU excedeu 33 GB de swap antes do primeiro passo no M5 de 24 GB.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
import uuid

COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
COMFY_DIR = os.path.expanduser(os.environ.get("COMFYUI_DIR", "~/comfyui"))
INPUT_DIR = os.path.join(COMFY_DIR, "input")
OUTPUT_DIR = os.path.join(COMFY_DIR, "output")

MODEL = "krea2_turbo_int8_convrot.safetensors"
CLIP = "qwen3vl_4b_bf16.safetensors"
VAE = "qwen_image_vae.safetensors"
REID = "krea2_reid_rank32.safetensors"


def post(path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        COMFY + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(request, timeout=120))


def get(path: str) -> dict:
    return json.load(urllib.request.urlopen(COMFY + path, timeout=120))


def stage(path: str) -> str:
    source = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(source):
        raise FileNotFoundError(source)
    name = f"krea2_reid_{uuid.uuid4().hex[:8]}_{os.path.basename(source)}"
    shutil.copy2(source, os.path.join(INPUT_DIR, name))
    return name


def build(prompt: str, image: str, seed: int, size: tuple[int, int], prefix: str) -> dict:
    width, height = size
    positive = ["34", 0]
    negative = ["35", 0]
    return {
        "1": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "2": {"class_type": "UNETLoader", "inputs": {
            "unet_name": MODEL, "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": CLIP, "type": "krea2", "device": "default"}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {
            "width": width, "height": height, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["33", 0], "positive": ["21", 0],
            "negative": ["24", 0], "latent_image": ["6", 0],
            "seed": seed, "control_after_generate": "fixed", "steps": 8,
            "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
            "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {
            "samples": ["7", 0], "vae": ["1", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {
            "images": ["8", 0], "filename_prefix": prefix}},
        "21": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {
            "conditioning": positive, "reference_latents_method": "index_timestep_zero"}},
        "24": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {
            "conditioning": negative, "reference_latents_method": "index_timestep_zero"}},
        "25": {"class_type": "LoadImage", "inputs": {"image": image}},
        "28": {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": ["25", 0], "upscale_method": "area",
            "megapixels": 0.140625, "resolution_steps": 16}},
        "32": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["2", 0], "lora_name": REID, "strength_model": 1.0}},
        "33": {"class_type": "Krea2OstrisEditModelPatch", "inputs": {
            "model": ["32", 0], "kv_cache": True}},
        "34": {"class_type": "TextEncodeKrea2OstrisEdit", "inputs": {
            "clip": ["3", 0], "vae": ["1", 0], "image1": ["28", 0],
            "prompt": prompt}},
        "35": {"class_type": "TextEncodeKrea2OstrisEdit", "inputs": {
            "clip": ["3", 0], "vae": ["1", 0], "image1": ["28", 0],
            "prompt": ""}},
    }


def run(graph: dict) -> tuple[list[dict] | None, float]:
    prompt_id = post("/prompt", {
        "prompt": graph, "client_id": uuid.uuid4().hex})["prompt_id"]
    started = time.time()
    while True:
        history = get(f"/history/{prompt_id}")
        if prompt_id in history:
            item = history[prompt_id]
            if item["status"].get("status_str") == "error":
                for message in item["status"].get("messages", []):
                    if message[0] == "execution_error":
                        print(json.dumps(message[1], ensure_ascii=False)[:2400],
                              file=sys.stderr)
                return None, time.time() - started
            images = [image for node in item["outputs"].values()
                      for image in node.get("images", [])]
            return images, time.time() - started
        print(f"  ... {time.time() - started:6.1f}s", flush=True)
        time.sleep(4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--ref", required=True,
                        help="retrato preparado, preferencialmente rosto + ombros")
    parser.add_argument("--tamanho", default="768x1024")
    parser.add_argument("--seed", type=int,
                        default=int(time.time()) % 1_000_000_000)
    parser.add_argument("--saida", required=True)
    parser.add_argument(
        "--permitir-mps-lento", action="store_true",
        help="ignora a trava de segurança do macOS; pode consumir dezenas de GB de swap",
    )
    args = parser.parse_args()

    if sys.platform == "darwin" and not args.permitir_mps_lento:
        print(
            "erro: Krea2-ReID INT8 usa aten::_int_mm, indisponível no MPS. "
            "O fallback para CPU não é praticável no M5 de 24 GB. Use o "
            "workflow 06 em CUDA; --permitir-mps-lento existe apenas para "
            "diagnóstico explícito.",
            file=sys.stderr,
        )
        return 2

    width, height = (int(value) for value in args.tamanho.lower().split("x", 1))
    name = stage(args.ref)
    destination = os.path.abspath(os.path.expanduser(args.saida))
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    prefix = "foto-macos/krea2-reid"
    print(f"[krea2-reid] {width}x{height} seed={args.seed}")
    outputs, elapsed = run(build(args.prompt, name, args.seed,
                                 (width, height), prefix))
    if not outputs:
        return 1
    image = outputs[-1]
    source = os.path.join(OUTPUT_DIR, image.get("subfolder", ""), image["filename"])
    shutil.copy2(source, destination)
    print(f"[krea2-reid] {destination} ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
