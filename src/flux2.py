#!/usr/bin/env python3
"""FLUX.2 Klein 4B: texto, edicao e composicao com multiplas referencias.

Este runner segue o workflow oficial do ComfyUI para o modelo distilled:
4 steps, CFG 1, sampler Euler e Flux2Scheduler. Ao contrario do Mage-Flow,
FLUX.2 foi treinado como um modelo unificado de texto->imagem e edicao
multi-referencia. Use este caminho para o caso "pessoa + roupa -> cena nova".
"""
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
OUT_DIR = os.environ.get("FOTO_OUT") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")

MODEL = "flux-2-klein-4b.safetensors"
CLIP = "qwen_3_4b.safetensors"
VAE = "flux2-vae.safetensors"


def post(path, payload):
    request = urllib.request.Request(
        COMFY + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(request, timeout=120))


def get(path):
    return json.load(urllib.request.urlopen(COMFY + path, timeout=120))


def stage(path):
    source = os.path.abspath(os.path.expanduser(path))
    name = f"flux2_{uuid.uuid4().hex[:8]}_{os.path.basename(source)}"
    shutil.copy(source, os.path.join(INPUT_DIR, name))
    return name


def build(prompt, images, seed, size, megapixels, prefix):
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": MODEL, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": CLIP, "type": "flux2"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "20": {"class_type": "CLIPTextEncode", "inputs": {
            "clip": ["2", 0], "text": prompt}},
        "21": {"class_type": "ConditioningZeroOut", "inputs": {
            "conditioning": ["20", 0]}},
    }

    scaled = []
    for index, name in enumerate(images, start=1):
        load_id = str(100 + index)
        scale_id = str(200 + index)
        graph[load_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
        graph[scale_id] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": [load_id, 0], "upscale_method": "nearest-exact",
            "megapixels": megapixels, "resolution_steps": 16}}
        scaled.append(scale_id)

    if size:
        width, height = size
    elif scaled:
        graph["22"] = {"class_type": "GetImageSize", "inputs": {
            "image": [scaled[0], 0]}}
        width, height = ["22", 0], ["22", 1]
    else:
        width, height = 1024, 1024

    positive = ["20", 0]
    negative = ["21", 0]
    for index, scale_id in enumerate(scaled, start=1):
        encode_id = str(300 + index)
        positive_id = str(400 + index)
        negative_id = str(500 + index)
        graph[encode_id] = {"class_type": "VAEEncode", "inputs": {
            "pixels": [scale_id, 0], "vae": ["3", 0]}}
        graph[positive_id] = {"class_type": "ReferenceLatent", "inputs": {
            "conditioning": positive, "latent": [encode_id, 0]}}
        graph[negative_id] = {"class_type": "ReferenceLatent", "inputs": {
            "conditioning": negative, "latent": [encode_id, 0]}}
        positive = [positive_id, 0]
        negative = [negative_id, 0]

    graph.update({
        "30": {"class_type": "EmptyFlux2LatentImage", "inputs": {
            "width": width, "height": height, "batch_size": 1}},
        "31": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "32": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "33": {"class_type": "Flux2Scheduler", "inputs": {
            "steps": 4, "width": width, "height": height}},
        "34": {"class_type": "CFGGuider", "inputs": {
            "model": ["1", 0], "positive": positive, "negative": negative, "cfg": 1.0}},
        "35": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["31", 0], "guider": ["34", 0], "sampler": ["32", 0],
            "sigmas": ["33", 0], "latent_image": ["30", 0]}},
        "36": {"class_type": "VAEDecode", "inputs": {
            "samples": ["35", 1], "vae": ["3", 0]}},
        "37": {"class_type": "SaveImage", "inputs": {
            "images": ["36", 0], "filename_prefix": prefix}},
    })
    return graph


def run(graph):
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
                        print(json.dumps(message[1], ensure_ascii=False)[:1600])
                return None, time.time() - started
            images = [image for node in item["outputs"].values()
                      for image in node.get("images", [])]
            return images, time.time() - started
        print(f"  ... {time.time() - started:6.1f}s", flush=True)
        time.sleep(4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--ref", action="append", default=[])
    parser.add_argument("--tamanho", default=None, help="LxA; vazio segue a primeira referencia")
    parser.add_argument("--mp", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=int(time.time()) % 1_000_000_000)
    parser.add_argument("--out", default="flux2")
    parser.add_argument("--saida", default=None)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    size = tuple(int(value) for value in args.tamanho.lower().split("x")) \
        if args.tamanho else None
    refs = [stage(path) for path in args.ref]
    print(f"[flux2] refs={len(refs)} size={size or 'auto'} seed={args.seed}")
    outputs, elapsed = run(build(args.prompt, refs, args.seed, size, args.mp, args.out))
    if not outputs:
        sys.exit(1)
    for image in outputs:
        source = os.path.join(OUTPUT_DIR, image.get("subfolder", ""), image["filename"])
        destination = os.path.abspath(os.path.expanduser(args.saida)) if args.saida \
            else os.path.join(OUT_DIR, image["filename"])
        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
        shutil.copy(source, destination)
        print(f"[flux2] {destination} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
