#!/usr/bin/env python3
"""Runner Mage-Flow-Edit (Microsoft 4B, MIT) no ComfyUI local.

Um so no (TextEncodeMageFlowEdit) recebe ate 16 imagens de referencia e devolve
positive, negative e o latent ja no tamanho certo. As referencias sao
redimensionadas para a resolucao de saida antes de entrar no RoPE, o que e o
que segura o drift.

Uso:
  mage.py --out NOME --prompt "..." --img FOTO [--img REF2 ...]
          [--size 1024x1408] [--steps 4] [--turbo/--quality] [--seed N]
"""
import argparse, json, os, shutil, sys, time, urllib.request, uuid

COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
INPUT_DIR = os.path.expanduser(
    os.environ.get("COMFYUI_DIR", "~/comfyui") + "/input")
OUT_DIR = os.environ.get("FOTO_OUT") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")

TURBO = "mage_flow_edit_turbo_bf16.safetensors"
QUALITY = "mage_flow_edit_bf16.safetensors"
CLIP = "qwen3vl_4b_bf16.safetensors"
VAE = "mage_flow_vae_bf16.safetensors"


def post(path, payload):
    req = urllib.request.Request(COMFY + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))


def get(path):
    return json.load(urllib.request.urlopen(COMFY + path, timeout=120))


def stage(path):
    name = f"in_{uuid.uuid4().hex[:8]}_{os.path.basename(os.path.expanduser(path))}"
    shutil.copy(os.path.expanduser(path), os.path.join(INPUT_DIR, name))
    return name


def build(prompt, negative, imgs, steps, cfg, seed, unet, size, mp):
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "mage"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
    }
    refs = []
    for i, name in enumerate(imgs[:16], start=1):
        li, si = f"10{i}", f"20{i}"
        g[li] = {"class_type": "LoadImage", "inputs": {"image": name, "upload": "image"}}
        g[si] = {"class_type": "ImageScaleToTotalPixels",
                 "inputs": {"image": [li, 0], "upscale_method": "lanczos",
                            "megapixels": mp, "resolution_steps": 16}}
        refs.append([si, 0])

    # O Autogrow chega ACHATADO pela API, com o id do slot como prefixo:
    # "images.image_1", "images.image_2"... O execution.py os re-aninha em
    # images={"image_1": ...} via build_nested_inputs. Sem o prefixo "images.",
    # o dynamic_paths sai vazio e o no roda SEM referencia nenhuma.
    # 'vae' e opcional na assinatura mas obrigatorio na pratica — e o que gera os
    # ref_latents; sem ele o modelo ignora as referencias e gera do zero.
    enc = {"clip": ["2", 0], "prompt": prompt, "negative_prompt": negative,
           "vae": ["3", 0],
           "width": size[0] if size else 0, "height": size[1] if size else 0,
           "batch_size": 1}
    for i, r in enumerate(refs, start=1):
        enc[f"images.image_{i}"] = r

    g["30"] = {"class_type": "TextEncodeMageFlowEdit", "inputs": enc}
    g["40"] = {"class_type": "KSampler",
               "inputs": {"model": ["1", 0], "positive": ["30", 0], "negative": ["30", 1],
                          "latent_image": ["30", 2], "seed": seed, "steps": steps,
                          "cfg": cfg, "sampler_name": "euler", "scheduler": "simple",
                          "denoise": 1.0}}
    g["50"] = {"class_type": "VAEDecode", "inputs": {"samples": ["40", 0], "vae": ["3", 0]}}
    g["60"] = {"class_type": "SaveImage", "inputs": {"images": ["50", 0]}}
    return g


def run(graph, prefix):
    graph["60"]["inputs"]["filename_prefix"] = prefix
    pid = post("/prompt", {"prompt": graph, "client_id": uuid.uuid4().hex})["prompt_id"]
    t0, last = time.time(), ""
    while True:
        h = get(f"/history/{pid}")
        if pid in h:
            st = h[pid]["status"]
            if st.get("status_str") == "error":
                for m in st.get("messages", []):
                    if m[0] == "execution_error":
                        print(json.dumps(m[1], ensure_ascii=False)[:1200])
                return None, time.time() - t0
            outs = [im for node in h[pid]["outputs"].values() for im in node.get("images", [])]
            return outs, time.time() - t0
        msg = f"  ... {time.time()-t0:6.1f}s"
        if msg != last:
            print(msg, flush=True); last = msg
        time.sleep(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="")
    ap.add_argument("--img", action="append", required=True)
    ap.add_argument("--out", default="mage")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--mp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=int(time.time()) % 100000)
    ap.add_argument("--size", default=None, help="LxA da saida; vazio = tamanho da 1a referencia")
    ap.add_argument("--quality", action="store_true", help="usa o modelo nao-turbo (30 steps)")
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    unet = QUALITY if a.quality else TURBO
    steps = a.steps or (30 if a.quality else 4)
    cfg = a.cfg if a.cfg != 1.0 else (4.0 if a.quality else 1.0)
    size = tuple(int(v) for v in a.size.lower().split("x")) if a.size else None
    imgs = [stage(p) for p in a.img]

    print(f"[mage] {unet} steps={steps} cfg={cfg} seed={a.seed} refs={len(imgs)} size={size or 'auto'}")
    g = build(a.prompt, a.negative, imgs, steps, cfg, a.seed, unet, size, a.mp)
    outs, dt = run(g, a.out)
    if not outs:
        sys.exit(1)
    for im in outs:
        src = os.path.join(os.path.expanduser(os.environ.get("COMFYUI_DIR", "~/comfyui") + "/output"),
                           im.get("subfolder", ""), im["filename"])
        dst = os.path.join(OUT_DIR, im["filename"])
        shutil.copy(src, dst)
        print(f"[mage] {dst}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
