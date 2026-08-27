#!/usr/bin/env python3
"""Estagio 2 — POLIMENTO. Da textura de foto real ao que o modelo redesenhou.

Receita extraida do "Qwen Image To Dataset Workflow" (r/comfyui). O que resolve
o aspecto plastico NAO e upscaler nem denoise alto — e uma passagem de denoise
MUITO baixo por um checkpoint SDXL fotorrealista, seguida de um upscaler 1x
especializado em pele:

    escala 1.5x (lanczos)
      -> KSampler SDXL, 20 steps, cfg 3.5, dpmpp_2m_sde/karras, denoise 0.03
      -> volta para 1 MP
      -> ImageUpscaleWithModel com 1x-ITF-SkinDiffDetail-Lite-v1

denoise 0.03 quase nao mexe na estrutura (nao troca rosto, nao move objetos):
ele so reescreve a camada de altissima frequencia, que e exatamente onde mora a
microtextura que o VAE de um modelo de edicao apaga. E o oposto do denoise 0.55
de outro workflow conhecido, que serve para imagem gerada do zero e reimagina
o conteudo.
"""
import argparse, json, os, shutil, sys, time, urllib.request, uuid

COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
INPUT_DIR = os.path.expanduser(
    os.environ.get("COMFYUI_DIR", "~/comfyui") + "/input")
OUT_DIR = os.environ.get("FOTO_OUT") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")

SDXL = "RealVisXL_V5.0_fp16.safetensors"
SKIN = "1x-ITF-SkinDiffDetail-Lite-v1.pth"

POS = ("photograph, natural skin texture, visible pores, fine fabric weave, "
       "sharp focus, realistic film grain, unretouched")
NEG = ("smooth plastic skin, airbrushed, waxy, blurry, cgi, 3d render, "
       "oversharpened, halo artifacts")


def post(p, payload):
    r = urllib.request.Request(COMFY + p, data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=120))


def get(p):
    return json.load(urllib.request.urlopen(COMFY + p, timeout=120))


def build(image_name, steps, cfg, denoise, scale, seed, mp, prefix, skin):
    g = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SDXL}},
        "2": {"class_type": "LoadImage", "inputs": {"image": image_name, "upload": "image"}},
        "3": {"class_type": "ImageScaleBy",
              "inputs": {"image": ["2", 0], "upscale_method": "lanczos", "scale_by": scale}},
        "4": {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": POS}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": NEG}},
        "7": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["5", 0], "negative": ["6", 0],
                         "latent_image": ["4", 0], "seed": seed, "steps": steps,
                         "cfg": cfg, "sampler_name": "dpmpp_2m_sde",
                         "scheduler": "karras", "denoise": denoise}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
        "9": {"class_type": "ImageScaleToTotalPixels",
              "inputs": {"image": ["8", 0], "upscale_method": "lanczos",
                         "megapixels": mp, "resolution_steps": 1}},
    }
    last = ["9", 0]
    if skin:
        g["10"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": SKIN}}
        g["11"] = {"class_type": "ImageUpscaleWithModel",
                   "inputs": {"upscale_model": ["10", 0], "image": last}}
        last = ["11", 0]
    g["20"] = {"class_type": "SaveImage", "inputs": {"images": last, "filename_prefix": prefix}}
    return g


def run(graph):
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
            outs = [im for n in h[pid]["outputs"].values() for im in n.get("images", [])]
            return outs, time.time() - t0
        msg = f"  ... {time.time()-t0:6.1f}s"
        if msg != last:
            print(msg, flush=True); last = msg
        time.sleep(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("imagem")
    ap.add_argument(
        "--out", default="polido",
        help=("prefixo dentro de FOTO_OUT ou caminho final explicito. "
              "Caminhos absolutos nunca sao enviados ao SaveImage do ComfyUI."),
    )
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=3.5)
    ap.add_argument("--denoise", type=float, default=0.03,
                    help="0.03 = so microtextura. Acima de ~0.15 comeca a mexer no rosto.")
    ap.add_argument("--escala", type=float, default=1.5)
    ap.add_argument("--mp", type=float, default=0.0,
                    help="megapixels de saida; 0 = mantem o tamanho de entrada")
    ap.add_argument("--sem-pele", action="store_true", help="pula o 1x-ITF-SkinDiffDetail")
    ap.add_argument("--seed", type=int, default=int(time.time()) % 100000)
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    src = os.path.abspath(os.path.expanduser(a.imagem))
    name = f"pol_{uuid.uuid4().hex[:8]}_{os.path.basename(src)}"
    shutil.copy(src, os.path.join(INPUT_DIR, name))

    from PIL import Image
    w, h = Image.open(src).size
    mp = a.mp if a.mp > 0 else (w * h) / 1_000_000

    explicit_out = (os.path.isabs(os.path.expanduser(a.out)) or
                    os.path.dirname(os.path.expanduser(a.out)) != "" or
                    os.path.splitext(a.out)[1].lower() in {".png", ".jpg", ".jpeg", ".webp"})
    requested_out = os.path.abspath(os.path.expanduser(a.out)) if explicit_out else None
    # SaveImage aceita apenas um prefixo relativo ao output do ComfyUI. Passar
    # /Users/.../Downloads fazia todo o grafo terminar e falhar somente ao salvar.
    prefix = (f"polir_{uuid.uuid4().hex[:10]}" if explicit_out else a.out)

    print(f"[polir] {w}x{h} · denoise={a.denoise} steps={a.steps} cfg={a.cfg} "
          f"escala={a.escala} pele={not a.sem_pele}")
    g = build(name, a.steps, a.cfg, a.denoise, a.escala, a.seed, mp, prefix,
              not a.sem_pele)
    outs, dt = run(g)
    if not outs:
        sys.exit(1)
    for index, im in enumerate(outs):
        s = os.path.join(os.path.expanduser(os.environ.get("COMFYUI_DIR", "~/comfyui") + "/output"), im.get("subfolder", ""), im["filename"])
        if requested_out and index == 0:
            dst = requested_out
        elif requested_out:
            root, ext = os.path.splitext(requested_out)
            dst = f"{root}_{index + 1}{ext or '.png'}"
        else:
            dst = os.path.join(OUT_DIR, im["filename"])
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.copy(s, dst)
        print(f"[polir] {dst}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
