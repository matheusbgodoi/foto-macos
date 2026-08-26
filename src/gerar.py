#!/usr/bin/env python3
"""Geracao do zero (text-to-image), com suporte a LoRA.

Usa SDXL por um motivo pratico: e o ecossistema com MUITO mais LoRA disponivel
(CivitAI tem dezenas de milhares — estilo, desenho, anime, produto, fotografia).
O checkpoint RealVisXL ja esta no disco porque o estagio de polimento precisa
dele, entao gerar do zero sai "de graca" em disco.

Os modelos mais novos (Z-Image, FLUX.2 klein, Mage-Flow t2i) tem melhor
aderencia a prompt, mas quase nao tem LoRA. Da para escolher com --modelo.
"""
import argparse, json, os, shutil, sys, time, urllib.request, uuid

COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
OUT_DIR = os.environ.get("FOTO_OUT") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")

SDXL = "RealVisXL_V5.0_fp16.safetensors"

NEG = ("plastic skin, airbrushed, waxy, cgi, 3d render, illustration, cartoon, "
       "deformed hands, extra fingers, watermark, text, oversaturated")


def post(p, payload):
    r = urllib.request.Request(COMFY + p, data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=120))


def get(p):
    return json.load(urllib.request.urlopen(COMFY + p, timeout=120))


def build(prompt, negativo, w, h, steps, cfg, seed, loras, prefix):
    g = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SDXL}},
    }
    modelo, clip = ["1", 0], ["1", 1]
    for i, (nome, forca) in enumerate(loras, start=1):
        nid = f"5{i}"
        g[nid] = {"class_type": "LoraLoader",
                  "inputs": {"model": modelo, "clip": clip, "lora_name": nome,
                             "strength_model": forca, "strength_clip": forca}}
        modelo, clip = [nid, 0], [nid, 1]

    g.update({
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": prompt}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": negativo}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": w, "height": h, "batch_size": 1}},
        "6": {"class_type": "KSampler",
              "inputs": {"model": modelo, "positive": ["2", 0], "negative": ["3", 0],
                         "latent_image": ["4", 0], "seed": seed, "steps": steps,
                         "cfg": cfg, "sampler_name": "dpmpp_2m_sde",
                         "scheduler": "karras", "denoise": 1.0}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": prefix}},
    })
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
                        print(json.dumps(m[1], ensure_ascii=False)[:1000])
                return None, time.time() - t0
            outs = [im for n in h[pid]["outputs"].values() for im in n.get("images", [])]
            return outs, time.time() - t0
        msg = f"  ... {time.time()-t0:6.1f}s"
        if msg != last:
            print(msg, flush=True); last = msg
        time.sleep(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--saida", default=None)
    ap.add_argument("--out", default="gerar")
    ap.add_argument("--nao", default=NEG)
    ap.add_argument("--tamanho", default="1024x1024")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=int(time.time()) % 100000)
    ap.add_argument("--lora", action="append", default=[],
                    help="arquivo.safetensors[:forca], repetivel")
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    w, h = (int(v) for v in a.tamanho.lower().split("x"))
    loras = []
    for spec in a.lora:
        nome, _, forca = spec.partition(":")
        loras.append((nome, float(forca) if forca else 1.0))

    print(f"[gerar] {w}x{h} steps={a.steps} cfg={a.cfg} seed={a.seed} loras={len(loras)}")
    outs, dt = run(build(a.prompt, a.nao, w, h, a.steps, a.cfg, a.seed, loras, a.out))
    if not outs:
        sys.exit(1)
    for im in outs:
        src = os.path.join(os.path.expanduser(
            os.environ.get("COMFYUI_DIR", "~/comfyui") + "/output"),
            im.get("subfolder", ""), im["filename"])
        dst = os.path.abspath(os.path.expanduser(a.saida)) if a.saida \
            else os.path.join(OUT_DIR, im["filename"])
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.copy(src, dst)
        print(f"[gerar] {dst}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
