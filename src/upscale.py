#!/usr/bin/env python3
"""Upscale por modelo (ESRGAN-like) via ComfyUI, com downscale de volta ao alvo.

4x-UltraSharpV2 amplia 4x. Ampliar 4x e reduzir para 2x (lanczos) da mais
detalhe real e menos artefato do que pedir 2x direto — e a pratica padrao.
"""
import argparse, json, os, shutil, time, urllib.request, uuid

COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
INPUT_DIR = os.path.expanduser(
    os.environ.get("COMFYUI_DIR", "~/comfyui") + "/input")
OUT_DIR = os.environ.get("FOTO_OUT") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")


def post(p, payload):
    r = urllib.request.Request(COMFY + p, data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=120))


def get(p):
    return json.load(urllib.request.urlopen(COMFY + p, timeout=120))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default="up")
    ap.add_argument("--model", default="4x-UltraSharpV2.safetensors")
    ap.add_argument("--scale", type=float, default=2.0,
                    help="fator final desejado (o modelo faz 4x e reduz para ca)")
    a = ap.parse_args()

    name = f"up_{uuid.uuid4().hex[:8]}_{os.path.basename(a.image)}"
    shutil.copy(os.path.expanduser(a.image), os.path.join(INPUT_DIR, name))
    from PIL import Image
    w, h = Image.open(os.path.expanduser(a.image)).size
    tw, th = int(w * a.scale), int(h * a.scale)

    g = {
        "1": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": a.model}},
        "2": {"class_type": "LoadImage", "inputs": {"image": name, "upload": "image"}},
        "3": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["1", 0], "image": ["2", 0]}},
        "4": {"class_type": "ImageScale", "inputs": {"image": ["3", 0], "upscale_method": "lanczos",
                                                     "width": tw, "height": th, "crop": "disabled"}},
        "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0], "filename_prefix": a.out}},
    }
    pid = post("/prompt", {"prompt": g, "client_id": uuid.uuid4().hex})["prompt_id"]
    t0 = time.time()
    while True:
        hst = get(f"/history/{pid}")
        if pid in hst:
            st = hst[pid]["status"]
            if st.get("status_str") == "error":
                for m in st.get("messages", []):
                    if m[0] == "execution_error":
                        print(json.dumps(m[1], ensure_ascii=False)[:900])
                return
            for node in hst[pid]["outputs"].values():
                for im in node.get("images", []):
                    src = os.path.join(os.path.expanduser(os.environ.get("COMFYUI_DIR", "~/comfyui") + "/output"),
                                       im.get("subfolder", ""), im["filename"])
                    dst = os.path.join(OUT_DIR, im["filename"])
                    shutil.copy(src, dst)
                    print(f"[up] {dst}  {w}x{h} -> {tw}x{th}  ({time.time()-t0:.1f}s)")
            return
        time.sleep(3)


if __name__ == "__main__":
    main()
