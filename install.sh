#!/usr/bin/env bash
# Instalador do foto-macos. Idempotente: pode rodar de novo sem estragar nada.
set -euo pipefail

COMFY="${COMFYUI_DIR:-$HOME/comfyui}"
PY="$COMFY/.venv/bin/python"
BIN="$HOME/.local/bin"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

info() { printf "\033[1;34m==>\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m !!\033[0m %s\n" "$*"; }
die()  { printf "\033[1;31m !!\033[0m %s\n" "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Darwin" ]] || die "isto e so para macOS"
[[ "$(uname -m)" == "arm64" ]]  || die "isto e so para Apple Silicon"
[[ -x "$PY" ]] || die "ComfyUI nao encontrado em $COMFY (defina COMFYUI_DIR)"
command -v uv >/dev/null || die "uv nao encontrado: https://docs.astral.sh/uv/"

# ── dependencias python ───────────────────────────────────────────────────────
info "dependencias python"
uv pip install -q --python "$PY" \
    numpy pillow opencv-python huggingface_hub 'mcp>=2' \
    pyobjc-framework-Vision pyobjc-framework-Quartz
uv tool install -q mflux 2>/dev/null || uv tool upgrade -q mflux 2>/dev/null || true
uv tool install -q "huggingface_hub[cli]" 2>/dev/null || true

# ── modelos ───────────────────────────────────────────────────────────────────
export HF_HUB_DISABLE_XET=1
M="$COMFY/models"
mkdir -p "$M"/{diffusion_models,text_encoders,vae,checkpoints,upscale_models}
DOWNLOAD_DIR="$(mktemp -d)"
trap 'rm -rf "$DOWNLOAD_DIR"' EXIT

baixar() { # repo arquivo destino
    local nome; nome="$(basename "$2")"
    if [[ -f "$3/$nome" ]]; then
        info "ja existe: $nome"
        return
    fi
    info "baixando $nome"
    "$BIN/hf" download "$1" "$2" --local-dir "$DOWNLOAD_DIR" >/dev/null
    mv "$DOWNLOAD_DIR/$2" "$3/"
}

baixar Comfy-Org/Mage-Flow diffusion_models/mage_flow_edit_turbo_bf16.safetensors "$M/diffusion_models"
baixar Comfy-Org/Mage-Flow text_encoders/qwen3vl_4b_bf16.safetensors             "$M/text_encoders"
baixar Comfy-Org/Mage-Flow vae/mage_flow_vae_bf16.safetensors                    "$M/vae"
baixar black-forest-labs/FLUX.2-klein-4B flux-2-klein-4b.safetensors             "$M/diffusion_models"
baixar Comfy-Org/z_image_turbo split_files/text_encoders/qwen_3_4b.safetensors   "$M/text_encoders"
baixar Comfy-Org/flux2-dev split_files/vae/flux2-vae.safetensors                  "$M/vae"
baixar SG161222/RealVisXL_V5.0 RealVisXL_V5.0_fp16.safetensors                   "$M/checkpoints"
baixar uwg/upscaler ESRGAN/1x-ITF-SkinDiffDetail-Lite-v1.pth                     "$M/upscale_models"
# ── patch do SeedVR2 (ver docs/BUGS.md) ───────────────────────────────────────
ATTN="$(find "$HOME/.local/share/uv/tools/mflux/lib" -path \
  '*/site-packages/mflux/models/seedvr2/model/seedvr2_transformer/attention.py' \
  -print -quit 2>/dev/null || true)"
if [[ -f "$ATTN" ]]; then
    if grep -q "_repeat_var" "$ATTN"; then
        info "patch do SeedVR2 ja aplicado"
    elif grep -q "mx.array(counts)" "$ATTN"; then
        info "aplicando patch do SeedVR2 (mx.repeat com repeats variavel)"
        cp "$ATTN" "$ATTN.bak"
        "$PY" "$HERE/src/patch_seedvr2.py" "$ATTN"
    fi
else
    warn "mflux nao encontrado; o estagio de ampliar vai cair no Lanczos"
fi

# ── CLI ───────────────────────────────────────────────────────────────────────
mkdir -p "$BIN"
ln -sf "$HERE/src/foto.py" "$BIN/foto"
chmod +x "$HERE/src/foto.py"
info "CLI instalado em $BIN/foto"

# ── workflows visuais ────────────────────────────────────────────────────────
"$PY" "$HERE/src/sync_workflows.py" >/dev/null
info "workflows instalados em $COMFY/user/default/workflows/foto-macos"

DRAW_MODEL="$HOME/Library/Application Support/local-photo-ai-m5/models/z_image_turbo_1.0_i8x.ckpt"
if command -v draw-things-cli >/dev/null && [[ -f "$DRAW_MODEL" ]]; then
    info "Draw Things + Z-Image i8x encontrados (backend rapido para gerar)"
else
    warn "Draw Things/Z-Image i8x ausente; foto gerar usara FLUX.2 no ComfyUI"
fi

cat <<EOF

Pronto.

  1. suba o ComfyUI:
       cd $COMFY && ./.venv/bin/python main.py \\
         --use-pytorch-cross-attention --reserve-vram 2 --listen 127.0.0.1

  2. edite uma foto:
       foto editar foto.jpg "troque a camiseta por um moletom preto"

EOF
