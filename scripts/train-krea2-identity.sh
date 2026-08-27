#!/usr/bin/env bash
# Treino reproduzivel de identidade Krea 2 numa NVIDIA de 24 GB.
# As fotos e os checkpoints ficam fora do repositorio.
set -euo pipefail

: "${MUSUBI_DIR:?defina MUSUBI_DIR para o clone de kohya-ss/musubi-tuner}"
: "${DATA_DIR:?defina DATA_DIR para as imagens e captions .txt}"
: "${DIT_RAW:?defina DIT_RAW para raw.safetensors do Krea 2}"
: "${VAE:?defina VAE para qwen_image_vae.safetensors}"
: "${TEXT_ENCODER:?defina TEXT_ENCODER para qwen3vl_4b_bf16.safetensors}"

PYTHON="${MUSUBI_PYTHON:-$MUSUBI_DIR/.venv/bin/python}"
ACCELERATE="${MUSUBI_ACCELERATE:-$MUSUBI_DIR/.venv/bin/accelerate}"
WORK_DIR="${WORK_DIR:-$DATA_DIR/../musubi-krea2}"
OUTPUT_NAME="${OUTPUT_NAME:-identity-krea2}"
STEPS="${STEPS:-2000}"
RESOLUTION="${RESOLUTION:-768}"
BLOCKS_TO_SWAP="${BLOCKS_TO_SWAP:-16}"
CONFIG="$WORK_DIR/dataset.toml"
CACHE_DIR="$WORK_DIR/cache-$RESOLUTION"
OUTPUT_DIR="$WORK_DIR/output-$RESOLUTION"

mkdir -p "$CACHE_DIR" "$OUTPUT_DIR"

if ! find "$DATA_DIR" -maxdepth 1 -type f -name '*.txt' -print -quit | grep -q .; then
  printf 'erro: nenhuma caption .txt em %s\n' "$DATA_DIR" >&2
  exit 2
fi

printf '%s\n' \
  '[general]' \
  "resolution = [$RESOLUTION, $RESOLUTION]" \
  'caption_extension = ".txt"' \
  'batch_size = 1' \
  'enable_bucket = true' \
  'bucket_no_upscale = false' \
  '' \
  '[[datasets]]' \
  "image_directory = \"$DATA_DIR\"" \
  "cache_directory = \"$CACHE_DIR\"" \
  'num_repeats = 1' >"$CONFIG"

"$PYTHON" "$MUSUBI_DIR/src/musubi_tuner/krea2_cache_latents.py" \
  --dataset_config "$CONFIG" --vae "$VAE"

"$PYTHON" "$MUSUBI_DIR/src/musubi_tuner/krea2_cache_text_encoder_outputs.py" \
  --dataset_config "$CONFIG" --text_encoder "$TEXT_ENCODER" --batch_size 1

exec "$ACCELERATE" launch \
  --num_processes 1 --num_machines 1 --num_cpu_threads_per_process 1 \
  --mixed_precision bf16 \
  "$MUSUBI_DIR/src/musubi_tuner/krea2_train_network.py" \
  --dit "$DIT_RAW" --vae "$VAE" --dataset_config "$CONFIG" \
  --sdpa --mixed_precision bf16 \
  --timestep_sampling krea2_shift --weighting_scheme none \
  --optimizer_type adamw8bit --learning_rate 1e-4 \
  --gradient_checkpointing \
  --max_data_loader_n_workers 2 --persistent_data_loader_workers \
  --network_module networks.lora_krea2 \
  --network_dim 32 --network_alpha 32 \
  --fp8_base --fp8_scaled \
  --blocks_to_swap "$BLOCKS_TO_SWAP" \
  --block_swap_h2d_only --block_swap_ring_size 2 \
  --max_train_steps "$STEPS" --save_every_n_steps 250 \
  --save_state --save_last_n_steps_state 500 --seed 42 \
  --output_dir "$OUTPUT_DIR" --output_name "$OUTPUT_NAME"
