#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
ROOT="/home/junho/jieun/mvsplat"
DATASET_ROOT="${DATASET_ROOT:-$ROOT/datasets/bear96_colmap}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/bear96_per_scene_4view_3000}"
CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/re10k.ckpt}"
NUM_CONTEXT="${NUM_CONTEXT:-4}"

source /home/junho/miniconda3/etc/profile.d/conda.sh
conda activate mvsplat

cd "$ROOT"

python scripts/prepare_bear96_re10k.py \
  --image_root /home/junho/jieun/3dgic_pixel_aligned_vggt/data/bear/bear/images \
  --sparse_root /home/junho/jieun/3dgic_pixel_aligned_vggt/data/bear/bear/sparse/0 \
  --output_root "$DATASET_ROOT"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
CUDA_HOME=/usr/local/cuda-12.1 \
LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}" \
python scripts/optimize_bear96_4view.py \
  --dataset_root "$DATASET_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --num_context "$NUM_CONTEXT" \
  --steps "${STEPS:-3000}" \
  --target_batch "${TARGET_BATCH:-4}" \
  --lr "${LR:-2e-5}" \
  --lpips_weight "${LPIPS_WEIGHT:-0.05}" \
  --print_every "${PRINT_EVERY:-50}" \
  --save_every "${SAVE_EVERY:-500}" \
  --render_chunk "${RENDER_CHUNK:-4}" \
  --image_height "${IMAGE_HEIGHT:-256}" \
  --image_width "${IMAGE_WIDTH:-256}" \
  "$@"
