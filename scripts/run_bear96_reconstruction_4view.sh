#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
ROOT="/home/junho/jieun/mvsplat"
DATASET_ROOT="${DATASET_ROOT:-$ROOT/datasets/bear96_colmap}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/bear96_reconstruction_4view}"
CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/re10k.ckpt}"

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
python scripts/reconstruct_bear96_4view.py \
  --dataset_root "$DATASET_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --num_context 4 \
  --render_chunk "${RENDER_CHUNK:-8}" \
  --image_height "${IMAGE_HEIGHT:-256}" \
  --image_width "${IMAGE_WIDTH:-256}" \
  "$@"
