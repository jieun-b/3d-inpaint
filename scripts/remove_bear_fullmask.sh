#!/usr/bin/env bash
set -euo pipefail

# Stage 2a: Removal — 16-view mask with depth-consistent consistency voting
#
# Usage:
#   bash scripts/remove_bear_fullmask.sh <checkpoint> [gpu]

DEPTHSPLAT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET=/home/junho/jieun/3dgic_pixel_aligned_vggt/external/depthsplat/datasets/bear_re10k_like

CKPT=${1:?"Usage: $0 <checkpoint> [gpu=1]"}
GPU=${2:-1}

cd "${DEPTHSPLAT_DIR}"

CUDA_VISIBLE_DEVICES=${GPU} python scripts/remove_bear_fullmask.py \
  --checkpoint "${CKPT}" \
  --datasets_root "${DATASET}" \
  --output_dir outputs/removal/bear_fullmask_consistency \
  --consistency_votes 2 \
  --gpu 0

echo "Done -> outputs/removal/bear_fullmask_consistency/"
