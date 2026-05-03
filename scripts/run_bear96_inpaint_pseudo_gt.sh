#!/usr/bin/env bash
set -euo pipefail

source /home/junho/miniconda3/etc/profile.d/conda.sh
conda activate mvsplat
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
export MPLCONFIGDIR=/tmp/matplotlib

cd /home/junho/jieun/mvsplat

python scripts/inpaint_bear96_pseudo_gt.py \
  --dataset_root "${DATASET_ROOT:-/home/junho/jieun/mvsplat/datasets/bear96_colmap}" \
  --checkpoint "${CHECKPOINT:-/home/junho/jieun/mvsplat/checkpoints/re10k.ckpt}" \
  --gaussians "${GAUSSIANS:-/home/junho/jieun/mvsplat/outputs/bear96_removal_lr2e-5/gaussians_removed_opacity.pt}" \
  --remove_mask "${REMOVE_MASK:-/home/junho/jieun/mvsplat/outputs/bear96_removal_lr2e-5/remove_mask.pt}" \
  --output_dir "${OUTPUT_DIR:-/home/junho/jieun/mvsplat/outputs/bear96_inpaint_pseudo_gt}" \
  --num_context "${NUM_CONTEXT:-8}" \
  --ref_view "${REF_VIEW:-0}" \
  --steps "${STEPS:-2000}" \
  --target_batch "${TARGET_BATCH:-4}" \
  --lr "${LR:-1e-3}" \
  --lr_opacity "${LR_OPACITY:-5e-2}" \
  --lambda_outside "${LAMBDA_OUTSIDE:-0.3}" \
  --lambda_warp "${LAMBDA_WARP:-1.0}" \
  --warp_l1_weight "${WARP_L1_WEIGHT:-0.2}" \
  --warp_lpips_weight "${WARP_LPIPS_WEIGHT:-0.5}" \
  --lambda_ref "${LAMBDA_REF:-2.0}" \
  --lambda_vggt_depth "${LAMBDA_VGGT_DEPTH:-0.1}" \
  --vggt_depth_dir "${VGGT_DEPTH_DIR:-/home/junho/jieun/3dgic_pixel_aligned_vggt/data/bear/bear/vggt_depth_inpaint_unseen}" \
  --vggt_depth_mask_mode "${VGGT_DEPTH_MASK_MODE:-masked}" \
  --opacity_init "${OPACITY_INIT:-0.05}" \
  --render_chunk "${RENDER_CHUNK:-4}" \
  --save_every "${SAVE_EVERY:-1000}" \
  --save_depth
