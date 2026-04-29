#!/usr/bin/env bash
set -euo pipefail

# Render bear scene from a trained checkpoint (16-view context).
# Runs mode=test: saves RGB images, depth maps, and optional video/PLY.
#
# Usage:
#   bash scripts/render_bear.sh <checkpoint> [gpu]
#
# Example:
#   bash scripts/render_bear.sh outputs/finetune/bear_multiview/checkpoints/epoch_1999-step_2000.ckpt 1

DEPTHSPLAT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET=/home/junho/jieun/3dgic_pixel_aligned_vggt/external/depthsplat/datasets/bear_re10k_like

CKPT=${1:?"Usage: $0 <checkpoint> [gpu=1]"}
GPU=${2:-1}

OUTPUT=${DEPTHSPLAT_DIR}/outputs/render/bear_ctx16

cd "${DEPTHSPLAT_DIR}"

echo "Rendering bear scene from checkpoint: ${CKPT}"
echo "Context: 16 views | Output: ${OUTPUT}"

CUDA_VISIBLE_DEVICES=${GPU} \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.main \
  +experiment=re10k \
  mode=test \
  wandb.mode=disabled \
  dataset.roots=["${DATASET}"] \
  dataset.skip_bad_shape=false \
  dataset.name=re10k \
  dataset.augment=false \
  dataset.make_baseline_1=false \
  dataset.baseline_scale_bounds=false \
  dataset.shuffle_val=false \
  dataset.test_chunk_interval=1 \
  dataset.train_times_per_scene=1 \
  dataset.use_index_to_load_chunk=false \
  dataset.highres=false \
  dataset.image_shape=[256,256] \
  dataset/view_sampler=arbitrary \
  dataset.view_sampler.num_context_views=16 \
  'dataset.view_sampler.context_views=[0,6,12,18,24,30,36,42,48,54,60,66,72,78,84,90]' \
  dataset.view_sampler.num_target_views=96 \
  'dataset.view_sampler.target_views=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95]' \
  test.render_chunk_size=8 \
  checkpointing.load="${CKPT}" \
  model.encoder.num_scales=2 \
  model.encoder.upsample_factor=2 \
  model.encoder.lowest_feature_resolution=4 \
  model.encoder.monodepth_vit_type=vitb \
  model.encoder.local_mv_match=2 \
  test.compute_scores=true \
  test.save_image=true \
  test.save_gt_image=true \
  test.save_depth=true \
  test.save_depth_concat_img=true \
  test.save_video=false \
  test.save_gaussian=false \
  data_loader.test.num_workers=0 \
  output_dir="${OUTPUT}"

echo ""
echo "Done. Results saved to: ${OUTPUT}"
echo "  RGB images:     ${OUTPUT}/images/<scene>/color/"
echo "  Encoder depth:  ${OUTPUT}/images/<scene>/depth/          (per context pixel, cost-volume)"
echo "  Rendered depth: ${OUTPUT}/images/<scene>/depth_rendered/ (per target pixel, Gaussian alpha-composite)"
echo "  Scores:         ${OUTPUT}/metrics/"
