#!/usr/bin/env bash
set -euo pipefail

# Multi-view bear scene fine-tuning (16 evenly-spaced context views).
# Context: 16 views sampled every 6 frames from 96 training frames.
# Target: random 1 view from all 96 frames each step (3DGS-style supervision).
# DINOv2 frozen. Memory: ~46 GB, ~0.55s/step → 2000 steps ≈ 18 min.

DEPTHSPLAT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET=/home/junho/jieun/3dgic_pixel_aligned_vggt/external/depthsplat/datasets/bear_re10k_like
PRETRAINED=/home/junho/jieun/3dgic_pixel_aligned_vggt/external/depthsplat/pretrained/depthsplat-gs-base-re10k-256x256-view2-ca7b6795.pth
OUTPUT=${DEPTHSPLAT_DIR}/outputs/finetune/bear_multiview

MAX_STEPS=${1:-3000}
GPU=${2:-0}

cd "${DEPTHSPLAT_DIR}"

echo "Running multi-view bear fine-tuning for ${MAX_STEPS} steps on GPU ${GPU}..."
echo "Context: 16 views | Target: random 1 view | DINOv2: frozen"

CUDA_VISIBLE_DEVICES=${GPU} \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.main \
  +experiment=re10k \
  mode=train \
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
  dataset.view_sampler.num_target_views=1 \
  'dataset.view_sampler.context_views=[0,6,12,18,24,30,36,42,48,54,60,66,72,78,84,90]' \
  checkpointing.pretrained_model="${PRETRAINED}" \
  checkpointing.every_n_train_steps=500 \
  checkpointing.save_top_k=5 \
  model.encoder.num_scales=2 \
  model.encoder.upsample_factor=2 \
  model.encoder.lowest_feature_resolution=4 \
  model.encoder.monodepth_vit_type=vitb \
  model.encoder.local_mv_match=2 \
  optimizer.lr=2e-4 \
  optimizer.lr_monodepth=0.0 \
  train.eval_model_every_n_val=0 \
  train.print_log_every_n_steps=1 \
  trainer.max_steps=${MAX_STEPS} \
  trainer.val_check_interval=null \
  trainer.num_sanity_val_steps=0 \
  data_loader.train.batch_size=1 \
  data_loader.train.num_workers=0 \
  data_loader.val.num_workers=0 \
  data_loader.test.num_workers=0 \
  output_dir="${OUTPUT}"
