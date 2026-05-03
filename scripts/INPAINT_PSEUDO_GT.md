# Bear96 Pseudo-GT Inpainting Baseline

## Purpose

This script is a hand-crafted inpainting baseline for checking whether removal holes can be filled by per-scene optimization of only the removed, inpainting-side Gaussians.

It is not the final VGGT cross-attention decoder method. It is intended to test the optimization-side pieces before replacing the hand-crafted cross-view warp with a VGGT-guided foundation component:

- freeze background Gaussians after reconstruction/removal
- optimize only the removed Gaussian subset
- anchor one reference view with a 2D inpainted pseudo-GT image
- supervise other views with rendered-depth warping from the reference image
- regularize rendered depth with precomputed VGGT pseudo depth
- preserve original RGB outside the object masks

## Entry Point

```bash
bash /home/junho/jieun/mvsplat/scripts/run_bear96_inpaint_pseudo_gt.sh
```

The wrapper activates the `mvsplat` conda environment and runs:

```bash
python scripts/inpaint_bear96_pseudo_gt.py --save_depth
```

Common overrides:

```bash
GPU_ID=1 STEPS=2000 TARGET_BATCH=8 LR=1e-3 LR_OPACITY=5e-2 \
OUTPUT_DIR=/home/junho/jieun/mvsplat/outputs/bear96_inpaint_pseudo_gt \
bash /home/junho/jieun/mvsplat/scripts/run_bear96_inpaint_pseudo_gt.sh
```

## Inputs

Default inputs:

- dataset: `/home/junho/jieun/mvsplat/datasets/bear96_colmap`
- pretrained MVSplat checkpoint: `/home/junho/jieun/mvsplat/checkpoints/re10k.ckpt`
- removed Gaussian scene: `/home/junho/jieun/mvsplat/outputs/bear96_removal_lr2e-5/gaussians_removed_opacity.pt`
- remove mask: `/home/junho/jieun/mvsplat/outputs/bear96_removal_lr2e-5/remove_mask.pt`
- pseudo inpainted images: `/home/junho/jieun/3dgic_pixel_aligned_vggt/data/bear/bear/images_inpaint_unseen`
- object masks: `/home/junho/jieun/3dgic_pixel_aligned_vggt/data/bear/bear/inpaint_object_mask_255`
- VGGT pseudo depth: `/home/junho/jieun/3dgic_pixel_aligned_vggt/data/bear/bear/vggt_depth_inpaint_unseen`

Auxiliary pseudo images and masks are resized with the same rescale-then-center-crop geometry as MVSplat target images.

## Optimized Variables

Background Gaussians are frozen.

For inpainting Gaussians, the optimized variables are:

- `log_depth`: scalar depth along the original source context pixel ray
- `harmonics`: color / SH coefficients
- `opacity`: visibility

The following are not optimized:

- raw `xyz` means
- raw covariance matrices

Instead, each inpainting Gaussian mean is recomputed every iteration:

```text
mean = source_camera_center + exp(log_depth) * source_pixel_ray_direction
```

This keeps the baseline pixel-aligned. It avoids letting Gaussians drift away from their source context pixel correspondence.

Covariances are frozen because directly optimizing raw covariance matrices can break positive-semidefinite structure and make rasterization unstable.

## Losses

Each step always includes the fixed reference view and randomly samples `target_batch` non-reference views.

Reference view:

```text
L_ref = L1(render_rgb, ref_inpaint_rgb) inside ref mask
L_bg  = L1(render_rgb, original_rgb) outside mask
```

Non-reference views:

```text
L_warp =
  warp_l1_weight * L1(render_rgb, warp(ref_inpaint_rgb, rendered_depth)) inside valid target mask
+ warp_lpips_weight * LPIPS(bbox_crop(render_rgb), bbox_crop(pseudo_target_rgb))
L_bg   = L1(render_rgb, original_rgb) outside mask
```

For LPIPS, the pseudo target is composed from the current rendered image with only the valid mask region replaced by the warped reference color. This keeps the LPIPS crop local while avoiding supervision from inconsistent independently-inpainted non-reference RGB.

Depth:

```text
L_vggt_depth = log-depth L1(render_depth, VGGT_depth) on sampled views with available VGGT depth
```

Total:

```text
L =
  lambda_ref * L_ref
+ lambda_warp * L_warp
+ lambda_outside * L_bg
+ lambda_vggt_depth * L_vggt_depth
```

The current baseline does not include VGGT feature loss, decoder learning, SSIM, semantic regularization, or depth TV. Depth TV can be added later if the ray-depth optimization becomes noisy, but it is intentionally omitted for now to keep the baseline simple.

## Outputs

The output directory contains:

- `metadata.json`: run configuration and parameterization
- `train_loss.log`: step-wise loss log
- `render_step_XXXXXX/color`: rendered RGB for all target views at save steps
- `render_step_XXXXXX/gt`: original target RGB saved for visual comparison; this is not the inpainting pseudo-GT
- `render_step_XXXXXX/depth_rendered`: colorized rendered depth if `--save_depth` is enabled
- `render_step_XXXXXX/depth_raw`: raw rendered depth tensors if `--save_depth` is enabled
- `gaussians_inpainted.pt`: final merged scene

The saved scene also includes:

- `inpaint_depths`
- `inpaint_source_context_view`
- `inpaint_source_pixel`

These are stored for debugging the pixel-aligned depth parameterization.

## Relation to the Current Research Plan

This script corresponds to the hand-crafted baseline of the plan.

It matches:

- background freeze
- inpainting-only optimization
- reference RGB anchor
- rendered-depth `L_warp` with weak L1 and bbox LPIPS
- VGGT pseudo-depth supervision
- background preservation
- pixel-aligned depth parameterization

It intentionally excludes:

- VGGT feature precomputation/cross-attention
- VGGT Q/K/V cross-attention
- learned Gaussian decoder / adapter
- VGGT feature consistency loss

Therefore, this baseline should be used to answer:

```text
Can direct pixel-aligned inpainting GS optimization work at all before adding VGGT?
```

It should not be used as the final novelty claim.
