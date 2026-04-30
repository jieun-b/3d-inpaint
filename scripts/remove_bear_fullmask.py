#!/usr/bin/env python3
"""
Stage 2a — Removal (full-mask variant).

Uses masks from ALL 16 context views. With --consistency_votes 0, each Gaussian
is directly identified by its originating context pixel. With --consistency_votes
> 0, depth-consistent object votes from all context masks are unioned in.

Pros: complete coverage, zero ambiguity
Cons: need masks for all context views (all 96 frames have masks here, so fine)

Output:
  outputs/removal/bear_fullmask/
    color/frame_XXXXX.png   — rendered RGB after removal (96 views)
    depth/frame_XXXXX.png   — rendered depth after removal
    keep_mask.pt            — [1, N] bool, for downstream inpainting
    removal_stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.global_cfg import set_cfg
from scripts._removal_utils import (
    CTX_VIEWS, TGT_VIEWS,
    load_encoder_weights, make_encoder_cfg, make_dataset_cfg,
    build_batch, build_models,
    load_masks,
    apply_opacity_mask,
    render_views_chunked, save_renders, save_encoder_depths,
)


def build_object_mask(
    context_mask: torch.Tensor,  # [B, V, H, W] bool
    num_surfaces: int,
    samples_per_pixel: int,
) -> torch.Tensor:
    """
    Map [B, V, H, W] pixel mask → [B, V*H*W*srf*spp] Gaussian mask.
    Gaussian layout follows encoder output: view-major, then pixel-major.
    """
    pix = rearrange(context_mask, "b v h w -> b v (h w)")
    return repeat(pix, "b v r -> b (v r srf spp)", srf=num_surfaces, spp=samples_per_pixel)


def project_gaussians_to_context(
    means: torch.Tensor,       # [B, N, 3]
    extrinsics: torch.Tensor,  # [B, V, 4, 4]
    intrinsics: torch.Tensor,  # [B, V, 3, 3]
    image_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project Gaussian means into every context view. Returns grid [B,V,N,2], camera z, valid [B,V,N]."""
    B, N, _ = means.shape
    _, V, _, _ = extrinsics.shape
    H, W = image_shape
    device = means.device

    pts_h = torch.cat([means, torch.ones(B, N, 1, device=device, dtype=means.dtype)], dim=-1)
    w2c = torch.inverse(extrinsics)
    pts_cam = torch.einsum("bvij,bnj->bvni", w2c, pts_h)[..., :3]
    z = pts_cam[..., 2]
    z_safe = z.clamp_min(1.0e-6)

    u = intrinsics[:, :, None, 0, 0] * W * pts_cam[..., 0] / z_safe
    u = u + intrinsics[:, :, None, 0, 2] * W
    v = intrinsics[:, :, None, 1, 1] * H * pts_cam[..., 1] / z_safe
    v = v + intrinsics[:, :, None, 1, 2] * H
    grid = torch.stack([(u / (W - 1)) * 2 - 1, (v / (H - 1)) * 2 - 1], dim=-1)

    in_bounds = (grid[..., 0] >= -1) & (grid[..., 0] <= 1) & (grid[..., 1] >= -1) & (grid[..., 1] <= 1)
    valid = (z > 0.01) & in_bounds
    return grid, z, valid


def build_consistency_object_mask(
    direct_object_mask: torch.Tensor,  # [B, N]
    gaussians_means: torch.Tensor,     # [B, N, 3]
    context_mask: torch.Tensor,        # [B, V, H, W] bool
    extrinsics: torch.Tensor,          # [B, V, 4, 4]
    intrinsics: torch.Tensor,          # [B, V, 3, 3]
    context_depths: torch.Tensor,      # [B, V, H, W]
    image_shape: tuple[int, int],
    min_object_votes: int,
    chunk_size: int,
    depth_abs_tolerance: float,
    depth_rel_tolerance: float,
) -> tuple[torch.Tensor, dict]:
    """Union direct fullmask removal with depth-consistent multi-view object voting."""
    B, N = direct_object_mask.shape
    V = context_mask.shape[1]
    object_votes = torch.zeros(B, N, dtype=torch.int16, device=gaussians_means.device)
    visible_votes = torch.zeros(B, N, dtype=torch.int16, device=gaussians_means.device)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        grid, z, valid = project_gaussians_to_context(
            gaussians_means[:, start:end],
            extrinsics,
            intrinsics,
            image_shape,
        )
        sampled = F.grid_sample(
            rearrange(context_mask.float(), "b v h w -> (b v) () h w"),
            rearrange(grid, "b v n xy -> (b v) n () xy"),
            mode="nearest",
            align_corners=True,
            padding_mode="zeros",
        )
        sampled = rearrange(sampled.squeeze(1).squeeze(-1), "(b v) n -> b v n", b=B, v=V) > 0.5
        sampled_depth = F.grid_sample(
            rearrange(context_depths.float(), "b v h w -> (b v) () h w"),
            rearrange(grid, "b v n xy -> (b v) n () xy"),
            mode="nearest",
            align_corners=True,
            padding_mode="zeros",
        )
        sampled_depth = rearrange(sampled_depth.squeeze(1).squeeze(-1), "(b v) n -> b v n", b=B, v=V)
        depth_tolerance = torch.maximum(
            torch.full_like(sampled_depth, depth_abs_tolerance),
            sampled_depth.abs() * depth_rel_tolerance,
        )
        same_surface = valid & (sampled_depth > 0) & ((z - sampled_depth).abs() <= depth_tolerance)
        object_votes[:, start:end] = (sampled & same_surface).sum(dim=1).to(torch.int16)
        visible_votes[:, start:end] = same_surface.sum(dim=1).to(torch.int16)

    strong_object_mask = object_votes >= min_object_votes
    object_mask = direct_object_mask | strong_object_mask
    details = {
        "min_object_votes": min_object_votes,
        "projection_vote_chunk": chunk_size,
        "depth_abs_tolerance": depth_abs_tolerance,
        "depth_rel_tolerance": depth_rel_tolerance,
        "num_direct_removed": int(direct_object_mask.sum().item()),
        "num_strong_object_vote": int(strong_object_mask.sum().item()),
        "num_union_removed": int(object_mask.sum().item()),
        "max_object_votes": int(object_votes.max().item()),
        "mean_object_votes_removed": float(object_votes[object_mask].float().mean().item()) if object_mask.any() else 0.0,
        "mean_visible_votes": float(visible_votes.float().mean().item()),
    }
    return object_mask, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2a: removal using all 16 context view masks."
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Stage 1 fine-tuned checkpoint (.ckpt)")
    parser.add_argument("--output_dir", type=Path,
                        default=ROOT / "outputs/removal/bear_fullmask")
    parser.add_argument("--datasets_root", type=Path,
                        default=ROOT / "datasets/bear_re10k_like")
    parser.add_argument("--mask_dir", type=Path,
                        default=Path("/home/junho/jieun/3dgic_pixel_aligned_vggt/data/bear/bear/inpaint_object_mask_255"))
    parser.add_argument("--context", type=int, nargs="+", default=CTX_VIEWS)
    parser.add_argument("--target", type=int, nargs="+", default=TGT_VIEWS)
    parser.add_argument("--image_shape", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--consistency_votes", type=int, default=2,
                        help="Also remove Gaussians projected into object masks from at least this many context views. Set 0 to disable.")
    parser.add_argument("--vote_chunk", type=int, default=131072)
    parser.add_argument("--vote_depth_abs_tolerance", type=float, default=0.15)
    parser.add_argument("--vote_depth_rel_tolerance", type=float, default=0.20)
    parser.add_argument("--render_chunk", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_shape = tuple(args.image_shape)
    H, W = image_shape
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    set_cfg(OmegaConf.create({
        "dataset": {"view_sampler": {
            "num_context_views": len(args.context),
            "num_target_views": len(args.target),
        }},
        "output_dir": str(args.output_dir),
    }))

    encoder_cfg = make_encoder_cfg()
    dataset_cfg = make_dataset_cfg(args.datasets_root, args.context, args.target, args.image_shape)
    batch = build_batch(dataset_cfg, device)
    encoder, decoder = build_models(encoder_cfg, dataset_cfg, device)
    load_encoder_weights(encoder, args.checkpoint)
    batch = encoder.get_data_shim()(batch)

    context_indices = batch["context"]["index"][0].detach().cpu().tolist()
    target_indices  = batch["target"]["index"][0].detach().cpu().tolist()

    # --- load masks for all 16 context views ---
    print(f"Loading masks for {len(context_indices)} context views...")
    context_mask = load_masks(args.mask_dir, context_indices, image_shape, args.threshold)
    context_mask = context_mask[None].to(device)  # [1, V, H, W]
    mask_ratios = context_mask[0].float().mean(dim=(-2, -1)).tolist()
    for idx, ratio in zip(context_indices, mask_ratios):
        print(f"  frame {idx+1:03d}: {ratio*100:.1f}% masked")

    # --- encode ---
    print("Encoding (16 views)...")
    with torch.no_grad():
        encoded = encoder(batch["context"], 0, deterministic=True)
        gaussians = encoded["gaussians"] if isinstance(encoded, dict) else encoded
    N = gaussians.means.shape[1]
    print(f"Gaussians: {N:,}  ({len(context_indices)}×{H}×{W})")

    # --- mask ---
    direct_obj_mask = build_object_mask(context_mask, encoder_cfg.num_surfaces, encoder_cfg.gaussians_per_pixel)
    consistency_details = {}
    if args.consistency_votes > 0:
        print(
            "Applying multi-view mask consistency voting "
            f"(object_votes>={args.consistency_votes})..."
        )
        obj_mask, consistency_details = build_consistency_object_mask(
            direct_obj_mask,
            gaussians.means,
            context_mask,
            batch["context"]["extrinsics"],
            batch["context"]["intrinsics"],
            encoded["depths"],
            image_shape,
            args.consistency_votes,
            args.vote_chunk,
            args.vote_depth_abs_tolerance,
            args.vote_depth_rel_tolerance,
        )
        print(f"  Direct mask removal: {consistency_details['num_direct_removed']:,}")
        print(f"  Vote mask removal:   {consistency_details['num_strong_object_vote']:,}")
        print(f"  Union removal:       {consistency_details['num_union_removed']:,}")
    else:
        obj_mask = direct_obj_mask
    keep_mask = ~obj_mask
    n_removed = int(obj_mask.sum())
    print(f"Removing {n_removed:,} / {N:,}  ({n_removed/N*100:.1f}%)")

    gaussians_removed = apply_opacity_mask(gaussians, keep_mask)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(keep_mask.cpu(), args.output_dir / "keep_mask.pt")

    # --- save encoder depth (context views, before removal) ---
    # encoded["depths"]: [B, V, H, W] — cost-volume predicted depth per context pixel
    if isinstance(encoded, dict) and "depths" in encoded:
        enc_depths = encoded["depths"][0].detach().cpu()  # [V, H, W]
        save_encoder_depths(enc_depths, context_indices, args.output_dir)
        print(f"Saved encoder depth for {len(context_indices)} context views → depth_encoder/")

    # --- render target views (after removal) ---
    print(f"Rendering {len(target_indices)} views (chunk={args.render_chunk})...")
    with torch.no_grad():
        colors, rendered_depths = render_views_chunked(
            decoder, gaussians_removed,
            batch["target"]["extrinsics"], batch["target"]["intrinsics"],
            batch["target"]["near"], batch["target"]["far"],
            image_shape, args.render_chunk,
        )
    save_renders(colors, rendered_depths, target_indices, args.output_dir)
    print(f"Saved rendered depth for {len(target_indices)} target views → depth_rendered/")

    stats = {
        "method": "fullmask_consistency" if args.consistency_votes > 0 else "fullmask",
        "checkpoint": str(args.checkpoint),
        "scene": batch["scene"][0],
        "context_indices": context_indices,
        "target_indices": target_indices,
        "num_gaussians_total": N,
        "num_gaussians_removed": n_removed,
        "removal_ratio": round(n_removed / N, 4),
        "context_mask_ratios": {str(idx+1): round(r, 4) for idx, r in zip(context_indices, mask_ratios)},
        "consistency_details": consistency_details,
    }
    with (args.output_dir / "removal_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
