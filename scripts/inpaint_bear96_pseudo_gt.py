#!/usr/bin/env python3
"""Inpaint bear96 with pseudo-GT, L_warp, and optional VGGT depth supervision.

Loss design (from spec Section 3.2):
  - ref view  (mask region) : L1(render, I_ref_inpainted)          — direct color GT
  - ref view  (bg region)   : L1(render, I_orig)
  - other views (mask region): L_warp via depth-warp of I_ref_inpainted
  - other views (bg region)  : L1(render, I_orig)
  - context views            : optional L1(log rendered depth, log VGGT depth)

Parameterization:
  - background Gaussians are frozen
  - inpaint Gaussians optimize depth-on-source-ray, color harmonics, and opacity
  - xyz means are recomputed from the source context pixel ray every iteration
  - covariances are frozen to preserve the pretrained pixel-aligned geometry prior
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from PIL import Image
import lpips

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from reconstruct_bear96_4view import (  # noqa: E402
    build_example,
    colorize_depth,
    evenly_spaced_indices,
    load_encoder_decoder,
    make_cfg,
    move_views_to_device,
)

from jaxtyping import install_import_hook  # noqa: E402

with install_import_hook(("src",), ("beartype", "beartype")):
    from src.misc.image_io import save_image
    from src.model.types import Gaussians


BEAR_DATA = Path("/home/junho/jieun/3dgic_pixel_aligned_vggt/data/bear/bear")
TO_TENSOR = tf.ToTensor()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def resize_crop_like_mvsplat(
    tensor: torch.Tensor,
    image_shape: tuple[int, int],
    mode: str,
) -> torch.Tensor:
    """Match MVSplat's rescale-then-center-crop geometry for auxiliary images."""
    h_out, w_out = image_shape
    _, h_in, w_in = tensor.shape
    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    kwargs = {"size": (h_scaled, w_scaled), "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    tensor = F.interpolate(tensor[None], **kwargs)[0]
    row = (h_scaled - h_out) // 2
    col = (w_scaled - w_out) // 2
    return tensor[:, row : row + h_out, col : col + w_out]


def load_rgb_dir(
    image_dir: Path,
    num_views: int,
    image_shape: tuple[int, int],
    original_indices: list[int] | None = None,
) -> torch.Tensor:
    imgs = []
    for i in range(num_views):
        original_i = original_indices[i] if original_indices is not None else i
        img = TO_TENSOR(Image.open(image_dir / f"frame_{original_i + 1:05d}.jpg").convert("RGB"))[:3]
        img = resize_crop_like_mvsplat(img, image_shape, mode="bilinear")
        imgs.append(img)
    return torch.stack(imgs)  # [N, 3, H, W]


def load_masks_dir(
    mask_dir: Path,
    num_views: int,
    image_shape: tuple[int, int],
    dilation: int,
    original_indices: list[int] | None = None,
) -> torch.Tensor:
    masks = []
    for i in range(num_views):
        original_i = original_indices[i] if original_indices is not None else i
        m = TO_TENSOR(Image.open(mask_dir / f"frame_{original_i + 1:05d}.png").convert("L"))[:1]
        m = resize_crop_like_mvsplat(m, image_shape, mode="nearest")[0]
        if dilation > 0:
            k = 2 * dilation + 1
            m = F.max_pool2d(m[None, None], kernel_size=k, stride=1, padding=dilation)[0, 0]
        masks.append(m > 0.5)
    return torch.stack(masks)  # [N, H, W] bool


def load_vggt_depth_dir(
    depth_dir: Path,
    num_views: int,
    image_shape: tuple[int, int],
    original_indices: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load VGGT depth/conf maps by original frame name.

    Missing files are allowed because a context-only VGGT run may only produce
    maps for the sparse context views.
    """
    depths = []
    confs = []
    available = []
    for i in range(num_views):
        original_i = original_indices[i] if original_indices is not None else i
        stem = f"frame_{original_i + 1:05d}"
        depth_path = depth_dir / f"{stem}_depth.npy"
        conf_path = depth_dir / f"{stem}_depth_conf.npy"
        if not depth_path.exists():
            depths.append(torch.full(image_shape, float("nan")))
            confs.append(torch.zeros(image_shape))
            available.append(False)
            continue

        depth = torch.from_numpy(np.load(depth_path)).float()[None]
        depth = resize_crop_like_mvsplat(depth, image_shape, mode="bilinear")[0]

        if conf_path.exists():
            conf = torch.from_numpy(np.load(conf_path)).float()[None]
            conf = resize_crop_like_mvsplat(conf, image_shape, mode="bilinear")[0]
        else:
            conf = torch.ones_like(depth)

        depths.append(depth)
        confs.append(conf)
        available.append(True)

    return torch.stack(depths), torch.stack(confs), torch.tensor(available, dtype=torch.bool)


def load_gaussians(path: Path, device: torch.device) -> Gaussians:
    data = torch.load(path, map_location=device)
    return Gaussians(
        means=data["means"].to(device),
        covariances=data["covariances"].to(device),
        harmonics=data["harmonics"].to(device),
        opacities=data["opacities"].to(device),
    )


def build_source_rays(
    context: dict,
    remove_mask: torch.Tensor,
    image_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return source context ray origins/directions for selected pixel-aligned GS.

    MVSplat emits one Gaussian per context-view pixel, flattened as
    [context_view, y, x]. Removal masks use the same flattened indexing.
    """
    h, w = image_shape
    num_context = context["image"].shape[1]
    expected = num_context * h * w
    if remove_mask.numel() != expected:
        raise ValueError(
            f"remove_mask has {remove_mask.numel()} entries, expected {expected} "
            f"for {num_context} context views at {h}x{w}."
        )

    device = remove_mask.device
    selected = torch.arange(expected, device=device)[remove_mask.reshape(-1)]
    source_view = selected // (h * w)
    pixel = selected % (h * w)
    y = pixel // w
    x = pixel % w

    intrinsics = context["intrinsics"][0, source_view]  # [N, 3, 3]
    c2w = context["extrinsics"][0, source_view]         # [N, 4, 4]
    x_n = (x.float() + 0.5) / w
    y_n = (y.float() + 0.5) / h
    dirs_cam = torch.stack(
        [
            (x_n - intrinsics[:, 0, 2]) / intrinsics[:, 0, 0],
            (y_n - intrinsics[:, 1, 2]) / intrinsics[:, 1, 1],
            torch.ones_like(x_n),
        ],
        dim=-1,
    )
    rotation = c2w[:, :3, :3]
    origins = c2w[:, :3, 3]
    directions = torch.bmm(dirs_cam[:, None, :], rotation.transpose(1, 2)).squeeze(1)
    directions = F.normalize(directions, dim=-1)
    return origins, directions, source_view, pixel


def depths_from_means(means: torch.Tensor, origins: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Project initial GS means onto their source rays to initialize depth."""
    return ((means - origins) * directions).sum(dim=-1).clamp_min(1e-4)


def means_from_depths(
    origins: torch.Tensor,
    directions: torch.Tensor,
    log_depths: torch.Tensor,
) -> torch.Tensor:
    depths = log_depths.exp()
    return origins + depths[..., None] * directions


# ---------------------------------------------------------------------------
# L_warp: depth-warp I_ref_inpainted → target view, then L1 vs render
#
# Intrinsics convention (MVSplat): normalized 3×3
#   K[0,0]=fx/W, K[1,1]=fy/H, K[0,2]=cx/W, K[1,2]=cy/H
# Extrinsics convention (MVSplat): c2w (camera-to-world) 4×4
# ---------------------------------------------------------------------------

def warp_ref_to_target(
    depth_t: torch.Tensor,    # [H, W]  rendered depth in target view
    I_ref: torch.Tensor,      # [3, H, W]  reference inpainted image
    K_t: torch.Tensor,        # [3, 3]  target intrinsics (normalized)
    c2w_t: torch.Tensor,      # [4, 4]  target extrinsics (c2w)
    K_ref: torch.Tensor,      # [3, 3]  ref intrinsics (normalized)
    c2w_ref: torch.Tensor,    # [4, 4]  ref extrinsics (c2w)
) -> torch.Tensor:             # [3, H, W]  I_ref warped to target view
    H, W = depth_t.shape
    device = depth_t.device

    # pixel grid (row = y, col = x), normalized to [0, 1]
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    x_n = (xs.float() + 0.5) / W  # [H, W]
    y_n = (ys.float() + 0.5) / H

    # target pixel → 3D point in target camera space
    x_cam = (x_n - K_t[0, 2]) / K_t[0, 0] * depth_t  # [H, W]
    y_cam = (y_n - K_t[1, 2]) / K_t[1, 1] * depth_t
    pts_cam_t = torch.stack([x_cam, y_cam, depth_t], dim=-1).reshape(-1, 3)  # [H*W, 3]

    # target camera → world
    R_t = c2w_t[:3, :3]
    t_t = c2w_t[:3, 3]
    pts_world = pts_cam_t @ R_t.T + t_t  # [H*W, 3]

    # world → ref camera
    w2c_ref = c2w_ref.inverse()
    R_ref_inv = w2c_ref[:3, :3]
    t_ref_inv = w2c_ref[:3, 3]
    pts_cam_ref = pts_world @ R_ref_inv.T + t_ref_inv  # [H*W, 3]

    # ref camera → normalized image coords [0, 1]
    z_ref = pts_cam_ref[:, 2].clamp_min(1e-6)
    u_ref = pts_cam_ref[:, 0] / z_ref * K_ref[0, 0] + K_ref[0, 2]  # [H*W]
    v_ref = pts_cam_ref[:, 1] / z_ref * K_ref[1, 1] + K_ref[1, 2]

    # to [-1, 1] for grid_sample
    grid = torch.stack([u_ref * 2 - 1, v_ref * 2 - 1], dim=-1).reshape(1, H, W, 2)
    I_warp = F.grid_sample(
        I_ref[None], grid, mode="bilinear", align_corners=False, padding_mode="zeros"
    )[0]  # [3, H, W]

    # valid mask: ref pixel in-bounds and depth > 0
    valid = (
        (u_ref >= 0) & (u_ref <= 1) &
        (v_ref >= 0) & (v_ref <= 1) &
        (pts_cam_ref[:, 2] > 0)
    ).reshape(H, W)

    return I_warp, valid  # [3,H,W], [H,W] bool


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def masked_l1(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """pred/gt: [3, H, W], mask: [H, W] bool."""
    denom = mask.sum().clamp_min(1.0) * 3
    return (pred - gt).abs()[:, mask].sum() / denom


def masked_log_depth_l1(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    conf: torch.Tensor,
    mask: torch.Tensor,
    near: torch.Tensor,
    far: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    valid = (
        mask
        & torch.isfinite(pred_depth)
        & torch.isfinite(gt_depth)
        & (pred_depth > eps)
        & (gt_depth > eps)
    )
    if not valid.any():
        return pred_depth.mean() * 0

    pred = pred_depth.clamp_min(eps).minimum(far).maximum(near)
    gt = gt_depth.clamp_min(eps).minimum(far).maximum(near)
    denom = (far.log() - near.log()).clamp_min(eps)
    pred = (pred.log() - near.log()) / denom
    gt = (gt.log() - near.log()) / denom

    weight = valid.float()
    finite_conf = conf[torch.isfinite(conf)]
    if finite_conf.numel() > 0:
        scale = torch.quantile(finite_conf.float(), 0.8).clamp_min(eps)
        weight = weight * (conf / scale).clamp(0.0, 1.0)

    return ((pred - gt).abs() * weight).sum() / weight.sum().clamp_min(1.0)


def mask_to_bbox(mask: torch.Tensor, padding: int, h: int, w: int) -> tuple[int, int, int, int] | None:
    ys, xs = torch.where(mask)
    if ys.numel() == 0:
        return None
    y0 = max(int(ys.min().item()) - padding, 0)
    y1 = min(int(ys.max().item()) + padding + 1, h)
    x0 = max(int(xs.min().item()) - padding, 0)
    x1 = min(int(xs.max().item()) + padding + 1, w)
    return y0, y1, x0, x1


def crop_for_lpips(image: torch.Tensor, bbox: tuple[int, int, int, int], min_size: int) -> torch.Tensor:
    y0, y1, x0, x1 = bbox
    crop = image[:, y0:y1, x0:x1]
    h, w = crop.shape[-2:]
    pad_h = max(min_size - h, 0)
    pad_w = max(min_size - w, 0)
    if pad_h > 0 or pad_w > 0:
        crop = F.pad(
            crop,
            (
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
            ),
            mode="replicate",
        )
    return crop


def lpips_crop_loss(
    lpips_model: torch.nn.Module,
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    padding: int,
    min_size: int,
) -> torch.Tensor:
    bbox = mask_to_bbox(mask, padding, pred.shape[-2], pred.shape[-1])
    if bbox is None:
        return pred.mean() * 0
    pred_crop = crop_for_lpips(pred, bbox, min_size)
    target_crop = crop_for_lpips(target, bbox, min_size)
    return lpips_model(pred_crop[None] * 2 - 1, target_crop[None] * 2 - 1).mean()


def sample_target(target: dict, indices: torch.Tensor) -> dict:
    return {k: (v[:, indices] if torch.is_tensor(v) and v.ndim >= 2 else v) for k, v in target.items()}


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_renders(decoder, gaussians, target, image_shape, output_dir, render_chunk, save_depth):
    output_dir.mkdir(parents=True, exist_ok=True)
    num_views = target["image"].shape[1]
    for start in range(0, num_views, render_chunk):
        end = min(start + render_chunk, num_views)
        sl = sample_target(target, torch.arange(start, end, device=target["image"].device))
        out = decoder.forward(
            gaussians,
            sl["extrinsics"], sl["intrinsics"], sl["near"], sl["far"],
            image_shape,
            depth_mode="depth" if save_depth else None,
        )
        for local_i, view_i in enumerate(range(start, end)):
            save_image(out.color[0, local_i], output_dir / "color" / f"{view_i:06d}.png")
            save_image(target["image"][0, view_i], output_dir / "gt" / f"{view_i:06d}.png")
            if save_depth and out.depth is not None:
                depth = out.depth[0, local_i]
                (output_dir / "depth_raw").mkdir(parents=True, exist_ok=True)
                torch.save(depth.detach().cpu(), output_dir / "depth_raw" / f"{view_i:06d}.pt")
                save_image(colorize_depth(depth), output_dir / "depth_rendered" / f"{view_i:06d}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, default=Path("/home/junho/jieun/mvsplat/datasets/bear96_colmap"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/junho/jieun/mvsplat/checkpoints/re10k.ckpt"))
    parser.add_argument("--gaussians", type=Path, default=Path("/home/junho/jieun/mvsplat/outputs/bear96_removal_lr2e-5/gaussians_removed_opacity.pt"))
    parser.add_argument("--remove_mask", type=Path, default=Path("/home/junho/jieun/mvsplat/outputs/bear96_removal_lr2e-5/remove_mask.pt"))
    parser.add_argument("--pseudo_dir", type=Path, default=BEAR_DATA / "images_inpaint_unseen")
    parser.add_argument("--mask_dir", type=Path, default=BEAR_DATA / "inpaint_object_mask_255")
    parser.add_argument("--output_dir", type=Path, default=Path("/home/junho/jieun/mvsplat/outputs/bear96_inpaint_pseudo_gt"))
    # Reference view: one view whose LaMa result is used as the primary color anchor.
    # Other views are supervised via L_warp (depth-warp I_ref_inpainted → target view).
    parser.add_argument("--ref_view", type=int, default=32, help="0-based index of the reference view (LaMa anchor)")
    parser.add_argument("--num_context", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--target_batch", type=int, default=4, help="Non-ref views sampled per step; ref view always included")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_opacity", type=float, default=5e-2)
    parser.add_argument("--lambda_outside", type=float, default=0.3)
    parser.add_argument("--lambda_warp", type=float, default=1.0)
    parser.add_argument("--warp_l1_weight", type=float, default=0.2)
    parser.add_argument("--warp_lpips_weight", type=float, default=0.5)
    parser.add_argument("--warp_lpips_padding", type=int, default=0)
    parser.add_argument("--warp_lpips_min_size", type=int, default=64)
    parser.add_argument("--lambda_ref", type=float, default=2.0, help="Weight for direct L1 on ref view mask")
    parser.add_argument("--lambda_vggt_depth", type=float, default=0.1)
    parser.add_argument(
        "--vggt_depth_dir",
        type=Path,
        default=BEAR_DATA / "vggt_depth_inpaint_unseen",
        help="Directory containing frame_XXXXX_depth.npy/conf.npy from VGGT inference.",
    )
    parser.add_argument(
        "--vggt_depth_mask_mode",
        choices=("all", "masked", "visible"),
        default="masked",
        help="Where to apply VGGT depth loss for views with available VGGT maps.",
    )
    parser.add_argument("--opacity_init", type=float, default=0.05)
    parser.add_argument("--render_chunk", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--image_width", type=int, default=256)
    parser.add_argument("--near", type=float, default=1.0)
    parser.add_argument("--far", type=float, default=100.0)
    parser.add_argument("--mask_dilation", type=int, default=1)
    parser.add_argument("--save_depth", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_shape = (args.image_height, args.image_width)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset (original images + cameras)
    chunk_path = args.dataset_root / "test" / "000000.torch"
    scene = torch.load(chunk_path, map_location="cpu")[0]
    num_views = len(scene["images"])
    original_indices_tensor = scene.get("original_indices")
    original_indices = (
        [int(i) for i in original_indices_tensor.tolist()]
        if original_indices_tensor is not None
        else None
    )
    context_indices = evenly_spaced_indices(num_views, args.num_context)
    example = build_example(chunk_path, context_indices, image_shape, args.near, args.far)
    context = move_views_to_device(example["context"], device)
    target = move_views_to_device(example["target"], device)

    # Decoder (frozen)
    cfg = make_cfg(args.num_context, image_shape, args.checkpoint, args.output_dir)
    _, decoder = load_encoder_decoder(cfg, args.checkpoint, device)
    decoder.eval()

    # Gaussians + remove_mask
    gaussians = load_gaussians(args.gaussians, device)
    remove_mask = torch.load(args.remove_mask, map_location=device).bool().reshape(-1)
    bg_mask = ~remove_mask

    # Split: bg frozen, inpaint trainable. Inpaint means remain pixel-aligned:
    # source context pixel ray + optimized depth -> xyz.
    bg_means      = gaussians.means[:, bg_mask].detach()
    bg_covariances = gaussians.covariances[:, bg_mask].detach()
    bg_harmonics  = gaussians.harmonics[:, bg_mask].detach()
    bg_opacities  = gaussians.opacities[:, bg_mask].detach()

    source_origins, source_dirs, source_view_ids, source_pixels = build_source_rays(
        context, remove_mask, image_shape
    )
    init_means = gaussians.means[:, remove_mask].clone().squeeze(0)
    init_depths = depths_from_means(init_means, source_origins, source_dirs).clamp(args.near, args.far)
    inp_log_depths = torch.nn.Parameter(init_depths.log())
    inp_covariances = gaussians.covariances[:, remove_mask].detach()
    inp_harmonics  = torch.nn.Parameter(gaussians.harmonics[:, remove_mask].clone())
    inp_opacities  = torch.nn.Parameter(
        gaussians.opacities[:, remove_mask].clone().clamp_min(0.0) + args.opacity_init
    )

    optimizer = torch.optim.Adam([
        {"params": [inp_log_depths, inp_harmonics], "lr": args.lr},
        {"params": [inp_opacities], "lr": args.lr_opacity},
    ])
    lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
    for param in lpips_model.parameters():
        param.requires_grad_(False)

    # Load images/masks/depth priors (CPU)
    print("Loading pseudo-GT, original GT, masks, VGGT depth...")
    pseudo_gt   = load_rgb_dir(args.pseudo_dir, num_views, image_shape, original_indices)
    masks = load_masks_dir(args.mask_dir, num_views, image_shape, args.mask_dilation, original_indices)
    original_gt = target["image"][0].detach().cpu()  # [96, 3, H, W]
    vggt_depth, vggt_conf, vggt_available = load_vggt_depth_dir(
        args.vggt_depth_dir, num_views, image_shape, original_indices
    )

    # Camera params for ref view (stay on CPU, moved to device per step)
    ref_idx = args.ref_view
    I_ref_inpainted = pseudo_gt[ref_idx].to(device)     # [3, H, W]
    K_ref  = target["intrinsics"][0, ref_idx].cpu()     # [3, 3]
    c2w_ref = target["extrinsics"][0, ref_idx].cpu()    # [4, 4]

    non_ref_indices = [i for i in range(num_views) if i != ref_idx]

    print(
        f"Inpaint Gaussians: {remove_mask.sum().item()}, bg: {bg_mask.sum().item()}, "
        f"ref_view: {ref_idx}"
    )

    metadata = {
        "gaussians": str(args.gaussians),
        "pseudo_dir": str(args.pseudo_dir),
        "num_views": num_views,
        "original_indices_zero_based": original_indices,
        "context_indices": context_indices,
        "context_original_indices_zero_based": (
            [original_indices[i] for i in context_indices]
            if original_indices is not None
            else context_indices
        ),
        "ref_view": ref_idx,
        "ref_original_index_zero_based": (
            original_indices[ref_idx] if original_indices is not None else ref_idx
        ),
        "steps": args.steps,
        "target_batch": args.target_batch,
        "lr": args.lr,
        "lr_opacity": args.lr_opacity,
        "lambda_outside": args.lambda_outside,
        "lambda_warp": args.lambda_warp,
        "warp_l1_weight": args.warp_l1_weight,
        "warp_lpips_weight": args.warp_lpips_weight,
        "warp_lpips_padding": args.warp_lpips_padding,
        "warp_lpips_min_size": args.warp_lpips_min_size,
        "lambda_ref": args.lambda_ref,
        "lambda_vggt_depth": args.lambda_vggt_depth,
        "vggt_depth_dir": str(args.vggt_depth_dir),
        "vggt_depth_mask_mode": args.vggt_depth_mask_mode,
        "vggt_depth_available_views": [
            i for i, available in enumerate(vggt_available.tolist()) if available
        ],
        "vggt_depth_loss_views": "sampled_batch_all_available_views",
        "opacity_init": args.opacity_init,
        "parameterization": "source_ray_depth_plus_color_opacity",
        "covariances": "frozen",
        "n_inpaint_gaussians": int(remove_mask.sum().item()),
        "init_depth_min": float(init_depths.min().item()) if init_depths.numel() else 0.0,
        "init_depth_mean": float(init_depths.mean().item()) if init_depths.numel() else 0.0,
        "init_depth_max": float(init_depths.max().item()) if init_depths.numel() else 0.0,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    log_path = args.output_dir / "train_loss.log"

    H, W = image_shape

    for step in range(1, args.steps + 1):
        # Always include ref view; sample target_batch additional non-ref views
        sampled_nonref = torch.randperm(len(non_ref_indices))[: args.target_batch].tolist()
        batch_indices = torch.tensor(
            [ref_idx] + [non_ref_indices[i] for i in sampled_nonref],
            device=device,
        )  # [1 + target_batch]

        inp_means = means_from_depths(source_origins, source_dirs, inp_log_depths).unsqueeze(0)

        # Merge Gaussians
        merged = Gaussians(
            means=torch.cat([bg_means, inp_means], dim=1),
            covariances=torch.cat([bg_covariances, inp_covariances], dim=1),
            harmonics=torch.cat([bg_harmonics, inp_harmonics], dim=1),
            opacities=torch.cat([bg_opacities, inp_opacities.clamp(0.0, 1.0)], dim=1),
        )

        # Render RGB + depth (depth needed for L_warp)
        sl = sample_target(target, batch_indices)
        out = decoder.forward(
            merged,
            sl["extrinsics"], sl["intrinsics"], sl["near"], sl["far"],
            image_shape,
            depth_mode="depth",
        )
        pred  = out.color  # [1, B, 3, H, W]
        depth = out.depth  # [1, B, H, W]

        # --- Loss computation per view in batch ---
        loss = torch.zeros((), device=device)
        loss_ref_acc   = torch.zeros((), device=device)
        loss_warp_acc  = torch.zeros((), device=device)
        loss_bg_acc    = torch.zeros((), device=device)
        loss_vggt_depth_acc = torch.zeros((), device=device)

        cpu_idx = batch_indices.cpu()
        for local_i, view_i in enumerate(cpu_idx.tolist()):
            pred_v  = pred[0, local_i]   # [3, H, W]
            orig_v  = original_gt[view_i].to(device)
            mask_v  = masks[view_i].to(device)    # [H, W] bool

            # background (outside mask): preserve original
            bg_loss = masked_l1(pred_v, orig_v, ~mask_v)
            loss_bg_acc = loss_bg_acc + bg_loss
            loss = loss + args.lambda_outside * bg_loss

            if view_i == ref_idx:
                # ref view mask: direct L1 to I_ref_inpainted
                ref_loss = masked_l1(pred_v, I_ref_inpainted, mask_v)
                loss_ref_acc = loss_ref_acc + ref_loss
                loss = loss + args.lambda_ref * ref_loss
            else:
                # non-ref views: L_warp (depth-warp I_ref to target view)
                depth_v = depth[0, local_i]  # [H, W]
                K_t   = target["intrinsics"][0, view_i]
                c2w_t = target["extrinsics"][0, view_i]
                I_warp, valid = warp_ref_to_target(
                    depth_v, I_ref_inpainted,
                    K_t, c2w_t,
                    K_ref.to(device), c2w_ref.to(device),
                )
                warp_mask = mask_v & valid
                if warp_mask.any():
                    pseudo_v = pred_v.detach().clone()
                    pseudo_v[:, warp_mask] = I_warp.detach()[:, warp_mask]
                    warp_l1 = masked_l1(pred_v, I_warp.detach(), warp_mask)
                    warp_lpips = lpips_crop_loss(
                        lpips_model,
                        pred_v,
                        pseudo_v,
                        warp_mask,
                        args.warp_lpips_padding,
                        args.warp_lpips_min_size,
                    )
                    warp_loss = (
                        args.warp_l1_weight * warp_l1
                        + args.warp_lpips_weight * warp_lpips
                    )
                    loss_warp_acc = loss_warp_acc + warp_loss
                    loss = loss + args.lambda_warp * warp_loss

            if (
                args.lambda_vggt_depth > 0
                and bool(vggt_available[view_i])
            ):
                depth_mask = torch.ones_like(mask_v)
                if args.vggt_depth_mask_mode == "masked":
                    depth_mask = mask_v
                elif args.vggt_depth_mask_mode == "visible":
                    depth_mask = ~mask_v

                depth_loss = masked_log_depth_l1(
                    depth[0, local_i],
                    vggt_depth[view_i].to(device),
                    vggt_conf[view_i].to(device),
                    depth_mask,
                    sl["near"][0, local_i, None, None],
                    sl["far"][0, local_i, None, None],
                )
                loss_vggt_depth_acc = loss_vggt_depth_acc + depth_loss
                loss = loss + args.lambda_vggt_depth * depth_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([inp_log_depths, inp_harmonics, inp_opacities], 1.0)
        optimizer.step()

        with torch.no_grad():
            inp_opacities.clamp_(0.0, 1.0)
            inp_log_depths.clamp_(
                min=torch.log(torch.tensor(args.near, device=device)),
                max=torch.log(torch.tensor(args.far, device=device)),
            )

        if step == 1 or step % 50 == 0:
            line = (
                f"step {step:06d} "
                f"loss={loss.item():.5f} "
                f"ref={loss_ref_acc.item():.5f} "
                f"warp={loss_warp_acc.item():.5f} "
                f"bg={loss_bg_acc.item():.5f} "
                f"vggt_depth={loss_vggt_depth_acc.item():.5f} "
                f"depth_mean={inp_log_depths.exp().mean().item():.4f} "
                f"opacity_mean={inp_opacities.mean().item():.4f}"
            )
            print(line, flush=True)
            with log_path.open("a") as f:
                f.write(line + "\n")

        if step % args.save_every == 0 or step == args.steps:
            with torch.no_grad():
                inp_means_eval = means_from_depths(source_origins, source_dirs, inp_log_depths).unsqueeze(0)
                merged_eval = Gaussians(
                    means=torch.cat([bg_means, inp_means_eval], dim=1),
                    covariances=torch.cat([bg_covariances, inp_covariances], dim=1),
                    harmonics=torch.cat([bg_harmonics, inp_harmonics], dim=1),
                    opacities=torch.cat([bg_opacities, inp_opacities.clamp(0.0, 1.0)], dim=1),
                )
                save_renders(
                    decoder, merged_eval, target, image_shape,
                    args.output_dir / f"render_step_{step:06d}",
                    args.render_chunk, args.save_depth,
                )

    # Save final Gaussians
    final_inp_means = means_from_depths(source_origins, source_dirs, inp_log_depths).unsqueeze(0)
    torch.save(
        {
            "means": torch.cat([bg_means, final_inp_means], dim=1).detach().cpu(),
            "covariances": torch.cat([bg_covariances, inp_covariances], dim=1).detach().cpu(),
            "harmonics": torch.cat([bg_harmonics, inp_harmonics], dim=1).detach().cpu(),
            "opacities": torch.cat([bg_opacities, inp_opacities.clamp(0.0, 1.0)], dim=1).detach().cpu(),
            "inpaint_depths": inp_log_depths.exp().detach().cpu(),
            "inpaint_source_context_view": source_view_ids.detach().cpu(),
            "inpaint_source_pixel": source_pixels.detach().cpu(),
            "metadata": metadata,
        },
        args.output_dir / "gaussians_inpainted.pt",
    )
    print(f"Done. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
