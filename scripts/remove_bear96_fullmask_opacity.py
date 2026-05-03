#!/usr/bin/env python3
"""Lower opacities of MVSplat Gaussians whose projected means fall inside bear masks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from PIL import Image

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
    from src.dataset.shims.crop_shim import rescale_and_crop
    from src.misc.image_io import save_image
    from src.model.types import Gaussians


def load_masks(
    mask_root: Path,
    image_shape: tuple[int, int],
    device: torch.device,
    dilation: int,
    selected_indices: list[int] | None = None,
) -> torch.Tensor:
    paths = sorted(mask_root.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No mask png files found in {mask_root}")
    if selected_indices is not None:
        paths = [paths[i] for i in selected_indices]
    masks = []
    to_tensor = tf.ToTensor()
    dummy_intr = torch.eye(3).repeat(len(paths), 1, 1)
    for path in paths:
        mask = to_tensor(Image.open(path).convert("L"))[:1].repeat(3, 1, 1)
        masks.append(mask)
    masks = torch.stack(masks)
    masks, _ = rescale_and_crop(masks, dummy_intr, image_shape)
    masks = (masks[:, 0] > 0.5).float().to(device)
    if dilation > 0:
        k = 2 * dilation + 1
        masks = F.max_pool2d(masks[:, None], kernel_size=k, stride=1, padding=dilation)[:, 0]
    return masks.bool()


def load_gaussians(path: Path, device: torch.device) -> Gaussians:
    data = torch.load(path, map_location=device)
    return Gaussians(
        means=data["means"].to(device),
        covariances=data["covariances"].to(device),
        harmonics=data["harmonics"].to(device),
        opacities=data["opacities"].to(device),
    )


def sample_targets(target: dict, indices: torch.Tensor) -> dict:
    return {
        k: (v[:, indices] if torch.is_tensor(v) and v.ndim >= 2 else v)
        for k, v in target.items()
    }


def save_gaussians(path: Path, gaussians: Gaussians, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "means": gaussians.means.detach().cpu(),
            "covariances": gaussians.covariances.detach().cpu(),
            "harmonics": gaussians.harmonics.detach().cpu(),
            "opacities": gaussians.opacities.detach().cpu(),
            "metadata": metadata,
        },
        path,
    )


def build_direct_object_mask(context_mask: torch.Tensor, num_surfaces: int, samples_per_pixel: int) -> torch.Tensor:
    pix = context_mask.flatten(start_dim=2)  # [B, V, H*W]
    pix = pix.reshape(pix.shape[0], -1)  # [B, V*H*W]
    return pix.repeat_interleave(num_surfaces * samples_per_pixel, dim=1)


@torch.no_grad()
def render_all(decoder, gaussians: Gaussians, target: dict, image_shape: tuple[int, int], output_dir: Path, render_chunk: int, save_depth: bool) -> None:
    num_views = target["image"].shape[1]
    for start in range(0, num_views, render_chunk):
        end = min(start + render_chunk, num_views)
        target_slice = sample_targets(target, torch.arange(start, end, device=target["image"].device))
        output = decoder.forward(
            gaussians,
            target_slice["extrinsics"],
            target_slice["intrinsics"],
            target_slice["near"],
            target_slice["far"],
            image_shape,
            depth_mode="depth" if save_depth else None,
        )
        for local_idx, view_idx in enumerate(range(start, end)):
            save_image(output.color[0, local_idx], output_dir / "color" / f"{view_idx:06d}.png")
            if save_depth and output.depth is not None:
                depth = output.depth[0, local_idx]
                (output_dir / "depth_raw").mkdir(parents=True, exist_ok=True)
                torch.save(depth.detach().cpu(), output_dir / "depth_raw" / f"{view_idx:06d}.pt")
                save_image(colorize_depth(depth), output_dir / "depth_rendered" / f"{view_idx:06d}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, default=Path("/home/junho/jieun/mvsplat/datasets/bear96_colmap"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/junho/jieun/mvsplat/checkpoints/re10k.ckpt"))
    parser.add_argument("--gaussians", type=Path, default=Path("/home/junho/jieun/mvsplat/outputs/bear96_per_scene_4view_3000/final_render/gaussians.pt"))
    parser.add_argument("--mask_root", type=Path, default=Path("/home/junho/jieun/3dgic_pixel_aligned_vggt/data/bear/bear/inpaint_object_mask_255"))
    parser.add_argument("--output_dir", type=Path, default=Path("/home/junho/jieun/mvsplat/outputs/bear96_removal_fullmask_opacity"))
    parser.add_argument("--num_context", type=int, default=4)
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--image_width", type=int, default=256)
    parser.add_argument("--near", type=float, default=1.0)
    parser.add_argument("--far", type=float, default=100.0)
    parser.add_argument("--opacity_scale", type=float, default=0.0)
    parser.add_argument("--mask_dilation", type=int, default=1)
    parser.add_argument("--render_chunk", type=int, default=4)
    parser.add_argument("--save_depth", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_shape = (args.image_height, args.image_width)
    chunk_path = args.dataset_root / "test" / "000000.torch"
    scene = torch.load(chunk_path, map_location="cpu")[0]
    num_views = len(scene["images"])
    original_indices = scene.get("original_indices")
    original_indices_list = (
        [int(i) for i in original_indices.tolist()]
        if original_indices is not None
        else None
    )
    context_indices = evenly_spaced_indices(num_views, args.num_context)
    cfg = make_cfg(args.num_context, image_shape, args.checkpoint, args.output_dir)
    _, decoder = load_encoder_decoder(cfg, args.checkpoint, device)
    decoder.eval()

    example = build_example(chunk_path, context_indices, image_shape, args.near, args.far)
    target = move_views_to_device(example["target"], device)
    gaussians = load_gaussians(args.gaussians, device)
    masks = load_masks(args.mask_root, image_shape, device, args.mask_dilation, original_indices_list)
    if masks.shape[0] != num_views:
        raise ValueError(f"Expected {num_views} masks, got {masks.shape[0]}")

    context_mask = masks[context_indices][None].to(device)
    remove_mask = build_direct_object_mask(
        context_mask,
        num_surfaces=1,
        samples_per_pixel=1,
    )[0]
    votes = torch.zeros_like(remove_mask, dtype=torch.int16)

    removed = int(remove_mask.sum().item())
    total = int(remove_mask.numel())
    edited_opacities = gaussians.opacities.clone()
    edited_opacities[:, remove_mask] = edited_opacities[:, remove_mask] * args.opacity_scale
    edited = Gaussians(
        means=gaussians.means,
        covariances=gaussians.covariances,
        harmonics=gaussians.harmonics,
        opacities=edited_opacities,
    )

    metadata = {
        "input_gaussians": str(args.gaussians),
        "mask_root": str(args.mask_root),
        "num_views": num_views,
        "original_indices_zero_based": original_indices_list,
        "context_indices_zero_based": context_indices,
        "context_original_indices_zero_based": (
            [original_indices_list[i] for i in context_indices]
            if original_indices_list is not None
            else context_indices
        ),
        "opacity_scale": args.opacity_scale,
        "mask_dilation": args.mask_dilation,
        "method": "context_direct_fullmask",
        "removed": removed,
        "total": total,
        "removed_fraction": removed / max(total, 1),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "removal_stats.json").write_text(json.dumps(metadata, indent=2))
    torch.save(remove_mask.detach().cpu(), args.output_dir / "remove_mask.pt")
    torch.save((~remove_mask).detach().cpu(), args.output_dir / "keep_mask.pt")
    torch.save(votes.detach().cpu(), args.output_dir / "votes.pt")
    save_gaussians(args.output_dir / "gaussians_removed_opacity.pt", edited, metadata)
    render_all(decoder, edited, target, image_shape, args.output_dir / "render", args.render_chunk, args.save_depth)
    print(f"Removed opacity for {removed}/{total} Gaussians ({removed / max(total, 1):.4%})")
    print(f"Saved removal outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
