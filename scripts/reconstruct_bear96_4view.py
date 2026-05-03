#!/usr/bin/env python3
"""Run MVSplat on 4 fixed bear context views and render all 96 target views."""

from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from hydra import compose, initialize_config_dir
from jaxtyping import install_import_hook
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

with install_import_hook(("src",), ("beartype", "beartype")):
    from src.config import load_typed_root_config
    from src.dataset.shims.crop_shim import apply_crop_shim
    from src.global_cfg import set_cfg
    from src.misc.image_io import save_image
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.visualization.color_map import apply_color_map_to_image


def convert_poses(poses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, _ = poses.shape
    intrinsics = torch.eye(3, dtype=torch.float32)
    intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
    fx, fy, cx, cy = poses[:, :4].T
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
    w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
    return w2c.inverse(), intrinsics


def convert_images(images: list[torch.Tensor]) -> torch.Tensor:
    to_tensor = tf.ToTensor()
    out = []
    for image in images:
        pil = Image.open(BytesIO(image.numpy().tobytes()))
        out.append(to_tensor(pil)[:3])
    return torch.stack(out)


def evenly_spaced_indices(num_views: int, num_context: int) -> list[int]:
    if num_context < 2:
        return [0]
    return [round(i * (num_views - 1) / (num_context - 1)) for i in range(num_context)]


def build_example(chunk_path: Path, context_indices: list[int], image_shape: tuple[int, int], near: float, far: float) -> dict:
    chunk = torch.load(chunk_path, map_location="cpu")
    if len(chunk) != 1:
        raise ValueError(f"Expected one scene in {chunk_path}, got {len(chunk)}")
    scene = chunk[0]
    extrinsics, intrinsics = convert_poses(scene["cameras"])
    target_indices = list(range(len(scene["images"])))

    context_index = torch.tensor(context_indices, dtype=torch.long)
    target_index = torch.tensor(target_indices, dtype=torch.long)
    example = {
        "context": {
            "extrinsics": extrinsics[context_index],
            "intrinsics": intrinsics[context_index],
            "image": convert_images([scene["images"][i] for i in context_indices]),
            "near": torch.full((len(context_indices),), near, dtype=torch.float32),
            "far": torch.full((len(context_indices),), far, dtype=torch.float32),
            "index": context_index,
        },
        "target": {
            "extrinsics": extrinsics[target_index],
            "intrinsics": intrinsics[target_index],
            "image": convert_images([scene["images"][i] for i in target_indices]),
            "near": torch.full((len(target_indices),), near, dtype=torch.float32),
            "far": torch.full((len(target_indices),), far, dtype=torch.float32),
            "index": target_index,
        },
        "scene": scene["key"],
    }
    example = apply_crop_shim(example, image_shape)
    for key in ("context", "target"):
        for subkey, value in example[key].items():
            if torch.is_tensor(value):
                example[key][subkey] = value.unsqueeze(0)
    return example


def make_cfg(num_context_views: int, image_shape: tuple[int, int], checkpoint: Path, output_dir: Path):
    overrides = [
        "+experiment=re10k",
        "mode=test",
        f"dataset.image_shape=[{image_shape[0]},{image_shape[1]}]",
        "dataset.roots=[datasets/bear96_colmap]",
        "dataset.near=1.0",
        "dataset.far=100.0",
        "dataset.baseline_scale_bounds=false",
        "dataset.make_baseline_1=false",
        f"dataset.view_sampler.num_context_views={num_context_views}",
        "dataset.view_sampler.num_target_views=1",
        "data_loader.test.batch_size=1",
        "data_loader.test.num_workers=0",
        f"checkpointing.load={checkpoint}",
        "checkpointing.resume=false",
        f"test.output_path={output_dir}",
        "test.compute_scores=false",
        "test.save_image=true",
        "test.save_video=false",
    ]
    with initialize_config_dir(config_dir=str(REPO_ROOT / "config"), version_base=None):
        cfg_dict = compose(config_name="main", overrides=overrides)
    set_cfg(cfg_dict)
    return load_typed_root_config(cfg_dict)


def load_encoder_decoder(cfg, checkpoint: Path, device: torch.device):
    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    state = torch.load(checkpoint, map_location="cpu")["state_dict"]
    encoder_state = {k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected encoder checkpoint keys: {unexpected[:10]}")
    if missing:
        print(f"Warning: missing encoder keys: {len(missing)}")
    return encoder.to(device).eval(), decoder.to(device).eval()


def move_views_to_device(views: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in views.items()}


def colorize_depth(depth: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(depth)
    positive = finite & (depth > 0)
    valid = depth[positive]
    if valid.numel() == 0:
        normalized = torch.zeros_like(depth)
    else:
        lo = torch.quantile(valid, 0.02)
        hi = torch.quantile(valid, 0.98)
        if (hi - lo).abs() < 1e-6:
            lo = valid.min()
            hi = valid.max()
        normalized = ((depth - lo) / (hi - lo + 1e-6)).clamp(0, 1)
        normalized = torch.where(finite, normalized, torch.zeros_like(normalized))
    return apply_color_map_to_image(normalized, "inferno")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, default=Path("/home/junho/jieun/mvsplat/datasets/bear96_colmap"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/junho/jieun/mvsplat/checkpoints/re10k.ckpt"))
    parser.add_argument("--output_dir", type=Path, default=Path("/home/junho/jieun/mvsplat/outputs/bear96_reconstruction_4view"))
    parser.add_argument("--num_context", type=int, default=4)
    parser.add_argument("--render_chunk", type=int, default=8)
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--image_width", type=int, default=256)
    parser.add_argument("--near", type=float, default=1.0)
    parser.add_argument("--far", type=float, default=100.0)
    parser.add_argument("--save_depth", action="store_true")
    parser.add_argument("--save_encoder_depth", action="store_true")
    args = parser.parse_args()

    chunk_path = args.dataset_root / "test" / "000000.torch"
    num_views = len(torch.load(chunk_path, map_location="cpu")[0]["images"])
    context_indices = evenly_spaced_indices(num_views, args.num_context)
    image_shape = (args.image_height, args.image_width)

    cfg = make_cfg(args.num_context, image_shape, args.checkpoint, args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder = load_encoder_decoder(cfg, args.checkpoint, device)
    example = build_example(chunk_path, context_indices, image_shape, args.near, args.far)
    context = move_views_to_device(example["context"], device)
    target = move_views_to_device(example["target"], device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "scene": example["scene"],
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "num_views": num_views,
        "context_indices_zero_based": context_indices,
        "context_frame_numbers_one_based": [i + 1 for i in context_indices],
        "target_indices_zero_based": list(range(num_views)),
        "image_shape": list(image_shape),
        "near": args.near,
        "far": args.far,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    with torch.no_grad():
        visualization_dump = {} if args.save_encoder_depth else None
        gaussians = encoder(
            context,
            global_step=300000,
            deterministic=True,
            visualization_dump=visualization_dump,
            scene_names=[example["scene"]],
        )
        torch.save(
            {
                "means": gaussians.means.detach().cpu(),
                "covariances": gaussians.covariances.detach().cpu(),
                "harmonics": gaussians.harmonics.detach().cpu(),
                "opacities": gaussians.opacities.detach().cpu(),
                "metadata": metadata,
            },
            args.output_dir / "gaussians.pt",
        )
        if args.save_encoder_depth and visualization_dump is not None:
            encoder_depth = visualization_dump["depth"][0].mean(dim=(-1, -2))
            for local_idx, view_idx in enumerate(context_indices):
                depth = encoder_depth[local_idx]
                (args.output_dir / "encoder_depth_raw").mkdir(parents=True, exist_ok=True)
                torch.save(depth.detach().cpu(), args.output_dir / "encoder_depth_raw" / f"{view_idx:06d}.pt")
                save_image(colorize_depth(depth), args.output_dir / "encoder_depth" / f"{view_idx:06d}.png")

        for start in range(0, num_views, args.render_chunk):
            end = min(start + args.render_chunk, num_views)
            target_slice = {k: v[:, start:end] if torch.is_tensor(v) and v.ndim >= 2 else v for k, v in target.items()}
            depth_mode = "depth" if args.save_depth else None
            output = decoder.forward(
                gaussians,
                target_slice["extrinsics"],
                target_slice["intrinsics"],
                target_slice["near"],
                target_slice["far"],
                image_shape,
                depth_mode=depth_mode,
            )
            for local_idx, view_idx in enumerate(range(start, end)):
                save_image(output.color[0, local_idx], args.output_dir / "color" / f"{view_idx:06d}.png")
                if args.save_depth and output.depth is not None:
                    depth = output.depth[0, local_idx]
                    (args.output_dir / "depth_raw").mkdir(parents=True, exist_ok=True)
                    torch.save(depth.detach().cpu(), args.output_dir / "depth_raw" / f"{view_idx:06d}.pt")
                    save_image(colorize_depth(depth), args.output_dir / "depth_rendered" / f"{view_idx:06d}.png")

    print(f"Rendered {num_views} views to {args.output_dir}")
    print(f"Context frames: {[i + 1 for i in context_indices]}")


if __name__ == "__main__":
    main()
