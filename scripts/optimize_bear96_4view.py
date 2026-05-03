#!/usr/bin/env python3
"""Per-scene fine-tune MVSplat on bear with fixed context views and random targets.

This is not direct Gaussian-tensor optimization. It follows the original MVSplat
training path: context images go through the encoder, predicted Gaussians are
rasterized by the decoder, and RGB/LPIPS losses update the encoder weights.
The final Gaussians are exported only after fine-tuning for removal/inpainting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from lpips import LPIPS

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
    from src.evaluation.metrics import compute_psnr


def sample_targets(target: dict, indices: torch.Tensor) -> dict:
    return {
        k: (v[:, indices] if torch.is_tensor(v) and v.ndim >= 2 else v)
        for k, v in target.items()
    }


def save_checkpoint(path: Path, encoder: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "encoder": encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metadata": metadata,
        "rng_state_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["rng_state_cuda_all"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def restore_or_replay_rng(resume: dict, resumed_step: int, num_views: int, device: torch.device) -> str:
    if "rng_state_cpu" in resume:
        torch.set_rng_state(resume["rng_state_cpu"].cpu())
        if torch.cuda.is_available() and "rng_state_cuda_all" in resume:
            cuda_states = [state.detach().cpu().to(torch.uint8) for state in resume["rng_state_cuda_all"]]
            torch.cuda.set_rng_state_all(cuda_states)
        return "checkpoint_rng_state"

    # Backward compatibility for old checkpoints. The only intended stochastic
    # operation in this loop is target sampling, so replay it to align the next
    # sampled target batch with a continuous run from the same seed.
    for _ in range(resumed_step):
        torch.randperm(num_views, device=device)
    return "replayed_target_sampling_rng"


@torch.no_grad()
def render_all(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    context: dict,
    target: dict,
    image_shape: tuple[int, int],
    output_dir: Path,
    render_chunk: int,
    global_step: int,
    save_depth: bool,
    save_encoder_depth: bool,
    context_indices: list[int],
    scene_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dump = {} if save_encoder_depth else None
    gaussians = encoder(
        context,
        global_step=global_step,
        deterministic=True,
        visualization_dump=visualization_dump,
        scene_names=[scene_name],
    )
    torch.save(
        {
            "means": gaussians.means.detach().cpu(),
            "covariances": gaussians.covariances.detach().cpu(),
            "harmonics": gaussians.harmonics.detach().cpu(),
            "opacities": gaussians.opacities.detach().cpu(),
        },
        output_dir / "gaussians.pt",
    )

    if save_encoder_depth and visualization_dump is not None:
        encoder_depth = visualization_dump["depth"][0].mean(dim=(-1, -2))
        for local_idx, view_idx in enumerate(context_indices):
            depth = encoder_depth[local_idx]
            (output_dir / "encoder_depth_raw").mkdir(parents=True, exist_ok=True)
            torch.save(depth.detach().cpu(), output_dir / "encoder_depth_raw" / f"{view_idx:06d}.pt")
            save_image(colorize_depth(depth), output_dir / "encoder_depth" / f"{view_idx:06d}.png")

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
            save_image(target_slice["image"][0, local_idx], output_dir / "gt" / f"{view_idx:06d}.png")
            if save_depth and output.depth is not None:
                depth = output.depth[0, local_idx]
                (output_dir / "depth_raw").mkdir(parents=True, exist_ok=True)
                torch.save(depth.detach().cpu(), output_dir / "depth_raw" / f"{view_idx:06d}.pt")
                save_image(colorize_depth(depth), output_dir / "depth_rendered" / f"{view_idx:06d}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, default=Path("/home/junho/jieun/mvsplat/datasets/bear96_colmap"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/junho/jieun/mvsplat/checkpoints/re10k.ckpt"))
    parser.add_argument("--output_dir", type=Path, default=Path("/home/junho/jieun/mvsplat/outputs/bear96_per_scene_4view_3000"))
    parser.add_argument("--num_context", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--target_batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lpips_weight", type=float, default=0.05)
    parser.add_argument("--lpips_after", type=int, default=0)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--render_chunk", type=int, default=4)
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--image_width", type=int, default=256)
    parser.add_argument("--near", type=float, default=1.0)
    parser.add_argument("--far", type=float, default=100.0)
    parser.add_argument("--save_depth", action="store_true")
    parser.add_argument("--save_encoder_depth", action="store_true")
    parser.add_argument("--resume_checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    chunk_path = args.dataset_root / "test" / "000000.torch"
    if not chunk_path.exists():
        raise FileNotFoundError(f"{chunk_path} does not exist. Run prepare_bear96_re10k.py first.")

    num_views = len(torch.load(chunk_path, map_location="cpu")[0]["images"])
    context_indices = evenly_spaced_indices(num_views, args.num_context)
    image_shape = (args.image_height, args.image_width)
    cfg = make_cfg(args.num_context, image_shape, args.checkpoint, args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder = load_encoder_decoder(cfg, args.checkpoint, device)
    decoder.eval()
    decoder.requires_grad_(False)

    example = build_example(chunk_path, context_indices, image_shape, args.near, args.far)
    context = move_views_to_device(example["context"], device)
    target = move_views_to_device(example["target"], device)

    trainable_params = [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    lpips_fn = LPIPS(net="vgg").to(device).eval() if args.lpips_weight > 0 else None
    if lpips_fn is not None:
        for p in lpips_fn.parameters():
            p.requires_grad_(False)

    metadata = {
        "scene": example["scene"],
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "num_views": num_views,
        "context_indices_zero_based": context_indices,
        "context_frame_numbers_one_based": [i + 1 for i in context_indices],
        "target_sampling": f"random {args.target_batch} of {num_views} per step",
        "optimization_type": "mvsplat_encoder_finetune",
        "optimization_target": "encoder_parameters",
        "gaussian_tensor_direct_optimization": False,
        "decoder": "frozen_splatting_cuda",
        "trainable_parameter_count": int(sum(p.numel() for p in trainable_params)),
        "steps": args.steps,
        "lr": args.lr,
        "weight_decay": 1e-4,
        "lpips_weight": args.lpips_weight,
        "loss": "mse + lpips_weight * lpips",
        "image_shape": list(image_shape),
        "near": args.near,
        "far": args.far,
    }
    start_step = 1
    if args.resume_checkpoint is not None:
        resume = torch.load(args.resume_checkpoint, map_location=device)
        encoder.load_state_dict(resume["encoder"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        resumed_step = int(resume["step"])
        start_step = resumed_step + 1
        rng_resume_mode = restore_or_replay_rng(resume, resumed_step, num_views, device)
        metadata["resume_checkpoint"] = str(args.resume_checkpoint)
        metadata["resume_step"] = resumed_step
        metadata["rng_resume_mode"] = rng_resume_mode
        if args.steps < start_step:
            raise ValueError(
                f"--steps ({args.steps}) must be >= resume step + 1 ({start_step}). "
                "Use --steps as the final global step, not the number of extra steps."
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    log_path = args.output_dir / "train_loss.log"

    encoder.train()
    for step in range(start_step, args.steps + 1):
        indices = torch.randperm(num_views, device=device)[: args.target_batch].sort().values
        target_slice = sample_targets(target, indices)

        optimizer.zero_grad(set_to_none=True)
        gaussians = encoder(context, global_step=step, deterministic=False, scene_names=[example["scene"]])
        output = decoder.forward(
            gaussians,
            target_slice["extrinsics"],
            target_slice["intrinsics"],
            target_slice["near"],
            target_slice["far"],
            image_shape,
            depth_mode=None,
        )
        mse = F.mse_loss(output.color, target_slice["image"])
        loss = mse
        lpips_loss = torch.zeros((), device=device)
        if lpips_fn is not None and step >= args.lpips_after:
            lpips_loss = lpips_fn(
                rearrange(output.color, "b v c h w -> (b v) c h w"),
                rearrange(target_slice["image"], "b v c h w -> (b v) c h w"),
                normalize=True,
            ).mean()
            loss = loss + args.lpips_weight * lpips_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.5)
        optimizer.step()

        if step == 1 or step % args.print_every == 0:
            with torch.no_grad():
                psnr = compute_psnr(
                    rearrange(target_slice["image"], "b v c h w -> (b v) c h w"),
                    rearrange(output.color, "b v c h w -> (b v) c h w"),
                ).mean()
            line = (
                f"step {step:06d} "
                f"loss={loss.item():.6f} "
                f"mse={mse.item():.6f} "
                f"lpips={lpips_loss.item():.6f} "
                f"psnr={psnr.item():.3f} "
                f"targets={[int(i) for i in indices.detach().cpu()]}"
            )
            print(line, flush=True)
            with log_path.open("a") as f:
                f.write(line + "\n")

        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(args.output_dir / "checkpoints" / f"step_{step:06d}.pt", encoder, optimizer, step, metadata)

    encoder.eval()
    render_all(
        encoder,
        decoder,
        context,
        target,
        image_shape,
        args.output_dir / "final_render",
        args.render_chunk,
        args.steps,
        args.save_depth,
        args.save_encoder_depth,
        context_indices,
        example["scene"],
    )
    print(f"Saved optimized outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
