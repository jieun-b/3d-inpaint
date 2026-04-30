"""
Shared utilities for Stage 2 removal scripts.
Not meant to be run directly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from PIL import Image

from src.dataset.dataset_re10k import DatasetRE10k, DatasetRE10kCfg
from src.dataset.view_sampler.view_sampler_arbitrary import (
    ViewSamplerArbitrary,
    ViewSamplerArbitraryCfg,
)
from src.misc.image_io import save_image
from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDA, DecoderSplattingCUDACfg
from src.model.encoder import get_encoder
from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
from src.model.encoder.encoder_depthsplat import EncoderDepthSplatCfg
from src.model.encoder.visualization.encoder_visualizer_depthsplat_cfg import (
    EncoderVisualizerDepthSplatCfg,
)
from src.model.types import Gaussians
from src.visualization.vis_depth import viz_depth_tensor


CTX_VIEWS = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90]
TGT_VIEWS = list(range(96))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_encoder_weights(module: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {
        key.removeprefix("encoder."): value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    current = module.state_dict()
    compatible = {k: v for k, v in state_dict.items()
                  if k in current and current[k].shape == v.shape}
    module.load_state_dict(compatible, strict=False)
    print(f"[encoder] loaded={len(compatible)}  "
          f"skipped={len(state_dict) - len(compatible)}  "
          f"missing={len(current) - len(compatible)}")


def make_encoder_cfg() -> EncoderDepthSplatCfg:
    return EncoderDepthSplatCfg(
        name="depthsplat",
        d_feature=128,
        num_depth_candidates=128,
        num_surfaces=1,
        visualizer=EncoderVisualizerDepthSplatCfg(num_samples=8, min_resolution=256, export_ply=False),
        gaussian_adapter=GaussianAdapterCfg(
            gaussian_scale_min=1.0e-10, gaussian_scale_max=3.0, sh_degree=2,
        ),
        gaussians_per_pixel=1,
        unimatch_weights_path="pretrained/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth",
        downscale_factor=4,
        shim_patch_size=4,
        multiview_trans_attn_split=2,
        costvolume_unet_feat_dim=128,
        costvolume_unet_channel_mult=[1, 1, 1],
        costvolume_unet_attn_res=[],
        depth_unet_feat_dim=64,
        depth_unet_attn_res=[],
        depth_unet_channel_mult=[1, 1, 1],
        num_scales=2,
        upsample_factor=2,
        lowest_feature_resolution=4,
        depth_unet_channels=128,
        grid_sample_disable_cudnn=False,
        large_gaussian_head=False,
        color_large_unet=False,
        init_sh_input_img=True,
        feature_upsampler_channels=64,
        gaussian_regressor_channels=64,
        supervise_intermediate_depth=True,
        return_depth=True,
        train_depth_only=False,
        monodepth_vit_type="vitb",
        local_mv_match=2,
    )


def make_dataset_cfg(
    datasets_root: Path,
    context_views: list[int],
    target_views: list[int],
    image_shape: list[int],
) -> DatasetRE10kCfg:
    return DatasetRE10kCfg(
        name="re10k",
        roots=[datasets_root],
        make_baseline_1=False,
        augment=False,
        image_shape=image_shape,
        background_color=[0.0, 0.0, 0.0],
        cameras_are_circular=False,
        overfit_to_scene=None,
        view_sampler=ViewSamplerArbitraryCfg(
            name="arbitrary",
            num_context_views=len(context_views),
            num_target_views=len(target_views),
            context_views=context_views,
            target_views=target_views,
        ),
        baseline_epsilon=1.0e-3,
        max_fov=100.0,
        test_len=-1,
        test_chunk_interval=1,
        skip_bad_shape=False,
        near=0.5,
        far=100.0,
        baseline_scale_bounds=False,
        shuffle_val=False,
        train_times_per_scene=1,
        highres=False,
        use_index_to_load_chunk=False,
    )


def build_batch(dataset_cfg: DatasetRE10kCfg, device: torch.device) -> dict:
    sampler = ViewSamplerArbitrary(
        dataset_cfg.view_sampler, "test",
        dataset_cfg.overfit_to_scene is not None,
        dataset_cfg.cameras_are_circular, None,
    )
    dataset = DatasetRE10k(dataset_cfg, "test", sampler)
    example = next(iter(dataset))
    batch = {"scene": [example["scene"]]}
    for split in ("context", "target"):
        batch[split] = {}
        for key, value in example[split].items():
            batch[split][key] = value[None].to(device) if isinstance(value, torch.Tensor) else value
    return batch


def build_models(encoder_cfg: EncoderDepthSplatCfg, dataset_cfg: DatasetRE10kCfg, device: torch.device):
    encoder, _ = get_encoder(encoder_cfg)
    decoder = DecoderSplattingCUDA(DecoderSplattingCUDACfg(name="splatting_cuda"), dataset_cfg)
    return encoder.to(device).eval(), decoder.to(device).eval()


# ---------------------------------------------------------------------------
# Mask loading
# ---------------------------------------------------------------------------

def load_mask(mask_dir: Path, frame_idx: int, image_shape: tuple[int, int], threshold: float) -> torch.Tensor:
    """Load mask for frame_idx (0-based). Returns [H, W] bool."""
    path = mask_dir / f"frame_{frame_idx + 1:05d}.png"
    if not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")
    raw = Image.open(path).convert("L")
    mask = torch.from_numpy(np.array(raw, dtype=np.uint8)).float() / 255.0
    h_in, w_in = mask.shape
    h_out, w_out = image_shape
    scale = max(h_out / h_in, w_out / w_in)
    h_s, w_s = round(h_in * scale), round(w_in * scale)
    mask = F.interpolate(mask[None, None], size=(h_s, w_s), mode="nearest")[0, 0]
    r, c = (h_s - h_out) // 2, (w_s - w_out) // 2
    return mask[r:r + h_out, c:c + w_out] > threshold


def load_masks(mask_dir: Path, frame_indices: list[int], image_shape: tuple[int, int], threshold: float) -> torch.Tensor:
    """Returns [V, H, W] bool."""
    return torch.stack([load_mask(mask_dir, i, image_shape, threshold) for i in frame_indices])


# ---------------------------------------------------------------------------
# Gaussian ops
# ---------------------------------------------------------------------------

def apply_opacity_mask(gaussians: Gaussians, keep_mask: torch.Tensor) -> Gaussians:
    """keep_mask: [B, N] bool. Zeros out opacity of removed Gaussians."""
    return Gaussians(
        means=gaussians.means,
        covariances=gaussians.covariances,
        harmonics=gaussians.harmonics,
        opacities=gaussians.opacities * keep_mask.to(gaussians.opacities.dtype),
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def depth_to_rgb(depth: torch.Tensor) -> torch.Tensor:
    depth = depth.detach().cpu()
    inv = torch.zeros_like(depth)
    valid = depth > 0
    inv[valid] = 1.0 / depth[valid].clamp_min(1e-6)
    inv = torch.nan_to_num(inv, nan=0.0, posinf=0.0, neginf=0.0)
    return viz_depth_tensor(inv, return_numpy=False).float() / 255.0


def render_views_chunked(
    decoder: DecoderSplattingCUDA,
    gaussians: Gaussians,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    near: torch.Tensor,
    far: torch.Tensor,
    image_shape: tuple[int, int],
    chunk_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Returns [V, 3, H, W] colors and [V, H, W] depths (both on CPU)."""
    V = extrinsics.shape[1]
    colors, depths = [], []
    for s in range(0, V, chunk_size):
        e = min(s + chunk_size, V)
        out = decoder.forward(
            gaussians,
            extrinsics[:, s:e], intrinsics[:, s:e],
            near[:, s:e], far[:, s:e],
            image_shape, depth_mode="depth",
        )
        colors.append(out.color[0].detach().cpu())
        if out.depth is not None:
            depths.append(out.depth[0].detach().cpu())
    return torch.cat(colors), (torch.cat(depths) if depths else None)


def save_renders(
    colors: torch.Tensor,             # [V, 3, H, W]
    rendered_depths: torch.Tensor | None,  # [V, H, W]  — Gaussian splatting depth (target views)
    frame_indices: list[int],
    output_dir: Path,
) -> None:
    """Save rendered RGB and rendered depth (per target view)."""
    color_dir = output_dir / "color"
    depth_rendered_dir = output_dir / "depth_rendered"
    color_dir.mkdir(parents=True, exist_ok=True)
    if rendered_depths is not None:
        depth_rendered_dir.mkdir(parents=True, exist_ok=True)
    for local_i, frame_idx in enumerate(frame_indices):
        name = f"frame_{int(frame_idx)+1:05d}.png"
        save_image(colors[local_i], color_dir / name)
        if rendered_depths is not None:
            save_image(depth_to_rgb(rendered_depths[local_i]), depth_rendered_dir / name)


def save_encoder_depths(
    encoder_depths: torch.Tensor,  # [V, H, W]  — cost-volume depth (context views)
    frame_indices: list[int],
    output_dir: Path,
) -> None:
    """Save encoder predicted depth for the context views."""
    depth_encoder_dir = output_dir / "depth_encoder"
    depth_encoder_dir.mkdir(parents=True, exist_ok=True)
    for local_i, frame_idx in enumerate(frame_indices):
        name = f"frame_{int(frame_idx)+1:05d}.png"
        save_image(depth_to_rgb(encoder_depths[local_i]), depth_encoder_dir / name)
