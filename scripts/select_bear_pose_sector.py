#!/usr/bin/env python3
"""Select a COLMAP pose-defined angular sector from the bear camera trajectory."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prepare_bear96_re10k import qvec_to_rotmat, read_images_binary  # noqa: E402


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def camera_center(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    # COLMAP image pose is world-to-camera: x_cam = R x_world + t.
    rotation = qvec_to_rotmat(qvec).astype(np.float64)
    return -rotation.T @ tvec.astype(np.float64)


def frame_index(path: Path) -> int:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot infer frame index from {path.name}")
    # bear frame names are 1-based: frame_00001.jpg -> index 0.
    return int(digits) - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=Path, default=Path("/home/junho/jieun/3dgic_pixel_aligned_vggt/data/bear/bear/images"))
    parser.add_argument("--sparse_root", type=Path, default=Path("/home/junho/jieun/3dgic_pixel_aligned_vggt/data/bear/bear/sparse/0"))
    parser.add_argument("--output", type=Path, default=Path("/home/junho/jieun/mvsplat/outputs/bear_pose_sectors/sector_270_centered_ref000.json"))
    parser.add_argument("--reference_index", type=int, default=0)
    parser.add_argument("--sector_degrees", type=float, default=270.0)
    args = parser.parse_args()

    colmap_images = read_images_binary(args.sparse_root / "images.bin")
    by_name = {Path(item["name"]).name: item for item in colmap_images}
    image_paths = sorted(args.image_root.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No jpg images found in {args.image_root}")
    if not (0 <= args.reference_index < len(image_paths)):
        raise ValueError(f"reference_index must be in [0, {len(image_paths) - 1}]")

    centers = []
    names = []
    indices = []
    for image_path in image_paths:
        item = by_name.get(image_path.name)
        if item is None:
            raise KeyError(f"{image_path.name} was not found in COLMAP images.bin")
        centers.append(camera_center(item["qvec"], item["tvec"]))
        names.append(image_path.name)
        indices.append(frame_index(image_path))

    centers = np.stack(centers, axis=0)
    scene_center = centers.mean(axis=0)
    centered = centers - scene_center

    # Use PCA to estimate the turntable/trajectory plane from camera centers.
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis_x = vh[0]
    axis_y = vh[1]
    coords = np.stack([centered @ axis_x, centered @ axis_y], axis=-1)
    angles = np.arctan2(coords[:, 1], coords[:, 0])
    ref_angle = float(angles[args.reference_index])
    signed_delta = wrap_to_pi(angles - ref_angle)
    signed_delta_deg = np.rad2deg(signed_delta)

    half_sector = args.sector_degrees / 2.0
    selected_mask = np.abs(signed_delta_deg) <= half_sector + 1e-6
    selected_indices = [int(indices[i]) for i in np.nonzero(selected_mask)[0]]
    selected_frame_names = [names[i] for i in np.nonzero(selected_mask)[0]]

    order_by_angle = np.argsort(signed_delta_deg)
    payload = {
        "mode": "centered",
        "definition": "PCA camera-center plane, signed angular delta from reference in [-sector/2, +sector/2]",
        "image_root": str(args.image_root),
        "sparse_root": str(args.sparse_root),
        "reference_index_zero_based": args.reference_index,
        "reference_frame_name": names[args.reference_index],
        "sector_degrees": args.sector_degrees,
        "half_sector_degrees": half_sector,
        "num_total": len(image_paths),
        "num_selected": len(selected_indices),
        "selected_indices_zero_based": selected_indices,
        "selected_frame_numbers_one_based": [i + 1 for i in selected_indices],
        "selected_frame_names": selected_frame_names,
        "excluded_indices_zero_based": [int(indices[i]) for i in np.nonzero(~selected_mask)[0]],
        "excluded_frame_names": [names[i] for i in np.nonzero(~selected_mask)[0]],
        "angles_degrees_by_index": {
            str(int(indices[i])): float(signed_delta_deg[i]) for i in range(len(indices))
        },
        "angle_order_zero_based": [int(indices[i]) for i in order_by_angle],
        "scene_center": scene_center.tolist(),
        "pca_axis_x": axis_x.tolist(),
        "pca_axis_y": axis_y.tolist(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(
        f"Selected {len(selected_indices)}/{len(image_paths)} frames "
        f"within centered {args.sector_degrees:.1f} deg sector from index {args.reference_index}."
    )
    print(f"Selected zero-based range/list: {selected_indices}")
    print(f"Excluded zero-based list: {payload['excluded_indices_zero_based']}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
