#!/usr/bin/env python3
"""Pack the bear COLMAP scene into the RE10K-like chunk format used by MVSplat."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
from PIL import Image


CAMERA_MODEL_IDS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def read_bytes(fid, num_bytes: int, fmt: str):
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError("Unexpected EOF while reading COLMAP binary.")
    return struct.unpack(fmt, data)


def read_cameras_binary(path: Path) -> dict[int, dict]:
    cameras = {}
    with path.open("rb") as fid:
        (num_cameras,) = read_bytes(fid, 8, "<Q")
        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_bytes(fid, 24, "<iiQQ")
            model_name, num_params = CAMERA_MODEL_IDS[model_id]
            params = np.array(read_bytes(fid, 8 * num_params, "<" + "d" * num_params))
            cameras[camera_id] = {
                "model": model_name,
                "width": int(width),
                "height": int(height),
                "params": params,
            }
    return cameras


def read_images_binary(path: Path) -> list[dict]:
    images = []
    with path.open("rb") as fid:
        (num_images,) = read_bytes(fid, 8, "<Q")
        for _ in range(num_images):
            raw = read_bytes(fid, 64, "<idddddddi")
            image_id = raw[0]
            qvec = np.array(raw[1:5], dtype=np.float64)
            tvec = np.array(raw[5:8], dtype=np.float64)
            camera_id = raw[8]

            name_bytes = bytearray()
            while True:
                c = fid.read(1)
                if c == b"\x00":
                    break
                name_bytes.extend(c)
            name = name_bytes.decode("utf-8")

            (num_points2d,) = read_bytes(fid, 8, "<Q")
            fid.seek(num_points2d * 24, 1)

            images.append(
                {
                    "id": image_id,
                    "qvec": qvec,
                    "tvec": tvec,
                    "camera_id": camera_id,
                    "name": name,
                }
            )
    return images


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float32,
    )


def camera_params_to_intrinsics(camera: dict, image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    model = camera["model"]
    params = camera["params"]
    width, height = image_size
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        fx = fy = f
    elif model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
        fx, fy, cx, cy = params[:4]
    elif model in ("SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL", "RADIAL_FISHEYE", "FOV"):
        f, cx, cy = params[:3]
        fx = fy = f
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {model}")
    return float(fx / width), float(fy / height), float(cx / width), float(cy / height)


def image_to_byte_tensor(path: Path) -> torch.Tensor:
    return torch.tensor(bytearray(path.read_bytes()), dtype=torch.uint8)


def load_selected_indices(path: Path | None) -> set[int] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    if "selected_indices_zero_based" not in payload:
        raise KeyError(f"{path} does not contain selected_indices_zero_based")
    return {int(i) for i in payload["selected_indices_zero_based"]}


def infer_frame_index(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot infer frame index from {path.name}")
    return int(digits) - 1


def build_chunk(
    image_root: Path,
    sparse_root: Path,
    scene_key: str,
    selected_indices: set[int] | None = None,
) -> list[dict]:
    cameras = read_cameras_binary(sparse_root / "cameras.bin")
    colmap_images = read_images_binary(sparse_root / "images.bin")
    by_name = {Path(item["name"]).name: item for item in colmap_images}
    image_paths = sorted(image_root.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No jpg images found in {image_root}")
    if selected_indices is not None:
        image_paths = [p for p in image_paths if infer_frame_index(p) in selected_indices]
        if not image_paths:
            raise ValueError("No images remain after applying selected_indices")

    cameras_out = []
    images_out = []
    timestamps = []
    original_indices = []
    for idx, image_path in enumerate(image_paths):
        item = by_name.get(image_path.name)
        if item is None:
            raise KeyError(f"{image_path.name} was not found in COLMAP images.bin")
        original_idx = infer_frame_index(image_path)

        with Image.open(image_path) as im:
            width, height = im.size
        fx, fy, cx, cy = camera_params_to_intrinsics(cameras[item["camera_id"]], (width, height))
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = qvec_to_rotmat(item["qvec"])
        w2c[:3, 3] = item["tvec"].astype(np.float32)

        camera_vec = np.concatenate(
            [
                np.array([fx, fy, cx, cy, 0.0, 0.0], dtype=np.float32),
                w2c[:3].reshape(-1).astype(np.float32),
            ]
        )
        cameras_out.append(torch.from_numpy(camera_vec))
        images_out.append(image_to_byte_tensor(image_path))
        timestamps.append(idx)
        original_indices.append(original_idx)

    return [
        {
            "key": scene_key,
            "url": str(image_root),
            "timestamps": torch.tensor(timestamps, dtype=torch.int64),
            "original_indices": torch.tensor(original_indices, dtype=torch.int64),
            "original_frame_names": [p.name for p in image_paths],
            "cameras": torch.stack(cameras_out),
            "images": images_out,
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=Path, default=Path("/home/junho/jieun/3dgic_pixel_aligned_vggt/data/bear/bear/images"))
    parser.add_argument("--sparse_root", type=Path, default=Path("/home/junho/jieun/3dgic_pixel_aligned_vggt/data/bear/bear/sparse/0"))
    parser.add_argument("--output_root", type=Path, default=Path("/home/junho/jieun/mvsplat/datasets/bear96_colmap"))
    parser.add_argument("--scene_key", default="bear96_colmap_original")
    parser.add_argument("--indices_json", type=Path, default=None, help="Optional pose-sector JSON with selected_indices_zero_based")
    args = parser.parse_args()

    selected_indices = load_selected_indices(args.indices_json)
    chunk = build_chunk(args.image_root, args.sparse_root, args.scene_key, selected_indices)
    for stage in ("train", "test"):
        stage_root = args.output_root / stage
        stage_root.mkdir(parents=True, exist_ok=True)
        torch.save(chunk, stage_root / "000000.torch")
        with (stage_root / "index.json").open("w") as f:
            json.dump({args.scene_key: "000000.torch"}, f, indent=2)

    print(f"Packed {len(chunk[0]['images'])} frames into {args.output_root}")
    if selected_indices is not None:
        print(f"Original zero-based indices: {chunk[0]['original_indices'].tolist()}")


if __name__ == "__main__":
    main()
