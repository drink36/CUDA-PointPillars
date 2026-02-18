#!/usr/bin/env python3
# batch_convert_bin_folder.py

import argparse
from pathlib import Path
import numpy as np


def load_bin(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"{path} is not (N,4) float32 bin")
    return raw.reshape(-1, 4)


def save_bin(path: Path, pts: np.ndarray):
    pts.astype(np.float32).tofile(str(path))


def convert_points(pts, swap_xy=False, flip_x=False, flip_y=False, flip_z=False):
    out = pts.copy()

    if swap_xy:
        out[:, [0, 1]] = out[:, [1, 0]]

    if flip_x:
        out[:, 0] *= -1
    if flip_y:
        out[:, 1] *= -1
    if flip_z:
        out[:, 2] *= -1
    valid_mask = (out[:, 2] > -5.0) & (out[:, 2] < 3.0)
    out = out[valid_mask]
    max_intensity = out[:, 3].max()
    if max_intensity > 1.0:
        out[:, 3] /= max_intensity
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="input folder containing .bin files")
    ap.add_argument("--out_dir", required=True, help="output folder")
    ap.add_argument("--swap_xy", action="store_true")
    ap.add_argument("--flip_x", action="store_true")
    ap.add_argument("--flip_y", action="store_true")
    ap.add_argument("--flip_z", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bin_files = sorted(in_dir.glob("*.bin"))

    print(f"Found {len(bin_files)} files")

    for i, bin_path in enumerate(bin_files):
        pts = load_bin(bin_path)
        pts2 = convert_points(
            pts,
            swap_xy=args.swap_xy,
            flip_x=args.flip_x,
            flip_y=args.flip_y,
            flip_z=args.flip_z,
        )

        out_path = out_dir / bin_path.name
        save_bin(out_path, pts2)

        if i % 100 == 0:
            print(f"[{i}/{len(bin_files)}] converted {bin_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
