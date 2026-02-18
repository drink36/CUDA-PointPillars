# compare_lidar_xyz_dist.py
# Usage:
#   python compare_lidar_xyz_dist.py --kitti_bin /path/to/kitti.bin --custom_bin /path/to/custom.bin
#
# Assumes each .bin is KITTI-style float32 with shape (N,4): x,y,z,intensity.
# If your custom format differs, edit load_bin().

import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_bin(path: str) -> np.ndarray:
    pts = np.fromfile(path, dtype=np.float32)
    if pts.size % 4 != 0:
        raise ValueError(f"{path}: size={pts.size} not divisible by 4. Your custom bin isn't KITTI (x,y,z,i).")
    pts = pts.reshape(-1, 4)[:, :3]  # keep xyz
    return pts

def summary(name: str, xyz: np.ndarray):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.sqrt(x * x + y * y)
    print(f"\n[{name}] N={len(xyz):,}")
    print(f"  x: min={x.min():.3f}, p1={np.percentile(x,1):.3f}, p50={np.median(x):.3f}, p99={np.percentile(x,99):.3f}, max={x.max():.3f}")
    print(f"  y: min={y.min():.3f}, p1={np.percentile(y,1):.3f}, p50={np.median(y):.3f}, p99={np.percentile(y,99):.3f}, max={y.max():.3f}")
    print(f"  z: min={z.min():.3f}, p1={np.percentile(z,1):.3f}, p50={np.median(z):.3f}, p99={np.percentile(z,99):.3f}, max={z.max():.3f}")
    print(f"  r: min={r.min():.3f}, p1={np.percentile(r,1):.3f}, p50={np.median(r):.3f}, p99={np.percentile(r,99):.3f}, max={r.max():.3f}")
    print(f"  front ratio (x>0): {np.mean(x>0):.3f}")
    print(f"  left ratio  (y>0): {np.mean(y>0):.3f}")
    print(f"  mean y: {y.mean():.3f}  (if sign differs vs KITTI, you likely need y flip)")

def common_range(a: np.ndarray, b: np.ndarray, lo_q=0.5, hi_q=99.5):
    # robust range using percentiles across both sets
    lo = min(np.percentile(a, lo_q), np.percentile(b, lo_q))
    hi = max(np.percentile(a, hi_q), np.percentile(b, hi_q))
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    return lo, hi

def plot_hist(ax, a, b, title, bins=200):
    lo, hi = common_range(a, b)
    edges = np.linspace(lo, hi, bins + 1)
    ax.hist(a, bins=edges, density=True, alpha=0.5, label="KITTI")
    ax.hist(b, bins=edges, density=True, alpha=0.5, label="Custom")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

def plot_cdf(ax, a, b, title):
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    a_y = np.linspace(0, 1, len(a_sorted), endpoint=False)
    b_y = np.linspace(0, 1, len(b_sorted), endpoint=False)
    ax.plot(a_sorted, a_y, label="KITTI")
    ax.plot(b_sorted, b_y, label="Custom")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

def ks_stat(a, b):
    # KS statistic without scipy (approx exact on merged grid)
    a = np.sort(a)
    b = np.sort(b)
    grid = np.sort(np.concatenate([a, b]))
    # CDFs via searchsorted
    ca = np.searchsorted(a, grid, side="right") / len(a)
    cb = np.searchsorted(b, grid, side="right") / len(b)
    return np.max(np.abs(ca - cb))

def plot_xy_heat(ax, xyz_a, xyz_b, title, max_points=400_000, bins=400):
    # compare XY density by plotting two heatmaps side-by-side is clearer;
    # here we overlay contour-ish via difference heatmap.
    def sample_xy(xyz):
        if len(xyz) > max_points:
            idx = np.random.choice(len(xyz), size=max_points, replace=False)
            xyz = xyz[idx]
        return xyz[:, 0], xyz[:, 1]

    xa, ya = sample_xy(xyz_a)
    xb, yb = sample_xy(xyz_b)

    # range based on robust percentiles over both
    x_all = np.concatenate([xa, xb])
    y_all = np.concatenate([ya, yb])
    xlo, xhi = np.percentile(x_all, 0.5), np.percentile(x_all, 99.5)
    ylo, yhi = np.percentile(y_all, 0.5), np.percentile(y_all, 99.5)

    Ha, xedges, yedges = np.histogram2d(xa, ya, bins=bins, range=[[xlo, xhi], [ylo, yhi]])
    Hb, _, _ = np.histogram2d(xb, yb, bins=[xedges, yedges])

    # normalize to probability mass then show signed difference
    Ha = Ha / (Ha.sum() + 1e-12)
    Hb = Hb / (Hb.sum() + 1e-12)
    Hdiff = Hb - Ha  # Custom - KITTI

    im = ax.imshow(
        Hdiff.T,
        origin="lower",
        extent=[xlo, xhi, ylo, yhi],
        aspect="auto",
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_bin", required=True)
    ap.add_argument("--custom_bin", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)

    k = load_bin(args.kitti_bin)
    c = load_bin(args.custom_bin)

    summary("KITTI", k)
    summary("Custom", c)

    # features
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    cx, cy, cz = c[:, 0], c[:, 1], c[:, 2]
    kr = np.sqrt(kx * kx + ky * ky)
    cr = np.sqrt(cx * cx + cy * cy)

    # KS stats
    print("\n[KS statistic] (bigger => more different; 0 means identical)")
    for name, a, b in [("x", kx, cx), ("y", ky, cy), ("z", kz, cz), ("r=sqrt(x^2+y^2)", kr, cr)]:
        print(f"  {name}: KS={ks_stat(a, b):.4f}")

    # plots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    plot_hist(ax0, kx, cx, "Histogram (density): x")
    plot_hist(ax1, ky, cy, "Histogram (density): y")
    plot_hist(ax2, kz, cz, "Histogram (density): z")
    ax0.legend()

    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])
    plot_cdf(ax3, kx, cx, "CDF: x")
    plot_cdf(ax4, ky, cy, "CDF: y")
    plot_cdf(ax5, kr, cr, "CDF: r = sqrt(x^2+y^2)")
    ax3.legend()

    ax6 = fig.add_subplot(gs[2, :])
    plot_xy_heat(ax6, k, c, "XY density difference heatmap (Custom - KITTI)")

    fig.suptitle("LiDAR coordinate distribution comparison (KITTI vs Custom)", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
