"""Plot reproducible benchmark maps and 3D slices for the Phase 2 bank.

The 3D gallery shows three x3 slices (0.25, 0.5, 0.75) to reveal how
landscape structure changes away from the midpoint.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.benchmarks import FAMILIES, get_benchmark


def main():
    out = Path("results/phase2/gallery")
    out.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.0, 1.0, 120)
    xx, yy = np.meshgrid(grid, grid)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    for dimension, names in [
        (2, list(FAMILIES) + ["branin"]),
        (3, list(FAMILIES) + ["hartmann3"]),
    ]:
        slices = (None,) if dimension == 2 else (0.25, 0.5, 0.75)
        for slice_x3 in slices:
            fig, axes = plt.subplots(3, 4, figsize=(14, 10), constrained_layout=True)
            for ax, name in zip(axes.flat, names):
                f = get_benchmark(name, dimension, field_seed=2026)
                points = xy if dimension == 2 else np.column_stack((xy, np.full(len(xy), slice_x3)))
                values = f(points).reshape(xx.shape)
                image = ax.imshow(values, origin="lower", extent=(0, 1, 0, 1), cmap="viridis", aspect="equal")
                ax.set_title(name.replace("_", " "), fontsize=9)
                ax.set_xlabel("x1")
                ax.set_ylabel("x2" if dimension == 2 else f"x2 (x3 = {slice_x3:g})")
                fig.colorbar(image, ax=ax, shrink=0.78)
            axes.flat[-1].axis("off")
            title = "2D benchmark landscapes" if dimension == 2 else f"3D slices with x3 fixed at {slice_x3:g}"
            fig.suptitle(title, fontsize=15)
            suffix = "2d" if dimension == 2 else f"3d_slices_x3_{int(slice_x3 * 100):03d}"
            if dimension == 3 and slice_x3 == 0.5:
                suffix = "3d_slices"
            fig.savefig(out / f"benchmark_gallery_{suffix}.png", dpi=160)
            plt.close(fig)
    print(f"Wrote gallery to {out}")


if __name__ == "__main__":
    main()
