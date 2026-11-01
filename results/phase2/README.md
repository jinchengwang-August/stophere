# Phase 2 results

This directory contains generated landscape visualizations for the validated Phase 2 benchmark bank.

- `gallery/benchmark_gallery_2d.png`: all 2D continuous benchmark landscapes.
- `gallery/benchmark_gallery_3d_slices_x3_025.png`: 3D benchmark slices at `x3 = 0.25`.
- `gallery/benchmark_gallery_3d_slices.png`: 3D benchmark slices at `x3 = 0.50`.
- `gallery/benchmark_gallery_3d_slices_x3_075.png`: 3D benchmark slices at `x3 = 0.75`.

Regenerate the figures from the repository root with:

```bash
python -m examples.phase2_gallery
```

These are objective-landscape visualizations, not optimization results.
