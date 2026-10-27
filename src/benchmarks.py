test"""Deterministic maximization benchmarks on the unit hypercube.

Every objective accepts a point of shape (d,) or a batch (n, d). The latent
objective is noise-free; add measurement noise in the runner, not here.
Random-field instances are constructed once and remain fixed.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
Array = np.ndarray

@dataclass(frozen=True)
class Benchmark:
    name: str
    dimension: int
    evaluate: Callable[[Array], Array]
    description: str
    reference_x: Array | None = None
    reference_value: float | None = None

    def __call__(self, x: Array) -> Array | float:
        points = np.asarray(x, dtype=float)
        single = points.ndim == 1
        if single:
            points = points[None, :]
        if points.ndim != 2 or points.shape[1] != self.dimension:
            raise ValueError(f"expected shape (n, {self.dimension}) or ({self.dimension},)")
        if not np.all(np.isfinite(points)):
            raise ValueError("inputs must be finite")
        if np.any((points < 0.0) | (points > 1.0)):
            raise ValueError("inputs must lie in [0, 1]^d")
        values = np.asarray(self.evaluate(points), dtype=float).reshape(-1)
        return float(values[0]) if single else values

def _check_dimension(d: int) -> None:
    if d not in (2, 3):
        raise ValueError("the Phase 2 bank currently supports dimensions 2 and 3")

def _gaussian(points: Array, center: Array, width: float) -> Array:
    return np.exp(-np.sum((points - center) ** 2, axis=1) / (2.0 * width**2))

def _random_field(d: int, seed: int, lengthscale: float, nu: float, n_features: int) -> Callable[[Array], Array]:
    """Finite random Fourier approximation to a stationary Matérn-like field.

    Frequencies follow the Student-t spectral representation of a Matérn
    covariance, scaled by the correlation length. This is an approximation,
    not an exact Matérn sample path.
    """
    if n_features < 16:
        raise ValueError("n_features must be at least 16")
    rng = np.random.default_rng(seed)
    normal = rng.normal(size=(n_features, d))
    chi = rng.chisquare(df=2.0 * nu, size=(n_features, 1))
    frequencies = normal / np.sqrt(chi / (2.0 * nu)) / lengthscale
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
    weights = rng.normal(size=n_features)
    denominator = np.sum(np.abs(weights))
    def evaluate(points: Array) -> Array:
        waves = np.cos(2.0 * np.pi * points @ frequencies.T + phases)
        return waves @ weights / denominator
    return evaluate

def get_benchmark(name: str, dimension: int, *, field_seed: int = 0, n_features: int = 256) -> Benchmark:
    """Create one named benchmark. All inputs are normalized to [0, 1]^d."""
    _check_dimension(dimension)
    d = dimension
    center = np.linspace(0.32, 0.68, d)
    if name == "quadratic_peak":
        return Benchmark(name, d, lambda x: 1.0 - np.sum((x - center) ** 2, axis=1), "Single smooth, broad peak.", center.copy(), 1.0)
    if name == "rotated_ellipsoid":
        rng = np.random.default_rng(41 + d)
        rotation, _ = np.linalg.qr(rng.normal(size=(d, d)))
        scales = np.geomspace(1.0, 0.08, d)
        def ellipsoid(x: Array) -> Array:
            z = (x - center) @ rotation
            return 1.0 - np.sum((z / scales) ** 2, axis=1) / np.sum(1.0 / scales**2)
        return Benchmark(name, d, ellipsoid, "Smooth peak with coupled, strongly anisotropic axes.", center.copy(), 1.0)
    if name == "rippled_peak":
        def rippled(x: Array) -> Array:
            broad = 1.0 - np.sum((x - center) ** 2, axis=1)
            ripple = 0.055 * np.sum(np.cos(8.0 * np.pi * x), axis=1) / d
            return broad + ripple
        return Benchmark(name, d, rippled, "Broad peak plus small-scale periodic variation; optimum is numerical.")
    if name == "competing_gaussian_peaks":
        centers = np.stack([np.full(d, v) for v in (0.22, 0.52, 0.81)])
        heights = np.array([0.91, 1.0, 0.96])
        def competing(x: Array) -> Array:
            peaks = np.stack([h * _gaussian(x, c, 0.105) for h, c in zip(heights, centers)])
            return peaks.max(axis=0)
        return Benchmark(name, d, competing, "Three separated near-tied peaks; tests global mode identification.", centers[1].copy(), 1.0)
    if name == "hidden_narrow_peak":
        broad_center, narrow_center = np.full(d, 0.24), np.full(d, 0.82)
        def hidden(x: Array) -> Array:
            return np.maximum(0.58 * _gaussian(x, broad_center, 0.25), _gaussian(x, narrow_center, 0.035))
        return Benchmark(name, d, hidden, "Broad attractive basin and a better, narrow, remote peak.", narrow_center, 1.0)
    if name == "rosenbrock":
        def rosenbrock(x: Array) -> Array:
            z = 4.0 * x - 2.0
            loss = np.sum(100.0 * (z[:, 1:] - z[:, :-1] ** 2) ** 2 + (1.0 - z[:, :-1]) ** 2, axis=1)
            return -loss / (1.0 + loss)
        optimum = np.full(d, 0.75)
        return Benchmark(name, d, rosenbrock, "Negative Rosenbrock loss: a narrow curved optimum.", optimum, 0.0)
    if name == "ackley":
        def ackley(x: Array) -> Array:
            z = 10.0 * x - 5.0
            a = -20.0 * np.exp(-0.2 * np.sqrt(np.mean(z**2, axis=1)))
            b = -np.exp(np.mean(np.cos(2.0 * np.pi * z), axis=1))
            return -(a + b + 20.0 + np.e)
        return Benchmark(name, d, ackley, "Negative Ackley: dominant center with local structure and flat outer region.", np.full(d, 0.5), 0.0)
    if name == "rastrigin":
        def rastrigin(x: Array) -> Array:
            z = 10.24 * x - 5.12
            loss = 10.0 * d + np.sum(z**2 - 10.0 * np.cos(2.0 * np.pi * z), axis=1)
            return -loss / (1.0 + loss)
        return Benchmark(name, d, rastrigin, "Negative scaled Rastrigin: repeated multimodal structure.", np.full(d, 0.5), 0.0)
    if name.startswith("grf_"):
        settings = {"grf_long": (0.35, 2.5, "Long-scale, relatively smooth random field."), "grf_short": (0.09, 2.5, "Short-scale random field with many local features."), "grf_rough": (0.18, 0.7, "Rougher Matérn-like field (lower smoothness).")}
        if name not in settings:
            raise KeyError(f"unknown benchmark: {name}")
        lengthscale, nu, description = settings[name]
        evaluator = _random_field(d, field_seed, lengthscale, nu, n_features)
        return Benchmark(name, d, evaluator, f"{description} Fixed field_seed={field_seed}, features={n_features}; finite spectral approximation; no analytic optimum.")
    if name == "branin":
        if d != 2:
            raise ValueError("Branin is a 2D benchmark")
        def branin(x: Array) -> Array:
            xx, yy = 15.0 * x[:, 0] - 5.0, 15.0 * x[:, 1]
            a, b, c = 1.0, 5.1 / (4.0 * np.pi**2), 5.0 / np.pi
            r, s, t = 6.0, 10.0, 1.0 / (8.0 * np.pi)
            value = a * (yy - b * xx**2 + c * xx - r) ** 2 + s * (1.0 - t) * np.cos(xx) + s
            return -value
        optimum = np.array([[(-np.pi + 5.0) / 15.0, 12.275 / 15.0], [(np.pi + 5.0) / 15.0, 2.275 / 15.0], [(9.42478 + 5.0) / 15.0, 2.475 / 15.0]])
        return Benchmark(name, d, branin, "Negative Branin-Hoo; three equivalent global maxima.", optimum[0], -0.3978873577)
    if name == "hartmann3":
        if d != 3:
            raise ValueError("Hartmann-3 is a 3D benchmark")
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([[3.0, 10.0, 30.0], [0.1, 10.0, 35.0], [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]])
        P = 1e-4 * np.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]])
        def hartmann(x: Array) -> Array:
            inner = np.sum(A[None, :, :] * (x[:, None, :] - P[None, :, :]) ** 2, axis=2)
            return np.sum(alpha[None, :] * np.exp(-inner), axis=1)
        return Benchmark(name, d, hartmann, "Standard Hartmann-3 maximization anchor; optimum is numerical.")
    raise KeyError(f"unknown benchmark: {name}")

FAMILIES = ("quadratic_peak", "rotated_ellipsoid", "rippled_peak", "competing_gaussian_peaks", "rosenbrock", "ackley", "rastrigin", "hidden_narrow_peak", "grf_long", "grf_short", "grf_rough")

def benchmark_suite(*, field_seed: int = 2026, n_features: int = 256) -> list[Benchmark]:
    """Return the 24 Phase 2 cases: 11 families in 2D and 3D plus two anchors."""
    suite = [get_benchmark(name, d, field_seed=field_seed, n_features=n_features) for name in FAMILIES for d in (2, 3)]
    suite.extend([get_benchmark("branin", 2), get_benchmark("hartmann3", 3)])
    return suite
