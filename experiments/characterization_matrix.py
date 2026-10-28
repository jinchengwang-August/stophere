"""Controlled GP/acquisition/hyperparameter/stopping-rule characterization.

This is a transparent NumPy exact-GP benchmark harness, not BoTorch. It runs
one-factor-at-a-time variants on fixed finite catalogues, then applies stop
rules to the resulting complete trajectories. Run:

    MPLBACKEND=Agg python -m experiments.characterization_matrix
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from itertools import product
from math import erf, pi, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.benchmarks import get_benchmark


@dataclass(frozen=True)
class System:
    name: str
    benchmark: str
    dimension: int
    base_lengthscale: float


SYSTEMS = (
    System("smooth_2d", "quadratic_peak", 2, 0.22),
    System("anisotropic_2d", "rotated_ellipsoid", 2, 0.22),
    System("rippled_2d", "rippled_peak", 2, 0.16),
    System("competing_modes_2d", "competing_gaussian_peaks", 2, 0.20),
    System("hidden_narrow_2d", "hidden_narrow_peak", 2, 0.16),
    System("curved_valley_3d", "rosenbrock", 3, 0.28),
    System("flat_multimodal_3d", "ackley", 3, 0.28),
    System("rough_field_2d", "grf_rough", 2, 0.14),
    System("ising_mrf_3x3", "mrf_ising_3x3", 9, 1.35),
)

N_SEEDS = 5
N_INITIAL = 5
BUDGET = 24
SUCCESS_EPSILON = 0.05
FIELD_SEED = 2026
JITTER = 1e-8
NOISE_LEVELS = (0.0, 0.03)  # Gaussian observation SD as a fraction of catalogue range.


def catalogue(system: System) -> np.ndarray:
    if system.benchmark == "mrf_ising_3x3":
        return np.asarray(list(product((0.0, 1.0), repeat=system.dimension)), dtype=float)
    n_axis = 21 if system.dimension == 2 else 9
    axis = np.linspace(0.0, 1.0, n_axis)
    mesh = np.meshgrid(*([axis] * system.dimension), indexing="ij")
    return np.column_stack([m.ravel() for m in mesh])


def kernel_matrix(x1: np.ndarray, x2: np.ndarray, kind: str, lengthscale: float) -> np.ndarray:
    r = np.sqrt(np.maximum(np.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=2), 0.0))
    q = sqrt(3.0) * r / lengthscale
    if kind == "rbf":
        return np.exp(-0.5 * (r / lengthscale) ** 2)
    if kind == "matern32":
        return (1.0 + q) * np.exp(-q)
    if kind == "matern52":
        q5 = sqrt(5.0) * r / lengthscale
        return (1.0 + q5 + q5**2 / 3.0) * np.exp(-q5)
    raise ValueError(f"unknown kernel: {kind}")


def normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))


def posterior(train_x: np.ndarray, train_y: np.ndarray, candidates: np.ndarray,
              kernel: str, lengthscale: float, noise_fraction: float):
    y_mean = float(train_y.mean())
    y_scale = max(float(train_y.std()), 1e-8)
    y = (train_y - y_mean) / y_scale
    k = kernel_matrix(train_x, train_x, kernel, lengthscale)
    # noise_fraction is defined relative to the observed response scale.
    k.flat[::len(k) + 1] += max(noise_fraction**2, JITTER)
    chol = np.linalg.cholesky(k)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
    cross = kernel_matrix(train_x, candidates, kernel, lengthscale)
    mu_z = cross.T @ alpha
    v = np.linalg.solve(chol, cross)
    var_z = np.maximum(1.0 - np.sum(v * v, axis=0), 0.0)
    mu = y_mean + y_scale * mu_z
    sigma = y_scale * np.sqrt(var_z)
    return mu, sigma, y_mean, y_scale


def choose_next(mu: np.ndarray, sigma: np.ndarray, best_y: float,
                acquisition: str, beta: float, used: np.ndarray) -> int:
    safe_sigma = np.maximum(sigma, 1e-12)
    improvement = mu - best_y
    z = improvement / safe_sigma
    if acquisition == "ucb":
        score = mu + sqrt(beta) * sigma
    elif acquisition == "ei":
        score = improvement * normal_cdf(z) + safe_sigma * np.exp(-0.5 * z**2) / sqrt(2 * pi)
    elif acquisition == "pi":
        score = normal_cdf(z)
    else:
        raise ValueError(acquisition)
    score[used] = -np.inf
    return int(np.argmax(score))


def configuration_rows(system: System):
    # One-factor-at-a-time contrasts around a declared nominal RBF/UCB setup.
    configs = [("nominal", "rbf", 1.0, "ucb", 2.0, 0.0)]
    configs.extend((f"kernel={kernel}", kernel, 1.0, "ucb", 2.0, 0.0)
                   for kernel in ("matern32", "matern52"))
    configs.extend((f"lengthscale_x{mult:g}", "rbf", mult, "ucb", 2.0, 0.0)
                   for mult in (0.5, 2.0))
    configs.extend((f"beta={beta:g}", "rbf", 1.0, "ucb", beta, 0.0)
                   for beta in (0.5, 8.0))
    configs.extend((f"acquisition={acq}", "rbf", 1.0, acq, 2.0, 0.0)
                   for acq in ("ei", "pi"))
    # Noise is an observation-process condition; the GP uses the matching
    # known likelihood variance. This is a controlled robustness scenario.
    configs.append(("observation_noise=3pct", "rbf", 1.0, "ucb", 2.0, 0.03))
    # One explicit interaction checks whether rougher kernel + longer scale
    # behaves like either main effect alone.
    configs.append(("matern32_lengthscale_x2", "matern32", 2.0, "ucb", 2.0, 0.0))
    configs.append(("random_baseline", "rbf", 1.0, "random", 2.0, 0.0))
    return configs


def run_trajectory(system: System, config: tuple, seed: int) -> list[dict]:
    label, kernel, ls_mult, acq, beta, noise_fraction = config
    objective = get_benchmark(system.benchmark, system.dimension, field_seed=FIELD_SEED)
    points = catalogue(system)
    latent = np.asarray(objective(points), dtype=float)
    best_ref = float(latent.max())
    value_range = float(latent.max() - latent.min()) or 1.0
    ls = system.base_lengthscale * ls_mult
    design_rng = np.random.default_rng(np.random.SeedSequence([seed, 3107]))
    obs_rng = np.random.default_rng(np.random.SeedSequence([seed, 9901]))
    policy_rng = np.random.default_rng(np.random.SeedSequence([seed, 4201]))
    initial_ids = design_rng.choice(len(points), N_INITIAL, replace=False)
    used = np.zeros(len(points), dtype=bool)
    used[initial_ids] = True
    train_x = points[initial_ids].copy()
    train_y = latent[initial_ids].copy()
    if noise_fraction:
        train_y += obs_rng.normal(0.0, noise_fraction * value_range, size=len(train_y))
    trace = []

    def observe(n_eval: int):
        mu, sigma, _, y_scale = posterior(train_x, train_y, points, kernel, ls,
                                           noise_fraction * value_range / max(float(train_y.std()), 1e-8))
        best_y = float(train_y.max())
        ucb_gap = max(float(np.max(mu + 1.96 * sigma) - best_y), 0.0)
        p_improve = float(np.max(normal_cdf((mu - best_y - 0.01 * y_scale) / np.maximum(sigma, 1e-12))))
        best_latent = float(latent[np.flatnonzero(np.all(points == train_x[np.argmax(train_y)], axis=1))[0]])
        regret = max(best_ref - best_latent, 0.0)
        trace.append({
            "system": system.name, "config": label, "kernel": kernel,
            "lengthscale_multiplier": ls_mult, "acquisition": acq,
            "beta": beta, "observation_noise_fraction": noise_fraction,
            "seed": seed, "n_eval": n_eval,
            "best_observed": best_y, "best_latent_at_incumbent": best_latent,
            "regret": regret, "normalized_regret": regret / value_range,
            "posterior_ucb_gap_over_y_sd": ucb_gap / y_scale,
            "max_marginal_p_improve": p_improve,
        })
        return mu, sigma

    mu, sigma = observe(len(train_y))
    while len(train_y) < BUDGET:
        if acq == "random":
            index = int(policy_rng.choice(np.flatnonzero(~used)))
        else:
            index = choose_next(mu, sigma, float(train_y.max()), acq, beta, used)
        used[index] = True
        x_new = points[index:index + 1]
        y_new = np.asarray(objective(x_new), dtype=float)
        if noise_fraction:
            y_new += obs_rng.normal(0.0, noise_fraction * value_range, size=1)
        train_x = np.vstack((train_x, x_new))
        train_y = np.concatenate((train_y, y_new))
        mu, sigma = observe(len(train_y))
    return trace


def stopping_summaries(traces: list[dict]) -> list[dict]:
    summaries = []
    rules = ("fixed_budget", "patience_5", "global_ucb_gap_0.05", "max_p_improve_0.05")
    groups = sorted({(r["system"], r["config"], r["seed"]) for r in traces})
    for system, config, seed in groups:
        run = sorted((r for r in traces if (r["system"], r["config"], r["seed"]) == (system, config, seed)), key=lambda r: r["n_eval"])
        for rule in rules:
            stopped = run[-1]
            if rule == "patience_5":
                for i, row in enumerate(run):
                    if row["n_eval"] < N_INITIAL + 5:
                        continue
                    prior = run[max(0, i - 5):i]
                    scale = max(float(np.std([p["best_observed"] for p in prior])), 1e-8)
                    if row["best_observed"] - prior[0]["best_observed"] <= 0.01 * scale:
                        stopped = row
                        break
            elif rule == "global_ucb_gap_0.05":
                candidates = [r for r in run if r["n_eval"] >= N_INITIAL + 2 and r["posterior_ucb_gap_over_y_sd"] <= 0.05]
                if candidates:
                    stopped = candidates[0]
            elif rule == "max_p_improve_0.05":
                candidates = [r for r in run if r["n_eval"] >= N_INITIAL + 2 and r["max_marginal_p_improve"] <= 0.05]
                if candidates:
                    stopped = candidates[0]
            summaries.append({
                "system": system, "config": config, "seed": seed, "stopping_rule": rule,
                "stop_at_evals": stopped["n_eval"],
                "final_normalized_regret": stopped["normalized_regret"],
                "success_le_0.05": int(stopped["normalized_regret"] <= SUCCESS_EPSILON),
                "false_stop": int(rule != "fixed_budget" and stopped["normalized_regret"] > SUCCESS_EPSILON),
                "evaluations_saved": BUDGET - stopped["n_eval"],
            })
    return summaries


def group_summary(traces: list[dict]) -> list[dict]:
    groups = sorted({(r["system"], r["config"]) for r in traces})
    out = []
    for system, config in groups:
        rows = [r for r in traces if r["system"] == system and r["config"] == config and r["n_eval"] == BUDGET]
        vals = np.asarray([r["normalized_regret"] for r in rows])
        out.append({
            "system": system, "config": config, "seeds": len(rows),
            "mean_final_normalized_regret": float(vals.mean()),
            "sd_final_normalized_regret": float(vals.std(ddof=1)),
            "success_rate_le_0.05": float(np.mean(vals <= SUCCESS_EPSILON)),
        })
    return out


def paired_deltas(traces: list[dict]) -> list[dict]:
    """Paired seed-wise regret changes versus the nominal RBF/UCB baseline."""
    out = []
    systems = sorted({r["system"] for r in traces})
    configs = sorted({r["config"] for r in traces} - {"nominal"})
    for system in systems:
        base = {r["seed"]: r["normalized_regret"] for r in traces
                if r["system"] == system and r["config"] == "nominal" and r["n_eval"] == BUDGET}
        for config in configs:
            candidate = {r["seed"]: r["normalized_regret"] for r in traces
                         if r["system"] == system and r["config"] == config and r["n_eval"] == BUDGET}
            if not candidate:
                continue
            delta = np.asarray([candidate[s] - base[s] for s in sorted(candidate)])
            out.append({
                "system": system, "config": config, "paired_seeds": len(delta),
                "mean_delta_normalized_regret_vs_nominal": float(delta.mean()),
                "sd_delta_normalized_regret": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                "fraction_seeds_improved": float(np.mean(delta < -1e-12)),
            })
    return out


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(summary: list[dict], stop_summary: list[dict], path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    # Kernel effect averaged over all systems; this is descriptive, not pooled
    # inferential evidence because systems have different landscapes.
    kernel_rows = [r for r in summary if r["config"] == "nominal" or r["config"].startswith("kernel=")]
    kernels = ("nominal", "kernel=matern32", "kernel=matern52")
    means, sds = [], []
    for kernel in kernels:
        vals = [r["mean_final_normalized_regret"] for r in kernel_rows if r["config"] == kernel]
        means.append(float(np.mean(vals)))
        sds.append(float(np.std(vals, ddof=1)))
    axes[0].bar(["RBF", "Matérn 3/2", "Matérn 5/2"], means, yerr=sds, capsize=4, color=["#4477AA", "#66AADD", "#228833"])
    axes[0].set_ylabel("Mean final normalized regret across systems")
    axes[0].set_title("Kernel family (descriptive average)")
    axes[0].grid(axis="y", alpha=0.2)
    rules = ("fixed_budget", "patience_5", "global_ucb_gap_0.05", "max_p_improve_0.05")
    rule_mean = [np.mean([r["evaluations_saved"] for r in stop_summary if r["stopping_rule"] == rule]) for rule in rules]
    false_rate = [np.mean([r["false_stop"] for r in stop_summary if r["stopping_rule"] == rule]) for rule in rules]
    x = np.arange(len(rules))
    axes[1].bar(x - 0.18, rule_mean, width=0.36, label="Evaluations saved", color="#4477AA")
    ax2 = axes[1].twinx()
    ax2.bar(x + 0.18, false_rate, width=0.36, label="False-stop rate", color="#CC6677")
    axes[1].set_xticks(x, ["budget", "patience", "UCB gap", "P(improve)"], rotation=12)
    axes[1].set_ylabel("Mean evaluations saved")
    ax2.set_ylabel("False-stop rate")
    axes[1].set_title("Stopping-rule trade-off (all optimizer trajectories)")
    axes[1].grid(axis="y", alpha=0.2)
    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[1].legend(h1 + h2, l1 + l2, loc="upper left")
    fig.suptitle("Phase 3: controlled BO characterization pilot", y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    traces = []
    configs = configuration_rows(None)
    for system in SYSTEMS:
        for config in configs:
            for seed in range(N_SEEDS):
                traces.extend(run_trajectory(system, config, seed))
    stop = stopping_summaries(traces)
    summary = group_summary(traces)
    deltas = paired_deltas(traces)
    out = Path("results/phase3/characterization_matrix")
    write_csv(out / "trajectories.csv", traces)
    write_csv(out / "stopping_rules.csv", stop)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "comparison_deltas.csv", deltas)
    make_plot(summary, stop, out / "characterization_summary.png")
    print(f"systems={len(SYSTEMS)} configs={len(configs)} seeds={N_SEEDS} total trajectories={len(SYSTEMS)*len(configs)*N_SEEDS}")
    print(f"budget={BUDGET}, initial={N_INITIAL}, output={out}")
    for rule in ("fixed_budget", "patience_5", "global_ucb_gap_0.05", "max_p_improve_0.05"):
        rows = [r for r in stop if r["stopping_rule"] == rule]
        print(rule, "mean regret", np.mean([r["final_normalized_regret"] for r in rows]),
              "success", np.mean([r["success_le_0.05"] for r in rows]),
              "false stop", np.mean([r["false_stop"] for r in rows]),
              "saved", np.mean([r["evaluations_saved"] for r in rows]))


if __name__ == "__main__":
    main()
