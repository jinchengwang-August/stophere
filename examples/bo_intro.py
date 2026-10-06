"""One-dimensional closed-loop Bayesian optimization with BoTorch.

The dense objective curve is for plotting only. The optimizer receives only
values returned by evaluate_objective at selected points.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

DTYPE = torch.double
BOUNDS = torch.tensor([[0.0], [1.0]], dtype=DTYPE)


def objective(x: torch.Tensor) -> torch.Tensor:
    """Toy maximization objective; x has shape (..., 1)."""
    return torch.sin(6 * torch.pi * x) + 0.4 * x


def fit_model(train_x: torch.Tensor, train_y: torch.Tensor) -> SingleTaskGP:
    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def propose(model: SingleTaskGP, best_f: float) -> torch.Tensor:
    acq = LogExpectedImprovement(model=model, best_f=best_f)
    x_next, _ = optimize_acqf(
        acq, bounds=BOUNDS, q=1, num_restarts=12, raw_samples=256
    )
    return x_next.detach()


def run_bo(seed: int = 17, n_steps: int = 6):
    torch.manual_seed(seed)
    train_x = torch.tensor([[0.08], [0.37], [0.72]], dtype=DTYPE)
    train_y = objective(train_x)
    history = []

    for step in range(n_steps):
        model = fit_model(train_x, train_y)
        x_next = propose(model, float(train_y.max()))
        y_next = objective(x_next)  # one new expensive evaluation
        history.append((x_next.item(), y_next.item(), train_y.max().item()))
        train_x = torch.cat([train_x, x_next])
        train_y = torch.cat([train_y, y_next])

    return train_x, train_y, fit_model(train_x, train_y), history


def make_figure(train_x, train_y, model, output_path: Path):
    grid = torch.linspace(0, 1, 600, dtype=DTYPE).unsqueeze(-1)
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(grid)
        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.sqrt().squeeze(-1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grid.squeeze(), objective(grid).squeeze(), "k--", label="true objective")
    ax.plot(grid.squeeze(), mean, color="#2364aa", label="GP posterior mean")
    ax.fill_between(
        grid.squeeze(), mean - 2 * std, mean + 2 * std,
        color="#2364aa", alpha=0.2, label="latent mean ± 2 SD"
    )
    ax.scatter(train_x.squeeze(), train_y.squeeze(), color="#d1495b", zorder=3,
               label="evaluated points")
    ax.set(xlabel="x", ylabel="f(x)", title="Closed-loop Bayesian optimization")
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    return fig


if __name__ == "__main__":
    x, y, fitted_model, trace = run_bo()
    for i, (x_next, y_next, previous_best) in enumerate(trace, start=1):
        print(f"step={i:02d} x_next={x_next:.5f} y={y_next:.5f} "
              f"previous_best={previous_best:.5f}")
    make_figure(x, y, fitted_model, Path("assets/bo_intro_result.png"))

