"""Synthetic difference-in-differences, following Arkhangelsky et al. (2021).

The Frank-Wolfe algorithm follows the BSD-licensed synthdid reference solver;
see THIRD_PARTY_NOTICES.md for attribution and the complete license.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from .panel import Panel
from .results import Estimate, make_estimate


def _fw(x, y, zeta, initial, max_iter, min_decrease):
    """Reference exact-line-search Frank-Wolfe with a profiled intercept."""
    x = x - x.mean(axis=0)
    y = y - y.mean()
    w = initial.copy()
    eta = len(y) * zeta**2
    previous = np.inf
    value = np.inf
    converged = False
    for iteration in range(max_iter):
        fitted = x @ w
        gradient = x.T @ (fitted - y) + eta * w
        vertex = np.argmin(gradient)
        direction = -w.copy()
        direction[vertex] += 1
        change = x[:, vertex] - fitted
        denominator = change @ change + eta * (direction @ direction)
        step = 0.0 if denominator <= 0 else np.clip(-gradient @ direction / denominator, 0, 1)
        w += step * direction
        error = x @ w - y
        value = zeta**2 * (w @ w) + error @ error / len(y)
        # Upstream always executes at least two iterations.
        if iteration >= 1 and previous - value <= min_decrease**2:
            converged = True
            break
        previous = value
    gradient = 2 * (x.T @ (x @ w - y) / len(y) + zeta**2 * w)
    return w, {
        "iterations": iteration + 1,
        "objective": float(value),
        "dual_gap": float(gradient @ w - gradient.min()),
        "stopping_rule_met": converged,
    }


@dataclass
class SyntheticDID:
    """SDID for a balanced panel with simultaneous treatment onset.

    Defaults reproduce ``synthdid_estimate`` without covariates: unit penalty
    ``eta_omega * noise``, time penalty ``eta_lambda * noise``, intercepts in
    both weight fits, and two-stage sparse Frank-Wolfe. ``noise`` defaults to
    the sample SD of donor pre-treatment first differences. ``eta_omega``
    defaults to ``(n_treated * n_post)**0.25``. Penalties are squared in the
    objective. A small objective decrease is a stopping rule, not a global
    optimality certificate; the final dual gaps are returned in metadata.
    """

    eta_omega: float | None = None
    eta_lambda: float = 1e-6
    noise: float | None = None
    max_iter: int = 10000
    min_decrease: float | None = None
    sparsify: bool = True

    def fit(self, panel: Panel) -> Estimate:
        """Estimate unit/time weights and a corrected synthetic outcome path."""
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 2:
            raise ValueError("max_iter must be an integer >= 2")
        differences = np.diff(panel.control[: panel.n_pre], axis=0).ravel()
        if self.noise is None and len(differences) < 2:
            raise ValueError("At least two donor first differences are needed to estimate noise")
        noise = float(np.std(differences, ddof=1)) if self.noise is None else self.noise
        eta = (panel.n_treated * panel.n_post) ** 0.25 if self.eta_omega is None else self.eta_omega
        decrease = 1e-5 * noise if self.min_decrease is None else self.min_decrease
        for name, value in [
            ("noise", noise),
            ("eta_omega", eta),
            ("eta_lambda", self.eta_lambda),
            ("min_decrease", decrease),
        ]:
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

        def solve(x, y, penalty):
            start = np.full(x.shape[1], 1 / x.shape[1])
            if self.sparsify:
                start, _ = _fw(x, y, penalty, start, 100, decrease)
                start[start <= start.max() / 4] = 0
                start /= start.sum()
            w, info = _fw(x, y, penalty, start, self.max_iter, decrease)
            if not info["stopping_rule_met"]:
                warnings.warn(
                    "SDID reached max_iter before the objective-decrease stopping rule",
                    RuntimeWarning,
                    stacklevel=3,
                )
            return w, info

        c, y, t = panel.control, panel.treated, panel.n_pre
        omega, omega_info = solve(c[:t], y[:t], eta * noise)
        lam, lambda_info = solve(c[:t].T, c[t:].mean(axis=0), self.eta_lambda * noise)
        offset = lam @ (y[:t] - c[:t] @ omega)
        return make_estimate(
            panel,
            "SyntheticDID",
            omega,
            intercept=offset,
            time_weights=lam,
            metadata={
                "noise": float(noise),
                "zeta_omega": float(eta * noise),
                "zeta_lambda": float(self.eta_lambda * noise),
                "omega_solver": omega_info,
                "lambda_solver": lambda_info,
            },
        )


@dataclass
class DifferenceInDifferences:
    """Unweighted two-way difference-in-differences for a treated block."""

    def fit(self, panel: Panel) -> Estimate:
        """Return a donor-mean path with the pre-treatment mean gap removed."""
        w = np.full(panel.n_control, 1 / panel.n_control)
        lam = np.full(panel.n_pre, 1 / panel.n_pre)
        offset = lam @ (panel.treated[: panel.n_pre] - panel.control[: panel.n_pre] @ w)
        return make_estimate(panel, "DifferenceInDifferences", w, offset, lam)
