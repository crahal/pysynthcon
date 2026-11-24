"""Synthetic difference-in-differences, following Arkhangelsky et al. (2021).

The Frank-Wolfe algorithm follows the BSD-licensed synthdid reference solver;
see THIRD_PARTY_NOTICES.md for attribution and the complete license.
"""

import hashlib
import warnings
from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import norm

from .panel import Panel
from .results import Estimate, make_estimate


def _fw(x, y, zeta, initial, max_iter, min_decrease, intercept=True):
    """Reference exact-line-search Frank-Wolfe with a profiled intercept."""
    if intercept:
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

    def _settings(self, panel):
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
        return {
            "noise": float(noise),
            "zeta_omega": float(eta * noise),
            "zeta_lambda": float(self.eta_lambda * noise),
            "min_decrease": float(decrease),
            "max_iter": int(self.max_iter),
        }

    @staticmethod
    def _estimate(
        panel,
        settings,
        *,
        omega_start=None,
        lambda_start=None,
        sparsify=True,
        omega_intercept=True,
        fixed_lambda=None,
        method="SyntheticDID",
        warn=True,
    ):
        def solve(x, y, penalty, initial, intercept=True):
            start = np.full(x.shape[1], 1 / x.shape[1]) if initial is None else initial.copy()
            if sparsify:
                start, _ = _fw(x, y, penalty, start, 100, settings["min_decrease"], intercept)
                start[start <= start.max() / 4] = 0
                start /= start.sum()
            w, info = _fw(
                x, y, penalty, start, settings["max_iter"], settings["min_decrease"], intercept
            )
            if warn and not info["stopping_rule_met"]:
                warnings.warn(
                    "SDID reached max_iter before the objective-decrease stopping rule",
                    RuntimeWarning,
                    stacklevel=3,
                )
            return w, info

        c, y, t = panel.control, panel.treated, panel.n_pre
        omega, omega_info = solve(
            c[:t], y[:t], settings["zeta_omega"], omega_start, omega_intercept
        )
        if fixed_lambda is None:
            lam, lambda_info = solve(
                c[:t].T, c[t:].mean(axis=0), settings["zeta_lambda"], lambda_start
            )
        else:
            lam, lambda_info = fixed_lambda, {"fixed": True}
        offset = lam @ (y[:t] - c[:t] @ omega)
        return make_estimate(
            panel,
            method,
            omega,
            intercept=offset,
            time_weights=lam,
            metadata={
                **settings,
                "omega_solver": omega_info,
                "lambda_solver": lambda_info,
                "omega_intercept": omega_intercept,
                "sparsify": sparsify,
                "panel_digest": hashlib.sha256(panel.y.tobytes()).hexdigest(),
            },
        )

    def fit(self, panel: Panel) -> Estimate:
        """Estimate unit/time weights and a corrected synthetic outcome path."""
        return self._estimate(panel, self._settings(panel), sparsify=self.sparsify)

    def fit_sc(self, panel: Panel, *, eta_omega: float = 1e-6) -> Estimate:
        """Fit exactly the no-covariate ``synthdid::sc_estimate`` specification.

        This reference SC uses an infinitesimal donor penalty, no donor-fit
        intercept and all-zero time weights. It reproduces the SC comparator
        in the SDID paper, which is distinct from SCM with fitted predictor
        weights. The explicit ``eta_omega`` overrides this object's SDID penalty.
        """
        settings = replace(self, eta_omega=eta_omega)._settings(panel)
        return self._estimate(
            panel,
            settings,
            sparsify=self.sparsify,
            omega_intercept=False,
            fixed_lambda=np.zeros(panel.n_pre),
            method="SynthdidSC",
        )

    def inference(
        self,
        panel: Panel,
        *,
        method: str = "bootstrap",
        replications: int = 200,
        random_state=None,
        estimate: Estimate | None = None,
        resamples=None,
    ) -> "SDIDInference":
        """Author-software versions of Algorithms 2–4 for no-covariate SDID.

        Bootstrap samples all units with replacement and rejects samples missing
        either treatment group. Placebo randomly partitions the original controls
        into pseudo-controls and pseudo-treated units. Both fully re-estimate
        donor and time weights, retain original penalties and stopping settings,
        and warm-start as in R synthdid. Jackknife leaves each unit out and keeps
        normalized fitted weights fixed. Its variance centers on the mean of
        leave-one-out estimates, matching R ``synthdid``; literal Algorithm 3
        in the paper instead centers on the original full-sample estimate.

        ``resamples`` optionally supplies exactly ``replications`` zero-based
        index vectors, enabling draw-for-draw comparison with another runtime.
        Random seeds alone do not yield identical draws between NumPy and R.
        For one treated unit only placebo inference is defined.
        """
        if method not in {"bootstrap", "jackknife", "placebo"}:
            raise ValueError("method must be 'bootstrap', 'jackknife', or 'placebo'")
        if method != "jackknife" and (
            isinstance(replications, (bool, np.bool_))
            or not isinstance(replications, (int, np.integer))
            or replications < 2
        ):
            raise ValueError("replications must be an integer >= 2")
        if method in {"bootstrap", "jackknife"} and panel.n_treated == 1:
            raise ValueError(f"{method} inference needs at least two treated units; use placebo")
        if method == "placebo" and panel.n_control <= panel.n_treated:
            raise ValueError("Placebo inference needs more controls than treated units")
        if method == "jackknife" and resamples is not None:
            raise ValueError("Jackknife uses deterministic leave-one-unit-out samples")
        fitted = self.fit(panel) if estimate is None else estimate
        if (
            not isinstance(fitted, Estimate)
            or fitted.method != "SyntheticDID"
            or fitted.n_pre != panel.n_pre
            or not fitted.observed.index.equals(panel.times)
            or not np.array_equal(fitted.observed.to_numpy(), panel.treated)
            or tuple(fitted.weights.index) != panel.units[: panel.n_control]
            or fitted.time_weights is None
            or not fitted.time_weights.index.equals(panel.times[: panel.n_pre])
            or fitted.metadata.get("panel_digest") != hashlib.sha256(panel.y.tobytes()).hexdigest()
        ):
            raise ValueError("estimate must be an SDID fit of this panel")
        settings = {
            key: fitted.metadata[key]
            for key in ("noise", "zeta_omega", "zeta_lambda", "min_decrease", "max_iter")
        }
        omega, lam = fitted.weights.to_numpy(), fitted.time_weights.to_numpy()
        if any(
            not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-10)
            for weights in (omega, lam)
        ):
            raise ValueError("estimate must contain finite simplex donor and time weights")
        if method == "jackknife" and np.count_nonzero(omega) == 1:
            raise ValueError("Fixed-weight jackknife needs at least two nonzero donor weights")
        n = panel.y.shape[1]
        rng = np.random.default_rng(random_state)
        draws = []
        if method == "jackknife":
            draws = [np.delete(np.arange(n), i) for i in range(n)]
        elif resamples is not None:
            raw = np.asarray(resamples)
            expected_width = n if method == "bootstrap" else panel.n_control
            if raw.shape != (replications, expected_width) or not np.issubdtype(
                raw.dtype, np.integer
            ):
                raise ValueError(
                    "resamples must be an integer matrix with one valid index row per replication"
                )
            for row in raw:
                if method == "bootstrap":
                    if (
                        np.any(row < 0)
                        or np.any(row >= n)
                        or np.all(row < panel.n_control)
                        or np.all(row >= panel.n_control)
                    ):
                        raise ValueError(
                            "Every bootstrap sample must contain both treatment groups"
                        )
                    draws.append(np.sort(row))
                else:
                    if not np.array_equal(np.sort(row), np.arange(panel.n_control)):
                        raise ValueError("Every placebo sample must permute all original controls")
                    draws.append(row.copy())
        else:
            while len(draws) < replications:
                if method == "placebo":
                    draws.append(rng.permutation(panel.n_control))
                else:
                    row = rng.integers(n, size=n)
                    if np.any(row < panel.n_control) and np.any(row >= panel.n_control):
                        draws.append(np.sort(row))
        values = []
        convergence = []
        for row in draws:
            n_control = (
                len(row) - panel.n_treated
                if method == "placebo"
                else int(np.count_nonzero(row < panel.n_control))
            )
            initial_omega = omega[row[:n_control]].copy()
            total = initial_omega.sum()
            initial_omega = initial_omega / total if total else np.full(n_control, 1 / n_control)
            # Repeated bootstrap indices get unique labels; their outcome columns
            # and duplicated initial weights retain the reference multiplicity.
            sample = Panel(panel.y[:, row], n_control, panel.n_pre, panel.times)
            if method == "jackknife":
                gap = sample.treated - sample.control @ initial_omega
                values.append(float(gap[panel.n_pre :].mean() - lam @ gap[: panel.n_pre]))
            else:
                fit = self._estimate(
                    sample,
                    settings,
                    omega_start=initial_omega,
                    lambda_start=lam,
                    sparsify=True,
                    warn=False,
                )
                values.append(fit.att)
                convergence.append(
                    {"omega": fit.metadata["omega_solver"], "lambda": fit.metadata["lambda_solver"]}
                )
        capped = sum(
            not (info["omega"]["stopping_rule_met"] and info["lambda"]["stopping_rule_met"])
            for info in convergence
        )
        if capped:
            warnings.warn(
                f"SDID inference: {capped} of {len(values)} draws reached max_iter before "
                "the objective-decrease stopping rule; inspect solver_diagnostics",
                RuntimeWarning,
                stacklevel=2,
            )
        values = np.asarray(values)
        # The R reference's sqrt((B-1)/B)*sd(draws) is the population SD.
        # R's fixed-weight jackknife multiplies that variance by n-1. This
        # uses the deletion mean, unlike the full-estimate center printed in
        # the paper's Algorithm 3 and appendix equation (11.3).
        variance = float(np.var(values) * (n - 1 if method == "jackknife" else 1))
        return SDIDInference(
            fitted.att,
            variance,
            method,
            values,
            np.asarray(draws),
            {
                "reference": "synthdid@70c1ce3eac58e28c30b67435ca377bb48baa9b8a",
                "variance_centering": (
                    "leave_one_out_mean" if method == "jackknife" else "resample_mean"
                ),
                "settings": settings,
                "solver_diagnostics": convergence,
            },
        )


@dataclass
class SDIDInference:
    """SDID reference variance estimate and the unit-resampling draws behind it.

    Confidence intervals use the reference asymptotic normal approximation;
    neither the draw distribution nor the interval is a randomization test.
    """

    estimate: float
    variance: float
    method: str
    draws: np.ndarray
    resamples: np.ndarray
    metadata: dict

    @property
    def standard_error(self) -> float:
        """Square root of the variance estimate."""
        return float(np.sqrt(self.variance))

    def confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Return the normal interval at the specified coverage level."""
        if not np.isfinite(level) or not 0 < level < 1:
            raise ValueError("level must be strictly between zero and one")
        width = norm.ppf((1 + level) / 2) * self.standard_error
        return float(self.estimate - width), float(self.estimate + width)


@dataclass
class DifferenceInDifferences:
    """Unweighted two-way difference-in-differences for a treated block."""

    def fit(self, panel: Panel) -> Estimate:
        """Return a donor-mean path with the pre-treatment mean gap removed."""
        w = np.full(panel.n_control, 1 / panel.n_control)
        lam = np.full(panel.n_pre, 1 / panel.n_pre)
        offset = lam @ (panel.treated[: panel.n_pre] - panel.control[: panel.n_pre] @ w)
        return make_estimate(panel, "DifferenceInDifferences", w, offset, lam)
