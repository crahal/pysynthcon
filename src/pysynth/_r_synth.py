"""Run the authors' installed R Synth package on a checked Python Dataprep."""

import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np


def _fit_in_r(matrices):
    # Each child embeds R once on its main thread. The host Python process
    # retains its own environment, R library paths and console callbacks.
    try:
        from rpy2.rinterface_lib import callbacks
        from rpy2.robjects import FloatVector, baseenv

        exported = baseenv["getExportedValue"]
        synth = exported("Synth", "synth")
        package_description = exported("utils", "packageDescription")
    except Exception as exc:
        raise ImportError(
            "Cannot load the local R Synth reference backend. Install the optional rpy2 "
            "bridge and R package Synth, and select its library with PYSYNTH_R_LIBRARY. "
            f"Loader error: {exc}"
        ) from exc

    args = {}
    for name, values in matrices.items():
        array = np.asarray(values, dtype=float)
        args[name] = baseenv["matrix"](
            FloatVector(array.ravel(order="F")), nrow=array.shape[0], ncol=array.shape[1]
        )
    # Collect the warning/error console stream independently of routine R
    # output. warn=1 delivers warnings during this call in our private process.
    diagnostics = []
    baseenv["options"](warn=1)
    with callbacks.replace_in_module(callbacks, "consolewrite_warnerror", diagnostics.append):
        try:
            fit = synth(**args)
        except Exception:
            sys.stderr.write("".join(diagnostics))
            raise
    numeric = baseenv["as.numeric"]
    return {
        "weights": list(numeric(fit.rx2("solution.w"))),
        "predictor_weights": list(numeric(fit.rx2("solution.v"))),
        "loss_v": numeric(fit.rx2("loss.v"))[0],
        "loss_w": numeric(fit.rx2("loss.w"))[0],
        "reference_package": "Synth",
        "reference_version": package_description("Synth", fields="Version")[0],
        "r_console_diagnostics": "".join(diagnostics).strip(),
        "r_version": baseenv["R.version.string"][0],
        "rpy2_version": version("rpy2"),
    }


def _run_synth(matrices, library):
    env = os.environ.copy()
    if library is not None:
        existing = env.get("R_LIBS")
        env["R_LIBS"] = os.pathsep.join([library, existing]) if existing else library
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        input=json.dumps(matrices, allow_nan=False),
        capture_output=True,
        text=True,
        env=env,
    )
    if child.returncode:
        error = ImportError if child.returncode == 2 else RuntimeError
        raise error(f"R Synth reference fit failed: {child.stderr.strip()}")
    return json.loads(child.stdout)


@dataclass
class ReferenceSynthDataprep:
    """Fit the installed R ``Synth::synth`` with its unmodified default optimizers.

    The Dataprep must specify one treated unit, at least two control units, the
    same outcomes as the supplied panel, its complete pretreatment fitting
    period, and exclusively pretreatment predictors. Its explicit donor order
    enters the original solver; output weights are aligned to the panel labels.
    The R package and optional Python rpy2 bridge must be installed locally.
    """

    dataprep: Any
    r_library: str | Path | None = None

    def fit(self, panel):
        from ._r import reference_library
        from .backends import _validate_dataprep
        from .results import make_estimate

        if panel.n_treated != 1 or panel.n_control < 2:
            raise ValueError("R Synth requires one treated unit and at least two control units")
        dp = self.dataprep
        order, _ = _validate_dataprep(panel, dp)
        donors = list(dp.controls_identifier)
        x0, x1 = dp.make_covariate_mats()
        if not x1.index.equals(x0.index) or not len(x0):
            raise ValueError("Dataprep must supply matching, nonempty predictor rows")
        # Validated Panel outcomes already contain the complete pre-fit Z data.
        # Assemble all four matrices in the original Dataprep's donor order.
        z0 = panel.control[: panel.n_pre, np.argsort(order)]
        matrices = {
            "X0": x0.loc[:, donors].to_numpy(dtype=float),
            "X1": x1.to_numpy(dtype=float).reshape(-1, 1),
            "Z0": z0,
            "Z1": panel.treated[: panel.n_pre, None],
        }
        if not all(np.isfinite(a).all() for a in matrices.values()):
            raise ValueError("All Dataprep predictor and outcome matrices must be finite")
        result = _run_synth(
            {key: values.tolist() for key, values in matrices.items()},
            reference_library(self.r_library),
        )
        weights = np.asarray(result.pop("weights"), dtype=float)
        metric = np.asarray(result["predictor_weights"], dtype=float)
        if (
            weights.shape != (panel.n_control,)
            or metric.shape != (len(x0),)
            or not np.isfinite([result["loss_v"], result["loss_w"]]).all()
            or not np.isfinite(weights).all()
            or not np.isfinite(metric).all()
            or weights.min() < -1e-6
            or abs(weights.sum() - 1) > 5e-4
            or metric.min() < 0
            or not np.isclose(metric.sum(), 1)
        ):
            raise RuntimeError(
                "R Synth returned invalid donor weights, predictor weights or losses"
            )
        return make_estimate(
            panel,
            "Reference SCM · R Synth",
            weights[order],
            metadata={**result, "predictor_names": x0.index.tolist()},
        )


if __name__ == "__main__":
    # Keep stdout reserved for the returned numeric fit.
    try:
        with redirect_stdout(sys.stderr):
            output = _fit_in_r(json.load(sys.stdin))
    except ImportError as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)
    print(json.dumps(output, allow_nan=False))
