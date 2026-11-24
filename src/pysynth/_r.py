"""Isolated execution of separately provisioned, local R reference resources."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np


def _checkout_root():
    """Find this checkout from its root or a notebook subdirectory."""
    working = Path.cwd().resolve()
    for directory in (working, *working.parents):
        if (directory / "pyproject.toml").is_file() and (directory / "src/pysynth").is_dir():
            return directory
    return working


def reference_directory(*required):
    """Locate local runtime resources, rejecting missing or incomplete sets.

    An explicit environment setting is authoritative. No installed package,
    remote URL, Git history, or alternative implementation is searched.
    """
    configured = os.environ.get("PYSYNTH_REFERENCE_DIR")
    directory = (
        Path(configured).expanduser().resolve()
        if configured
        else _checkout_root() / ".local/reference/runtime"
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if not directory.is_dir() or missing:
        detail = f" Missing files: {', '.join(missing)}." if missing else ""
        raise ImportError(
            f"Local R reference resources are unavailable in {directory}.{detail} "
            "They are not distributed with this repository or its Python packages. "
            "Provision the local reference resources separately and set "
            "PYSYNTH_REFERENCE_DIR to their runtime directory; see "
            "docs/reference_backends.md. No resources are downloaded and no "
            "approximation is substituted."
        )
    return directory


def reference_library(r_library=None):
    """Use the explicit library or the checkout's existing local installation."""
    library = r_library or os.environ.get("PYSYNTH_R_LIBRARY")
    if library:
        selected = Path(library).expanduser().resolve()
        if not selected.is_dir():
            # R silently drops nonexistent directories from .libPaths().
            raise ImportError(f"Selected R library directory does not exist: {selected}")
        return str(selected)
    local = _checkout_root() / ".r-library"
    return str(local) if local.is_dir() else None


def require_rscript():
    executable = shutil.which("Rscript")
    if executable is None:
        raise ImportError(
            "The local reference backend requires Rscript. Install base R and "
            "provision the required local reference resources and pinned R packages; "
            "see docs/reference_backends.md. There is no automatic approximation fallback."
        )
    return executable


def run_reference(name, matrices, options, *, r_library=None):
    """Run a separately provisioned R script with numeric files and JSON arguments.

    No packages or code are downloaded during estimation. R is a separate
    process, so reference package loading and random seeds cannot alter the
    Python process or another fit. Failures retain the R error message.
    """
    required = {
        "scul": ("scul.R", "common.R", "scul/SCUL.R", "scul/getOptcv.scul.R", "scul/mysd.R"),
        "isc": ("isc.R", "common.R"),
    }
    if name not in required:
        raise ValueError(f"Unknown local reference backend: {name}")
    directory_reference = reference_directory(*required[name])
    executable = require_rscript()
    library = reference_library(r_library)
    with tempfile.TemporaryDirectory(prefix=f"pysynth-{name}-") as temporary:
        directory = Path(temporary)
        for key, value in matrices.items():
            np.savetxt(directory / f"{key}.csv", np.asarray(value), delimiter=",")
        arguments = dict(
            options,
            r_library=str(library) if library else None,
            python_executable=sys.executable,
        )
        (directory / "options.json").write_text(json.dumps(arguments), encoding="utf-8")
        result = subprocess.run(
            [
                executable,
                "--vanilla",
                str(directory_reference / f"{name}.R"),
                str(directory),
                str(directory_reference),
            ],
            env=dict(os.environ, PYSYNTH_R_LIBRARY=str(library)) if library else None,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"The {name} reference backend failed:\n{result.stderr.strip()}\n"
                "Check the separately provisioned resources and pinned dependencies; "
                "select the R library with PYSYNTH_R_LIBRARY or r_library. "
                "See docs/reference_backends.md."
            )
        output = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if result.stderr.strip():
        warnings.warn(
            f"{name} reference backend: {result.stderr.strip()}", RuntimeWarning, stacklevel=2
        )
    return output
