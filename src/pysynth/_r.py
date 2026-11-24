"""Isolated, offline execution of the original R reference implementations."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

REFERENCE_DIR = Path(__file__).with_name("_reference")


def run_reference(name, matrices, options, *, r_library=None):
    """Run a packaged R script with numeric files and JSON arguments.

    No packages or code are downloaded during estimation. R is a separate
    process, so reference package loading and random seeds cannot alter the
    Python process or another fit. Failures retain the R error message.
    """
    executable = shutil.which("Rscript")
    if executable is None:
        raise ImportError(
            "The original-method backend requires Rscript. Install R and run "
            "`python scripts/install_r_references.py`; see docs/reference_backends.md. "
            "There is no automatic approximation fallback."
        )
    library = r_library or os.environ.get("PYSYNTH_R_LIBRARY")
    if library is None and Path(".r-library").is_dir():
        library = str(Path(".r-library").resolve())
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
                str(REFERENCE_DIR / f"{name}.R"),
                str(directory),
                str(REFERENCE_DIR),
            ],
            env=dict(os.environ, PYSYNTH_R_LIBRARY=str(library)) if library else None,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"The {name} reference backend failed:\n{result.stderr.strip()}\n"
                "Install its pinned dependencies with `python scripts/install_r_references.py` "
                "and select the library with PYSYNTH_R_LIBRARY or r_library."
            )
        output = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if result.stderr.strip():
        warnings.warn(
            f"{name} reference backend: {result.stderr.strip()}", RuntimeWarning, stacklevel=2
        )
    return output
