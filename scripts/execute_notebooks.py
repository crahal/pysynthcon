"""Execute the empirical and simulated notebooks using the active interpreter.

Usage: python scripts/execute_notebooks.py [--output-dir /tmp/executed]
Default execution updates the notebooks in place. Figures, tables and
diagnostics are written under the checkout's outputs/ directory.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

EXAMPLES = {
    "basque": "empirical_examples",
    "california": "empirical_examples",
    "germany": "empirical_examples",
    "kansas": "empirical_examples",
    "rhode_island": "empirical_examples",
    "donor_spillovers": "simulated_examples",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, help="Save executed notebook copies in this directory"
    )
    parser.add_argument("--timeout", type=int, default=14400, help="Per-cell timeout in seconds")
    parser.add_argument(
        "--examples",
        nargs="+",
        choices=EXAMPLES,
        help="Execute selected examples, or run all empirical and simulated examples by default",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pysynth-mpl"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    for example, directory in EXAMPLES.items():
        if args.examples is not None and example not in args.examples:
            continue
        path = root / directory / f"{example}.ipynb"
        print(f"Executing {path.name}", flush=True)
        nb = nbformat.read(path, as_version=4)
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        client = NotebookClient(
            nb, timeout=args.timeout, km=manager, resources={"metadata": {"path": str(root)}}
        )
        # An explicitly supplied manager is considered externally owned by
        # nbclient. Request its finally cleanup on both success and failure.
        client.execute(cleanup_kc=True)
        destination = path if args.output_dir is None else args.output_dir / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, destination)
        print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
