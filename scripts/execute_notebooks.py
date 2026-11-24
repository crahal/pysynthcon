"""Execute the empirical notebooks using the active interpreter as the kernel.

Usage: python scripts/execute_notebooks.py [--output-dir /tmp/executed]
Default execution updates tracked notebooks with their outputs. Failures stop
the run; no failed or missing estimator is silently skipped.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=int, default=14400, help="Per-cell timeout in seconds")
    parser.add_argument(
        "--examples",
        nargs="+",
        choices=["basque", "california", "germany", "kansas"],
        help="Execute selected examples; by default execute all four",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pysynth-mpl"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if args.output_dir is not None:
        os.environ["PYSYNTH_FIGURE_DIR"] = str(args.output_dir.resolve() / "figures")
    for path in sorted((root / "empirical_examples").glob("*.ipynb")):
        if args.examples is not None and path.stem not in args.examples:
            continue
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
