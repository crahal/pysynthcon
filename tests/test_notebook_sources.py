"""Validate generated notebook code without executing expensive estimators."""

import ast
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("nbformat")


@pytest.fixture
def generator(monkeypatch, tmp_path):
    path = Path(__file__).parents[1] / "scripts/create_notebooks.py"
    spec = importlib.util.spec_from_file_location("notebook_generator", path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    monkeypatch.setattr(generator, "ROOT", tmp_path)
    return generator


@pytest.mark.parametrize("name", ["basque", "california", "germany", "kansas"])
def test_generated_code_compiles_and_diagnostics_are_valid_json(name, generator, tmp_path):
    generator.build(name, *generator.EXAMPLES[name])
    path = tmp_path / "empirical_examples" / f"{name}.ipynb"
    notebook = generator.nbf.read(path, as_version=4)
    writes = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = "\n".join(line for line in cell.source.splitlines() if not line.startswith("%"))
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "write_text":
                    statement = ast.fix_missing_locations(ast.Expression(node))
                    eval(
                        compile(statement, str(path), "eval"),
                        {
                            "output_dir": tmp_path,
                            "name": name,
                            "diagnostics": {"estimator": {"iterations": 10}},
                            "json": json,
                        },
                    )
                    writes += 1
    assert writes == 1
    assert json.loads((tmp_path / f"{name}-diagnostics.json").read_text()) == {
        "estimator": {"iterations": 10}
    }


def test_common_estimator_specs_use_paper_methods_and_pretreatment_predictors(generator, tmp_path):
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    import pysynth

    generator.build("kansas", *generator.EXAMPLES["kansas"])
    nb = generator.nbf.read(tmp_path / "empirical_examples/kansas.ipynb", as_version=4)
    cell = next(cell.source for cell in nb.cells if "models = {" in cell.source)
    times = pd.Index(2000 + np.arange(15) / 4)
    scope = {
        **vars(pysynth),
        "np": np,
        "panel": SimpleNamespace(times=times, n_pre=10),
        "name": "kansas",
    }
    exec(cell.split("results, elapsed =")[0], scope)
    models = list(scope["models"].values())
    types = [type(model) for model in models]
    for method in (
        pysynth.PredictorSyntheticControl,
        pysynth.AugmentedSyntheticControl,
        pysynth.PenalizedSyntheticControl,
        pysynth.RobustSyntheticControl,
        pysynth.DifferencedSyntheticControl,
    ):
        assert types.count(method) == 1
    assert not {method.__name__ for method in types} & {"PySyncon", "SCM"}
    predictor = models[types.index(pysynth.PredictorSyntheticControl)]
    assert [list(window) for window in predictor.predictor_windows] == [
        times[:4].tolist(),
        times[4:7].tolist(),
        times[7:10].tolist(),
    ]


@pytest.mark.parametrize("from_notebook_directory", [False, True])
def test_empirical_export_paths_and_result_tables(
    from_notebook_directory, generator, monkeypatch, tmp_path
):
    import pandas as pd

    name = "california"
    generator.build(name, *generator.EXAMPLES[name])
    examples = tmp_path / "empirical_examples"
    nb = generator.nbf.read(examples / f"{name}.ipynb", as_version=4)
    monkeypatch.chdir(examples if from_notebook_directory else tmp_path)

    # Execute the generated path setup and version export before model fitting.
    source = next(cell.source for cell in nb.cells if "def export_figure(" in cell.source)
    setup = source.split("def export_figure(")[0]
    namespace = {
        "Path": Path,
        "name": name,
        "LITERATURE_BENCHMARKS": {name: None},
        "versions": pd.Series({"numpy": "test"}, name="version"),
    }
    exec(setup, namespace)
    destination = tmp_path / "outputs" / "empirical_examples" / name
    assert namespace["output_dir"] == destination
    assert (destination / f"{name}-versions.csv").is_file()
    assert not (examples / "figures").exists()

    # Every plot and result-file write in the notebook uses this directory.
    types = {"to_csv", "savetxt", "write_text", "save_figure"}
    filenames = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse("\n".join(s for s in cell.source.splitlines() if not s.startswith("%")))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            method = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if method not in types:
                continue
            if method == "write_text":
                output = func.value
            else:
                output = node.args[1 if method == "save_figure" else 0]
            assert isinstance(output, ast.BinOp) and isinstance(output.op, ast.Div)
            assert isinstance(output.left, ast.Name) and output.left.id == "output_dir"
            filenames.append(ast.unparse(output.right))
    for suffix in ["versions", "replication-checks", "selection", "donor-weights", "warnings"]:
        assert any(f"-{suffix}.csv" in template for template in filenames)
