"""Validate generated notebook code without executing expensive estimators."""

import ast
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("nbformat")


@pytest.mark.parametrize("name", ["basque", "california", "germany", "kansas"])
def test_generated_code_compiles_and_diagnostics_are_valid_json(name, monkeypatch, tmp_path):
    path = Path(__file__).parents[1] / "scripts/create_notebooks.py"
    spec = importlib.util.spec_from_file_location("notebook_generator", path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    notebooks = []
    monkeypatch.setattr(generator.nbf, "write", lambda nb, _: notebooks.append(nb))
    generator.build(name, *generator.EXAMPLES[name])
    writes = 0
    for cell in notebooks[0].cells:
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
                            "figure_dir": tmp_path,
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
