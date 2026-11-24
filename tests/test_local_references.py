"""Local dependency discovery remains usable without R or private resources."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pysynth import SCUL, IndividualSC, Panel, SyntheticDID, TwoStageIndividualSC, _r
from pysynth.replication import _CALIFORNIA_DONORS, california_paper_inference


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "src/pysynth").mkdir(parents=True)
    (tmp_path / "empirical_examples").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYSYNTH_REFERENCE_DIR", raising=False)
    monkeypatch.delenv("PYSYNTH_R_LIBRARY", raising=False)
    return tmp_path


def test_checkout_resources_and_library_resolve_from_notebook_directory(checkout, monkeypatch):
    runtime = checkout / ".local/reference/runtime"
    runtime.mkdir(parents=True)
    (runtime / "isc.R").touch()
    library = checkout / ".r-library"
    library.mkdir()
    monkeypatch.chdir(checkout / "empirical_examples")
    assert _r.reference_directory("isc.R") == runtime
    assert _r.reference_library() == str(library)


def test_explicit_resource_directory_is_authoritative(checkout, monkeypatch):
    (checkout / ".local/reference/runtime").mkdir(parents=True)
    missing = checkout / "separately-provisioned"
    monkeypatch.setenv("PYSYNTH_REFERENCE_DIR", str(missing))
    with pytest.raises(ImportError, match="PYSYNTH_REFERENCE_DIR") as error:
        _r.reference_directory("isc.R")
    assert str(missing) in str(error.value)
    assert "isc.R" in str(error.value)


def test_incomplete_reference_directory_is_rejected(checkout, monkeypatch):
    runtime = checkout / ".local/reference/runtime"
    runtime.mkdir(parents=True)
    (runtime / "isc.R").touch()
    monkeypatch.setattr(_r.subprocess, "run", lambda *a, **k: pytest.fail("Must not execute R"))
    with pytest.raises(ImportError, match="common.R"):
        _r.run_reference("isc", {}, {})


@pytest.mark.parametrize(
    "model",
    [
        SCUL(initial_window=8, horizon=2),
        IndividualSC(),
        TwoStageIndividualSC(backend="mscmt", n_bootstrap=50),
    ],
)
def test_missing_local_resources_never_fall_back(checkout, monkeypatch, panel, model):
    monkeypatch.setattr(_r.subprocess, "run", lambda *a, **k: pytest.fail("Must not execute R"))
    with pytest.raises(ImportError, match="not distributed") as error:
        model.fit(panel)
    assert str(checkout / ".local/reference/runtime") in str(error.value)


def test_explicit_library_takes_precedence(checkout, monkeypatch):
    (checkout / "environment-library").mkdir()
    (checkout / "argument-library").mkdir()
    monkeypatch.setenv("PYSYNTH_R_LIBRARY", str(checkout / "environment-library"))
    assert _r.reference_library() == str(checkout / "environment-library")
    assert _r.reference_library(checkout / "argument-library") == str(checkout / "argument-library")
    with pytest.raises(ImportError, match="Selected R library"):
        _r.reference_library(checkout / "typo")


def test_reference_process_uses_local_file_and_preserves_failure(checkout, monkeypatch):
    runtime = checkout / ".local/reference/runtime"
    runtime.mkdir(parents=True)
    for name in ("isc.R", "common.R"):
        (runtime / name).touch()
    monkeypatch.setattr(_r.shutil, "which", lambda name: "/local/bin/Rscript")
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        work = Path(command[3])
        options = json.loads((work / "options.json").read_text())
        assert options["n_pre"] == 4
        (work / "result.json").write_text('{"weights": [1.0]}')
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(_r.subprocess, "run", execute)
    assert _r.run_reference("isc", {"outcomes": np.ones((6, 2))}, {"n_pre": 4}) == {
        "weights": [1.0]
    }
    assert commands[0][:3] == ["/local/bin/Rscript", "--vanilla", str(runtime / "isc.R")]
    assert commands[0][4] == str(runtime)
    monkeypatch.setattr(
        _r.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stderr="upstream failure"),
    )
    with pytest.raises(RuntimeError, match="upstream failure"):
        _r.run_reference("isc", {}, {})


def test_california_requires_local_rng_resource_before_execution(checkout, monkeypatch):
    values = np.random.default_rng(19).normal(size=(31, 39))
    panel = Panel(
        values, 38, 19, times=np.arange(1970, 2001), units=(*_CALIFORNIA_DONORS, "California")
    )
    estimate = SimpleNamespace(
        metadata={**SyntheticDID()._settings(panel), "sparsify": True, "omega_intercept": True}
    )
    monkeypatch.setattr(_r.subprocess, "run", lambda *a, **k: pytest.fail("Must not execute R"))
    with pytest.raises(ImportError, match="california_rng.R"):
        california_paper_inference(panel, estimate)
