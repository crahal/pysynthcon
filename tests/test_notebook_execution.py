"""The runner releases each real kernel, including after a failed cell."""

import importlib.util
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")


@pytest.fixture
def runner(monkeypatch, tmp_path):
    script = Path(__file__).parents[1] / "scripts/execute_notebooks.py"
    spec = importlib.util.spec_from_file_location("notebook_runner", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    # Run the actual entry point against a tiny temporary notebook, leaving
    # all empirical notebooks untouched and avoiding estimator calculations.
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts/execute_notebooks.py"))
    monkeypatch.setattr(runner.sys, "argv", [str(script)])
    monkeypatch.setenv("JUPYTER_RUNTIME_DIR", str(tmp_path / "runtime"))
    directory = tmp_path / "empirical_examples"
    directory.mkdir()
    return runner


def test_default_runner_ignores_scratch_files_and_reports_missing_example(runner, tmp_path):
    directory = tmp_path / "empirical_examples"
    (directory / "aaa.executed.ipynb").write_text("scratch file")
    with pytest.raises(FileNotFoundError) as exc:
        runner.main()
    assert Path(exc.value.filename) == directory / "basque.ipynb"


@pytest.mark.parametrize("fail", [False, True])
def test_runner_shuts_down_kernel_and_channels(runner, fail, monkeypatch, tmp_path):
    monkeypatch.setattr(runner.sys, "argv", [runner.__file__, "--examples", "basque"])
    path = tmp_path / "empirical_examples" / "basque.ipynb"
    code = "raise ValueError('intentional execution failure')" if fail else "print(42)"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)]), path)
    clients, managers = [], []
    original_client = runner.NotebookClient

    def record_client(*args, **kwargs):
        client = original_client(*args, **kwargs)
        clients.append(client)
        managers.append(kwargs["km"])
        return client

    monkeypatch.setattr(runner, "NotebookClient", record_client)
    try:
        if fail:
            with pytest.raises(nbclient.exceptions.CellExecutionError, match="intentional"):
                runner.main()
        else:
            runner.main()
            executed = nbformat.read(path, as_version=4)
            assert executed.cells[0].outputs[0].text == "42\n"
        assert len(managers) == 1
        manager = managers[0]
        assert not manager.has_kernel
        assert not manager.is_alive()
        assert not Path(manager.connection_file).exists()
        assert clients[0].kc is None
    finally:
        # A failing regression must not itself leave an orphaned test kernel.
        for client in clients:
            if client.kc is not None:
                client.kc.stop_channels()
        for manager in managers:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
            manager.cleanup_resources()
