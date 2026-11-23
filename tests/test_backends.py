import numpy as np
import pytest

from pysynth import SCM, PySyncon


@pytest.mark.parametrize(
    "method,options",
    [
        ("synth", {}),
        ("augsynth", {}),
        ("pensynth", {}),
        ("robust", {"lambda_": 0.1, "sv_count": 3}),
    ],
)
def test_pysyncon_adapter_preserves_backend_prediction(panel, method, options):
    pytest.importorskip("pysyncon")
    adapter = PySyncon(method, options=options)
    result = adapter.fit(panel)
    np.testing.assert_allclose(result.counterfactual, panel.control @ adapter.backend_.W)
    assert list(result.weights.index) == list("ABCDEF")


@pytest.mark.parametrize("method", ["synth", "diffsynth"])
def test_scm_adapter_preserves_backend_prediction(panel, method):
    pytest.importorskip("SyntheticControlMethods")
    adapter = SCM(method)
    result = adapter.fit(panel)
    np.testing.assert_allclose(
        result.counterfactual, adapter.backend_.original_data.synth_outcome.ravel()
    )
    if method == "diffsynth":
        assert result.effects.iloc[: panel.n_pre].mean() == pytest.approx(0, abs=1e-10)


@pytest.mark.parametrize("method", ["synth", "diffsynth"])
def test_scm_corrected_covariate_objective_recovers_exact_mixture(panel, method):
    pytest.importorskip("SyntheticControlMethods")
    result = SCM(method, n_lags=10).fit(panel)
    assert result.att == pytest.approx(3, abs=1e-4)
    assert result.pre_rmse < 1e-4
