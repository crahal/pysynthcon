import numpy as np
import pytest

from pysynth import SCM, Panel, PySyncon


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


def test_full_scm_defaults_and_pysyncon_learns_v(panel):
    pytest.importorskip("SyntheticControlMethods")
    pytest.importorskip("pysyncon")
    fit = SCM().fit(panel)
    assert fit.metadata["lag_periods"] == list(range(panel.n_pre))
    assert len(fit.metadata["optimization"]) == 10
    assert all(d["attempts"][-1]["success"] for d in fit.metadata["optimization"])
    adapter = PySyncon()
    result = adapter.fit(panel)
    assert "custom_V" not in result.metadata["options"]
    assert adapter.backend_.V.sum() == pytest.approx(1)


def test_scm_pairwise_penalty_has_analytic_two_donor_solution():
    pytest.importorskip("SyntheticControlMethods")
    # For donor outcomes 0,2 and target .8, the objective with pen=1 is
    # (2b-.8)^2 + .64(1-b) + 1.44b, minimized at b=.3.
    outcome = np.tile([0.0, 2.0, 0.8], (8, 1))
    outcome[5:, -1] += 1
    fit = SCM(pen=1).fit(Panel(outcome, 2, 5))
    np.testing.assert_allclose(fit.weights, [0.7, 0.3], atol=1e-8)


def test_failed_line_search_is_reoptimized_not_accepted(panel, monkeypatch):
    pytest.importorskip("SyntheticControlMethods")
    import pysynth._nested as nested

    original = nested.minimize
    methods = []

    def fail_gradient_search(fun, initial, *, method, **kwargs):
        methods.append(method)
        if method == "L-BFGS-B":
            from scipy.optimize import OptimizeResult

            return OptimizeResult(
                x=initial,
                fun=fun(initial),
                success=False,
                nfev=1,
                message="ABNORMAL_TERMINATION_IN_LNSRCH",
            )
        return original(fun, initial, method=method, **kwargs)

    monkeypatch.setattr(nested, "minimize", fail_gradient_search)
    fit = SCM(n_optim=2).fit(panel)
    assert methods == ["L-BFGS-B", "Powell", "L-BFGS-B", "Powell"]
    assert fit.att == pytest.approx(3, abs=1e-4)
    for start in fit.metadata["optimization"]:
        assert not start["attempts"][0]["success"]
        assert start["attempts"][1]["success"]


def test_unrecovered_outer_failure_raises(panel, monkeypatch):
    pytest.importorskip("SyntheticControlMethods")
    from scipy.optimize import OptimizeResult

    import pysynth._nested as nested

    def fail(fun, initial, **kwargs):
        return OptimizeResult(
            x=initial, fun=fun(initial), success=False, nfev=1, message="iteration limit"
        )

    monkeypatch.setattr(nested, "minimize", fail)
    with pytest.raises(RuntimeError, match="optimization start 1 failed"):
        SCM().fit(panel)
