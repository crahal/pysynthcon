import numpy as np
import pandas as pd
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


def _dataprep(panel, *, predictor_periods=None, special=None, reverse=False):
    upstream = pytest.importorskip("pysyncon")
    controls = list(panel.units[:-1])
    return upstream.Dataprep(
        foo=panel.to_frame(),
        predictors=["outcome"],
        predictors_op="mean",
        dependent="outcome",
        unit_variable="unit",
        time_variable="time",
        treatment_identifier=panel.units[-1],
        controls_identifier=controls[::-1] if reverse else controls,
        time_predictors_prior=list(panel.times[: panel.n_pre])
        if predictor_periods is None
        else predictor_periods,
        time_optimize_ssr=list(panel.times[: panel.n_pre]),
        special_predictors=special,
    )


@pytest.mark.parametrize("special", [False, True])
def test_custom_dataprep_rejects_post_predictor_leakage(panel, special):
    period = panel.times[panel.n_pre]
    dp = _dataprep(
        panel,
        predictor_periods=None if special else [period],
        special=[("outcome", [period], "mean")] if special else None,
    )
    with pytest.raises(ValueError, match="post-treatment"):
        PySyncon(dataprep=dp).fit(panel)


def test_historical_post_predictors_require_explicit_recorded_opt_in(panel):
    period = panel.times[panel.n_pre]
    dp = _dataprep(panel, predictor_periods=[period])
    fit = PySyncon(dataprep=dp, options={"custom_V": [1.0]}, allow_post_predictors=True).fit(panel)
    assert fit.metadata["post_predictor_periods"] == [period]


def test_outer_iteration_failure_is_not_silently_accepted(panel):
    pytest.importorskip("pysyncon")
    with pytest.raises(RuntimeError, match="outer optimization failed"):
        PySyncon(options={"optim_options": {"maxiter": 1}}).fit(panel)


def test_default_outer_failure_recovers_same_objective(panel, monkeypatch):
    upstream = pytest.importorskip("pysyncon.synth")
    from scipy.optimize import OptimizeResult

    original = upstream.minimize

    def failed_first(fun, x0, method, options):
        if method != "Powell":
            return OptimizeResult(x=np.array(x0), fun=fun(x0), success=False, message="limit")
        return original(fun=fun, x0=x0, method=method, options=options)

    monkeypatch.setattr(upstream, "minimize", failed_first)
    fit = PySyncon(dataprep=_dataprep(panel)).fit(panel)
    attempts = fit.metadata["optimization"]["outer_attempts"]
    assert len(attempts) == 2 and not attempts[0]["success"]
    assert attempts[1]["solver"] == "Powell" and attempts[1]["success"]
    assert fit.weights.sum() == pytest.approx(1)


@pytest.mark.parametrize("penalty,expected", [(0.0, [0.6, 0.4]), (1.0, [0.7, 0.3])])
def test_failed_inner_qp_recovers_analytic_simplex_solution(penalty, expected, monkeypatch):
    upstream = pytest.importorskip("pysyncon.penalized")
    from scipy.optimize import OptimizeResult

    from pysynth.backends import _checked_upstream_optimizers

    def failed(**kwargs):
        return OptimizeResult(
            x=kwargs["x0"], fun=np.float64(0), success=False, message="line search failed"
        )

    monkeypatch.setattr(upstream, "minimize", failed)
    model = upstream.PenalizedSynth()
    _checked_upstream_optimizers(model)
    weights, _ = model.w_optimize(
        np.eye(1),
        np.array([[0.0, 2.0]]),
        np.array([0.8]),
        lambda_=penalty,
    )
    np.testing.assert_allclose(weights, expected, atol=1e-7)
    assert model.optimization_diagnostics_["inner_recoveries"][0]["success"]
    # Installing recovery on one instance must not change the upstream class.
    _, untouched_loss = upstream.PenalizedSynth().w_optimize(
        np.eye(1),
        np.array([[0.0, 2.0]]),
        np.array([0.8]),
        lambda_=penalty,
    )
    assert untouched_loss == 0


@pytest.mark.parametrize("options", [{"threshold": 0.0}, {"sv_count": 2}])
def test_robust_matches_paper_svd_ridge_equations_with_observed_zeros(options):
    pytest.importorskip("pysyncon")
    donors = np.array([[0, 1, 3], [1, 0, 2], [3, 1, 0], [2, 0, 4], [5, 2, 0]], float)
    treated = np.array([0.8, 1.4, 1.1, 3.0, 4.0])
    times = pd.to_datetime(["2000-01-01", "2000-03-01", "2000-07-01", "2001-02-01", "2002-01-01"])
    panel = Panel(np.column_stack([donors, treated]), 3, 3, times=times, units=tuple("ABCZ"))
    adapter = PySyncon(
        "robust", dataprep=_dataprep(panel, reverse=True), options={"lambda_": 0.3, **options}
    )
    fit = adapter.fit(panel)
    u, singular, vh = np.linalg.svd(donors, full_matrices=False)
    rank = options.get("sv_count", 3)
    denoised = (u[:, :rank] * singular[:rank]) @ vh[:rank]
    weights = np.linalg.solve(
        denoised[:3].T @ denoised[:3] + 0.3 * np.eye(3), denoised[:3].T @ treated[:3]
    )
    np.testing.assert_allclose(fit.weights, weights, atol=1e-12)
    np.testing.assert_allclose(fit.counterfactual, denoised @ weights, atol=1e-12)
    assert fit.metadata["observed_fraction"] == 1.0
    assert fit.metadata["retained_rank"] == rank


def test_robust_zero_ridge_handles_rank_deficiency(panel):
    pytest.importorskip("pysyncon")
    fit = PySyncon("robust", options={"lambda_": 0, "sv_count": 1}).fit(panel)
    assert np.isfinite(fit.counterfactual).all()


@pytest.mark.parametrize("method", ["synth", "diffsynth"])
def test_scm_retains_small_unit_lags_and_is_scale_invariant(panel, method):
    pytest.importorskip("SyntheticControlMethods")
    base = SCM(method, n_optim=1).fit(panel)
    small = SCM(method, n_optim=1)
    fit = small.fit(Panel(panel.y * 1e-14, panel.n_control, panel.n_pre))
    np.testing.assert_allclose(fit.weights, base.weights, atol=1e-7)
    assert len(small.backend_.original_data.covariates) == panel.n_pre + 1
    assert fit.att / 1e-14 == pytest.approx(base.att, abs=1e-7)


@pytest.mark.parametrize("scale,shift,change", [(1e-14, 0, 1e-9), (1, 1e9, 1)])
def test_dataprep_must_contain_exactly_the_reported_outcomes(panel, scale, shift, change):
    transformed = Panel(panel.y * scale + shift, panel.n_control, panel.n_pre, units=panel.units)
    dp = _dataprep(transformed)
    dp.foo.loc[0, "outcome"] += change
    with pytest.raises(ValueError, match="outcomes must match"):
        PySyncon(dataprep=dp).fit(transformed)


@pytest.mark.parametrize("metric", [[-1, 2], [0, 0], [np.nan, 1]])
def test_scm_predictor_metric_must_define_nonnegative_nonzero_distance(panel, metric):
    pytest.importorskip("pysyncon")
    with pytest.raises(ValueError, match="custom_V"):
        PySyncon(options={"custom_V": metric}).fit(panel)
