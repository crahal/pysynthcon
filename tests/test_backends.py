import numpy as np
import pandas as pd
import pytest

from pysynth import (
    SCM,
    AugmentedSyntheticControl,
    DifferencedSyntheticControl,
    Panel,
    PenalizedSyntheticControl,
    PredictorSyntheticControl,
    PySyncon,
    RobustSyntheticControl,
)


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


@pytest.mark.parametrize("method, penalty", [("synth", 0), ("pensynth", 0.01)])
def test_large_common_level_does_not_mask_feasible_qp_improvements(method, penalty):
    pytest.importorskip("pysyncon")
    # For w on the second donor, the half-loss is
    # (2w - .8)^2 / 2 + penalty * (.64 + .8w) / 2.
    values = np.tile([100000.0, 100002.0, 100000.8], (6, 1))
    values[4:, -1] += 1
    panel = Panel(values, 2, 4)
    options = {"custom_V": [1.0] * 4} if not penalty else {"lambda_": penalty}
    result = PySyncon(method, options=options).fit(panel)
    expected = 0.4 - 0.1 * penalty
    np.testing.assert_allclose(result.weights, [1 - expected, expected], atol=1e-7)
    assert result.att == pytest.approx(1 + 0.2 * penalty, abs=1e-6)


@pytest.mark.parametrize("covariates", [False, True])
def test_augmented_exact_uniform_fit_keeps_estimate_when_scaled_imbalance_undefined(covariates):
    pytest.importorskip("pysyncon")
    p = Panel([[0, 2, 1], [3, 1, 2], [3, 5, 4], [0, 2, 2], [1, 1, 2]], 2, 3)
    adapter = PySyncon(
        "augsynth", dataprep=_dataprep(p), options={"lambda_": 0.1, "use_covariates": covariates}
    )
    fit = adapter.fit(p)
    np.testing.assert_allclose(fit.weights, 0.5, atol=1e-10)
    assert fit.att == pytest.approx(1)
    assert np.isnan(adapter.backend_.scaled_l2_imbalance)
    if covariates:
        assert adapter.backend_.covariate_l2_imbalance == pytest.approx(0, abs=1e-10)


def test_zero_ridge_cannot_invert_rank_deficient_augmented_design():
    pytest.importorskip("pysyncon")
    rng = np.random.default_rng(0)
    panel = Panel(rng.normal(size=(20, 5)), 4, 16)
    # Floating-point inv() can return weights summing to -14 on this panel
    # despite a centered design of rank at most 3 and a 16-by-16 Gram matrix.
    with pytest.raises(ValueError, match="lambda_=0 requires a full-row-rank"):
        PySyncon("augsynth", options={"lambda_": 0.0}).fit(panel)
    corrected = PySyncon("augsynth", options={"lambda_": 0.1}).fit(panel)
    assert corrected.weights.sum() == pytest.approx(1)


def test_residualizing_duplicate_covariates_rejects_singular_ols(panel):
    pytest.importorskip("pysyncon")
    dp = _dataprep(panel)
    dp.foo["duplicate"] = dp.foo["outcome"]
    dp.predictors = ["outcome", "duplicate"]
    with pytest.raises(ValueError, match="linearly independent centered covariates"):
        PySyncon("augsynth", dataprep=dp, options={"lambda_": 0.1, "residualize": True}).fit(panel)


def test_predictor_windows_fit_labelled_means_and_validate_the_panel(panel):
    pytest.importorskip("pysyncon")
    dates = pd.date_range("1980", periods=len(panel.y), freq="YS")
    labelled = Panel(panel.y, panel.n_control, panel.n_pre, times=dates, units=panel.units)
    windows = [dates[:7], dates[7:20]]
    spec = PredictorSyntheticControl(predictor_windows=windows, custom_V=[0.6, 0.4])
    fit = spec.fit(labelled)
    predictors, target = spec.backend_.dataprep.make_covariate_mats()
    expected = np.array([panel.y[:7].mean(axis=0), panel.y[7:20].mean(axis=0)])
    np.testing.assert_allclose(predictors, expected[:, :-1])
    np.testing.assert_allclose(target, expected[:, -1])
    assert fit.method == "Predictor SCM"
    assert fit.metadata["predictor_windows"] == [list(w) for w in windows]
    assert fit.metadata["method_reference"].endswith("jasa.2009.ap08746")

    for invalid in ([], [[]], [dates[19:21]], [[dates[0], dates[0]]], "1980", [1980]):
        with pytest.raises(ValueError, match="window"):
            PredictorSyntheticControl(predictor_windows=invalid).fit(labelled)
    with pytest.raises(ValueError, match="either predictor_windows or dataprep"):
        PredictorSyntheticControl(dataprep=spec.backend_.dataprep, predictor_windows=windows).fit(
            labelled
        )


def test_named_penalized_and_augmented_fits_have_distinct_paper_objectives():
    pytest.importorskip("pysyncon")
    donors = np.tile([0.0, 2.0], (6, 1))
    panel = Panel(np.column_stack([donors, np.full(6, 0.8)]), 2, 4)
    penalized = PenalizedSyntheticControl(lambda_=1.0).fit(panel)
    # With donor-2 weight b the objective is proportional to
    # (2b-.8)^2 + .64(1-b) + 1.44b, minimized at b=.3.
    np.testing.assert_allclose(penalized.weights, [0.7, 0.3], atol=1e-8)
    assert penalized.method == "Penalized SCM"
    assert penalized.metadata["method_reference"].endswith("1971535")

    # The target is outside the donor hull. With four identical pre-periods,
    # SCM puts all mass on donor 2, leaving four residuals of +1. Ridge
    # augmentation with lambda=2 and donor-centered design [-1, 1] per row
    # adds [-.4, +.4] to the initial weights.
    outside = Panel(np.column_stack([donors, np.full(6, 3.0)]), 2, 4)
    augmented = AugmentedSyntheticControl(lambda_=2.0).fit(outside)
    np.testing.assert_allclose(augmented.weights, [-0.4, 1.4], atol=1e-8)
    np.testing.assert_allclose(augmented.counterfactual, 2.8, atol=1e-8)
    assert augmented.method == "Ridge-augmented SCM"
    assert augmented.metadata["ridge_penalty"] == 2
    assert augmented.metadata["method_reference"].endswith("1929245")


def test_named_robust_fit_needs_no_optional_backend_and_predicts_on_denoised_data(monkeypatch):
    def unexpected(*args):
        pytest.fail("RobustSyntheticControl should work from a complete Panel alone")

    monkeypatch.setattr("pysynth.backends._require", unexpected)
    donors = np.array([[0.0, 1.0], [2.0, 0.0], [1.0, 2.0], [2.0, 3.0], [4.0, 2.0]])
    outcome = np.array([1.0, 2.0, 1.2, 5.0, 7.0])
    panel = Panel(np.column_stack([donors, outcome]), 2, 3)
    result = RobustSyntheticControl(lambda_=0.2, sv_count=1).fit(panel)
    u, singular, vh = np.linalg.svd(donors, full_matrices=False)
    projected = singular[0] * np.outer(u[:, 0], vh[0])
    weights = np.linalg.solve(
        projected[:3].T @ projected[:3] + 0.2 * np.eye(2), projected[:3].T @ outcome[:3]
    )
    np.testing.assert_allclose(result.weights, weights)
    np.testing.assert_allclose(result.counterfactual, projected @ weights)
    assert result.method == "Robust SCM"
    assert result.metadata["observed_fraction"] == 1.0
    assert result.metadata["retained_rank"] == 1
    np.testing.assert_allclose(result.metadata["singular_values"], singular)


def test_named_differenced_fit_recovers_known_slope_and_level():
    pytest.importorskip("SyntheticControlMethods")
    t = np.arange(8, dtype=float)
    donors = np.column_stack([t, 2 + 3 * t])
    untreated = donors @ [0.4, 0.6] + 2
    observed = untreated.copy()
    observed[5:] += 4
    panel = Panel(np.column_stack([donors, observed]), 2, 5)
    estimator = DifferencedSyntheticControl()
    fit = estimator.fit(panel)
    np.testing.assert_allclose(fit.weights, [0.4, 0.6], atol=1e-7)
    np.testing.assert_allclose(fit.counterfactual, untreated, atol=1e-7)
    assert fit.intercept == pytest.approx(2)
    assert fit.att == pytest.approx(4)
    assert fit.metadata["outer_fit_periods"] == [1, 2, 3, 4]
    assert fit.method == "Differenced SCM"
    migrated = SCM("diffsynth").fit(panel)
    np.testing.assert_allclose(fit.counterfactual, migrated.counterfactual)
    assert migrated.method == "SyntheticControlMethods.diffsynth"


def test_augmented_custom_dataprep_with_no_covariates_uses_outcome_only_path(panel):
    pytest.importorskip("pysyncon")
    dp = _dataprep(panel)
    dp.predictors = []
    direct = AugmentedSyntheticControl(lambda_=0.3).fit(panel)
    with_prep = AugmentedSyntheticControl(lambda_=0.3, dataprep=dp).fit(panel)
    np.testing.assert_allclose(with_prep.weights, direct.weights)
    assert not with_prep.metadata["options"]["use_covariates"]
    with pytest.raises(ValueError, match="requires covariates"):
        AugmentedSyntheticControl(lambda_=0.3, use_covariates=True).fit(panel)
