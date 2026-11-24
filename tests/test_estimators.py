import json
from pathlib import Path

import numpy as np
import pytest

from pysynth import (
    SCUL,
    DifferenceInDifferences,
    IndividualSC,
    Panel,
    SyntheticControl,
    SyntheticDID,
    TwoStageIndividualSC,
    compare,
    placebo_test,
)


def test_convex_known_counterfactual(panel):
    result = SyntheticControl().fit(panel)
    assert result.att == pytest.approx(3, abs=2e-4)
    assert result.pre_rmse < 1e-4
    assert result.weights.sum() == pytest.approx(1)
    assert result.weights.min() >= 0


def test_sdid_independent_r_reference():
    reference = json.loads((Path(__file__).parent / "fixtures/sdid_r.json").read_text())
    panel = Panel(reference["y"], reference["n_control"], reference["n_pre"])
    result = SyntheticDID().fit(panel)
    assert result.att == pytest.approx(reference["att"], abs=1e-10)
    np.testing.assert_allclose(result.weights, reference["omega"], atol=1e-10)
    np.testing.assert_allclose(result.time_weights, reference["lambda"], atol=1e-10)


def test_scul_independent_r_procedure_and_high_precision_glmnet():
    reference = json.loads((Path(__file__).parent / "fixtures/scul_r.json").read_text())
    panel = Panel(reference["y"], 6, 20)
    result = SCUL(backend="python", selection="min", initial_window=8, horizon=2, tol=1e-10).fit(
        panel
    )
    assert result.metadata["alpha"] == pytest.approx(reference["alpha"], abs=1e-9)
    # Default glmnet's coefficient-change stopping rule leaves a small path
    # error. Verify the estimator separately at a tighter glmnet tolerance.
    np.testing.assert_allclose(result.counterfactual, reference["counterfactual"], atol=0.01)
    np.testing.assert_allclose(result.counterfactual, reference["tight_counterfactual"], atol=1e-5)


def test_did_four_cell_identity(panel):
    t, c, y = panel.n_pre, panel.control, panel.treated
    expected = y[t:].mean() - y[:t].mean() - c[t:].mean() + c[:t].mean()
    assert DifferenceInDifferences().fit(panel).att == pytest.approx(expected)


@pytest.mark.parametrize("model", [DifferenceInDifferences(), SyntheticDID()])
def test_additive_multi_treated_exact(model):
    y = np.arange(8)[None, :] + (np.arange(20)[:, None] / 3) ** 2
    y[12:, 5:] -= 4
    panel = Panel(y, 5, 12)
    assert model.fit(panel).att == pytest.approx(-4, abs=1e-8)


@pytest.mark.parametrize(
    "model",
    [
        SyntheticControl(),
        SCUL(
            backend="python",
            selection="min",
        ),
        IndividualSC(backend="fixed_metric", n_bootstrap=0),
        TwoStageIndividualSC(backend="fixed_metric", n_neighbors=3),
        SyntheticDID(),
    ],
)
def test_no_treated_post_leakage(panel, model):
    y = panel.y.copy()
    y[panel.n_pre :, -1] += 1000
    changed = Panel(y, panel.n_control, panel.n_pre, units=panel.units)
    before, after = model.fit(panel), model.fit(changed)
    np.testing.assert_allclose(before.weights, after.weights, atol=1e-10)
    assert after.att - before.att == pytest.approx(1000)
    if before.time_weights is not None:
        np.testing.assert_allclose(before.time_weights, after.time_weights)


def test_sdid_scale_and_unit_level_invariance(panel):
    base = SyntheticDID().fit(panel)
    scaled = SyntheticDID().fit(Panel(panel.y * 10, 6, 20))
    shifted = SyntheticDID().fit(Panel(panel.y + np.arange(7) * 100, 6, 20))
    assert scaled.att == pytest.approx(base.att * 10, abs=1e-7)
    assert shifted.att == pytest.approx(base.att, abs=1e-7)


def test_scul_orthogonal_soft_threshold():
    # Orthogonal, centered, unit-variance columns: coefficients have closed form.
    x = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], float)
    y = 5 + 2 * x[:, 0] - 3 * x[:, 1]
    panel = Panel(np.column_stack([np.vstack([x, x]), np.tile(y, 2)]), 2, 4)
    fit = SCUL(backend="python", selection="min", alpha=0.25).fit(panel)
    np.testing.assert_allclose(fit.weights, [1.75, -2.75], atol=1e-10)
    assert fit.intercept == pytest.approx(5)


def test_scul_validation_origins_and_constant_column(panel):
    y = panel.y.copy()
    y[:, 0] = 1
    p = Panel(y, 6, 20)
    result = SCUL(backend="python", selection="min", initial_window=8, horizon=3).fit(p)
    assert result.metadata["origins"] == list(range(8, 18))
    assert result.weights.iloc[0] == 0
    assert len(result.metadata["fold_mse"]) == 10


def test_scul_all_penalty_rules(panel):
    for selection in ["min", "median", "1se"]:
        r = SCUL(backend="python", selection=selection, alphas=(1, 0.1, 0.01)).fit(panel)
        assert np.isfinite(r.att)
        assert 0.01 <= r.metadata["alpha"] <= 1


@pytest.mark.parametrize(
    "model",
    [
        IndividualSC(backend="fixed_metric", n_bootstrap=10, random_state=42),
        TwoStageIndividualSC(
            backend="fixed_metric", n_neighbors=4, n_bootstrap=10, random_state=42
        ),
    ],
)
def test_individual_bootstrap_reproducibility(panel, model):
    a, b = model.fit(panel), model.fit(panel)
    np.testing.assert_array_equal(a.weights, b.weights)
    assert len(a.metadata["bootstrap_atts"]) == 10
    assert a.weights.sum() == pytest.approx(1)


@pytest.mark.parametrize(
    "model",
    [
        IndividualSC(backend="fixed_metric", n_bootstrap=0),
        TwoStageIndividualSC(backend="fixed_metric", n_bootstrap=0, n_neighbors=6),
    ],
)
def test_individual_event_aggregation(panel, model):
    shorter = Panel(panel.y[2:-2], 6, 18, units=panel.units)
    fitted = model.fit_many({"one": panel, "two": shorter}, person_weights={"one": 1, "two": 3})
    assert fitted.att == pytest.approx(3, abs=2e-4)
    assert fitted.summary.loc[-20, "n"] == 1
    assert fitted.summary.loc[0, "n"] == 2


def test_placebo_rank_and_treated_exclusion(panel):
    result = placebo_test(SyntheticControl(), panel)
    assert len(result.placebos) == 6
    assert 1 / 7 <= result.p_value <= 1
    for unit, fit in result.placebos.items():
        assert "Z" not in fit.weights.index
        assert unit not in fit.weights.index
    assert result.table.loc["Z", "treated"]


def test_comparison(panel):
    table = compare({"scm": SyntheticControl(), "did": DifferenceInDifferences()}, panel)
    assert list(table.index) == ["scm", "did"]
    assert table.loc["scm", "att"] == pytest.approx(3, abs=2e-4)


def test_comparison_rejects_different_samples(panel):
    original = SyntheticControl().fit(panel)
    changed = SyntheticControl().fit(Panel(panel.y, 6, 19))
    with pytest.raises(ValueError, match="share observed"):
        compare({"original": original, "changed": changed})


@pytest.mark.parametrize(
    "model",
    [
        SCUL(backend="python", selection="min", alpha=-1),
        SCUL(backend="python", selection="min", horizon=100),
        SCUL(backend="python", selection="bad"),
        TwoStageIndividualSC(backend="fixed_metric", n_neighbors=0),
        IndividualSC(backend="fixed_metric", n_bootstrap=-1),
        SyntheticDID(eta_lambda=-1),
        SyntheticDID(noise=np.nan),
        SyntheticControl(ridge=-1),
    ],
)
def test_invalid_estimator_options(panel, model):
    with pytest.raises(ValueError):
        model.fit(panel)
