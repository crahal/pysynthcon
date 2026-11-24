"""Original implementations versus independent direct-R reference fixtures."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pysynth import SCUL, IndividualSC, Panel, TwoStageIndividualSC
from pysynth._r import reference_directory, reference_library, require_rscript, run_reference

FIXTURES = Path(__file__).with_name("fixtures")
pytestmark = pytest.mark.r_reference


@pytest.fixture(scope="module", autouse=True)
def original_r_packages():
    try:
        directory = reference_directory(
            "scul.R",
            "common.R",
            "isc.R",
            "scul/SCUL.R",
            "scul/getOptcv.scul.R",
            "scul/mysd.R",
            "../tests/check_packages.R",
        )
        executable = require_rscript()
    except ImportError as error:
        pytest.skip(str(error))
    library = reference_library()
    check = subprocess.run(
        [
            executable,
            "--vanilla",
            str(directory.parent / "tests/check_packages.R"),
        ],
        env=dict(os.environ, R_LIBS=library) if library else None,
        capture_output=True,
        check=False,
    )
    if check.returncode:
        pytest.skip("The local reference resources require pinned glmnet 4.0-2 and MSCMT 1.4.0")


@pytest.mark.parametrize("selection", ["median", "min", "1se"])
def test_original_scul_all_rules_against_direct_r(selection):
    old = json.loads((FIXTURES / "scul_r.json").read_text())
    references = json.loads((FIXTURES / "reference_backends_r.json").read_text())
    expected = next(item for item in references["scul"] if item["selection"] == selection)
    result = SCUL(initial_window=8, horizon=2, selection=selection).fit(Panel(old["y"], 6, 20))
    assert result.metadata["alpha"] == pytest.approx(expected["alpha"], abs=1e-14)
    np.testing.assert_allclose(result.counterfactual, expected["counterfactual"], atol=1e-10)
    np.testing.assert_allclose(result.weights, expected["coefficients"][1:], atol=1e-10)
    assert result.intercept == pytest.approx(expected["coefficients"][0], abs=1e-10)
    assert result.metadata["backend"] == "original_r"


def test_scul_single_period_validation_preserves_matrix_shape():
    old = json.loads((FIXTURES / "scul_r.json").read_text())
    result = SCUL(initial_window=8, horizon=1).fit(Panel(old["y"], 6, 20))
    assert np.isfinite(result.att)
    assert result.metadata["origins"] == list(range(8, 20))


@pytest.mark.parametrize("selection", ["median", "min"])
def test_scul_single_origin_uses_the_original_defined_selection_rule(selection):
    original = json.loads((FIXTURES / "scul_r.json").read_text())
    panel = Panel(original["y"], 6, 20)
    # Upstream also computes an unused 1se candidate, warning that this
    # candidate has no defined SD with one fold. Preserve that warning.
    with pytest.warns(RuntimeWarning, match="no non-missing arguments"):
        expected = run_reference(
            "scul",
            {"outcomes": panel.y},
            {
                "n_pre": 20,
                "initial_window": 18,
                "horizon": 2,
                "selection": f"lambda.{selection}",
                "alpha": None,
            },
        )
    with pytest.warns(RuntimeWarning, match="no non-missing arguments"):
        result = SCUL(initial_window=18, horizon=2, selection=selection).fit(panel)
    assert result.metadata["origins"] == [18]
    assert result.metadata["alpha"] == expected["alpha"]
    np.testing.assert_allclose(result.counterfactual, expected["counterfactual"], atol=1e-12)


def test_scul_original_defaults_and_no_post_treatment_leakage():
    old = json.loads((FIXTURES / "scul_r.json").read_text())
    panel = Panel(old["y"], 6, 22)
    fit = SCUL().fit(panel)
    assert fit.metadata["selection"] == "median"
    assert fit.metadata["initial_window"] == fit.metadata["horizon"] == 8
    changed = panel.y.copy()
    changed[22:, -1] += 500
    other = SCUL().fit(Panel(changed, 6, 22))
    np.testing.assert_array_equal(fit.weights, other.weights)
    assert other.att - fit.att == pytest.approx(500)


def test_scul_near_constant_donor_matches_original_screened_design():
    rng = np.random.default_rng(731)
    donors = rng.normal(size=(40, 3))
    rare = np.zeros(40)
    rare[0] = 1
    rare[25:] = np.arange(15)
    target = donors @ [0.2, 0.5, 0.3] + rare
    target[30:] += 2
    model = SCUL(initial_window=10, horizon=5)
    # Original caret flags the rare column at the 21-observation training
    # window (frequency 20:1, 2/21 unique), despite passing the first window.
    full = model.fit(Panel(np.column_stack([donors, rare, target]), 4, 30))
    screened = model.fit(Panel(np.column_stack([donors, target]), 3, 30))
    assert full.weights.iloc[3] == 0
    np.testing.assert_allclose(full.weights.iloc[:3], screened.weights, atol=1e-12)
    np.testing.assert_allclose(full.counterfactual, screened.counterfactual, atol=1e-12)
    assert full.metadata["removed_near_zero_variance_donors"] == [3]
    assert full.metadata["removed_constant_donors"] == []


def test_isc_covariates_and_fifty_resamples_match_direct_mscmt():
    reference = json.loads((FIXTURES / "reference_backends_r.json").read_text())["isc"]
    panel = Panel(reference["y"], 6, 20)
    predictors = {name: np.asarray(reference[name]) for name in ["age", "education"]}
    result = IndividualSC().fit(panel, predictors=predictors)
    np.testing.assert_allclose(result.weights, reference["weights"], atol=1e-10)
    np.testing.assert_allclose(result.counterfactual, reference["counterfactual"], atol=1e-10)
    assert result.metadata["bootstrap_donor_indices"] == reference["samples"]
    expected_atts = (
        panel.treated[20:, None] - panel.control[20:] @ np.asarray(reference["draw_weights"]).T
    ).mean(axis=0)
    np.testing.assert_allclose(result.metadata["bootstrap_atts"], expected_atts, atol=1e-10)
    assert result.metadata["predictor_groups"] == ["outcome", "age", "education"]
    assert result.metadata["n_bootstrap"] == 50
    assert result.metadata["backend"] == "original_mscmt"


def test_2sisc_full_pool_equals_original_isc_and_ignores_post_outcomes():
    reference = json.loads((FIXTURES / "reference_backends_r.json").read_text())["isc"]
    panel = Panel(reference["y"], 6, 20)
    full = IndividualSC().fit(panel)
    staged = TwoStageIndividualSC(n_neighbors=6).fit(panel)
    np.testing.assert_array_equal(full.weights, staged.weights)
    assert full.metadata["bootstrap_donor_indices"] == staged.metadata["bootstrap_donor_indices"]
    changed = panel.y.copy()
    changed[20:, :] *= -20
    second = TwoStageIndividualSC(n_neighbors=6).fit(Panel(changed, 6, 20))
    np.testing.assert_array_equal(staged.weights, second.weights)


def test_isc_fit_many_forwards_case_specific_covariates():
    reference = json.loads((FIXTURES / "reference_backends_r.json").read_text())["isc"]
    panel = Panel(reference["y"], 6, 20)
    predictors = {name: np.asarray(reference[name]) for name in ["age", "education"]}
    individual = IndividualSC(n_bootstrap=0).fit(panel, predictors=predictors)
    combined = IndividualSC(n_bootstrap=0).fit_many(
        {"person": panel}, predictors={"person": predictors}
    )
    np.testing.assert_array_equal(combined.estimates["person"].weights, individual.weights)
    assert combined.att == individual.att
    with pytest.raises(ValueError, match="predictors must cover"):
        IndividualSC().fit_many({"person": panel}, predictors={})


def test_isc_rejects_failed_wnnls_iterate_and_recovers_exact_convex_fit():
    # The original WNNLS routine reports failure and returns a poor fit on
    # this exact convex combination on affected platforms. The mathematically
    # correct solution has zero pretreatment error, irrespective of whether
    # that platform needs the numerical repair.
    reference = json.loads((FIXTURES / "scul_r.json").read_text())
    result = IndividualSC(n_bootstrap=0).fit(Panel(reference["y"], 6, 20))
    assert result.pre_rmse < 2e-6
    assert result.att == pytest.approx(3, abs=1e-5)
    for repair in result.metadata["inner_solver_repairs"]:
        assert repair["dual_gap"] <= 1e-8
        assert repair["original_status"] == "error in wnnls"


@pytest.mark.parametrize("model", [IndividualSC(), TwoStageIndividualSC(n_neighbors=6)])
def test_isc_fit_many_matches_original_parallel_sampling_stream(model):
    reference = json.loads((FIXTURES / "reference_backends_r.json").read_text())
    original = reference["isc"]
    panels, predictors = {}, {}
    for key, expected in enumerate(reference["isc_many"]):
        columns = expected["columns"]
        panels[key] = Panel(np.asarray(original["y"])[:, columns], len(columns) - 1, 20)
        predictors[key] = {
            name: np.asarray(original[name])[:, columns] for name in ["age", "education"]
        }
    result = model.fit_many(panels, predictors=predictors, person_weights={0: 2, 1: 3})
    expected_atts = []
    for key, expected in enumerate(reference["isc_many"]):
        fit = result.estimates[key]
        assert fit.metadata["bootstrap_donor_indices"] == expected["samples"]
        np.testing.assert_allclose(fit.weights, expected["weights"], atol=1e-10)
        np.testing.assert_allclose(fit.counterfactual, expected["counterfactual"], atol=1e-10)
        expected_atts.append(
            (panels[key].treated[20:] - np.asarray(expected["counterfactual"])[20:]).mean()
        )
    assert result.att == pytest.approx((2 * expected_atts[0] + 3 * expected_atts[1]) / 5, abs=1e-10)
    # A standalone second case correctly restarts at 42 and must not reuse the
    # stream advanced by the first case. Repeated fit_many calls start afresh.
    alone = model.fit(panels[1], predictors=predictors[1])
    assert alone.metadata["bootstrap_donor_indices"] != reference["isc_many"][1]["samples"]
    repeated = model.fit_many(panels, predictors=predictors)
    for key in panels:
        np.testing.assert_array_equal(
            result.estimates[key].weights, repeated.estimates[key].weights
        )
