"""Draw-for-draw checks against the original R synthdid variance estimators."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from pysynth import DifferenceInDifferences, Panel, SyntheticDID


@pytest.fixture(scope="module")
def reference():
    return json.loads((Path(__file__).parent / "fixtures/sdid_inference_r.json").read_text())


@pytest.fixture(scope="module")
def multi_panel(reference):
    return Panel(reference["y"], reference["n_control"], reference["n_pre"])


@pytest.mark.parametrize("method", ["bootstrap", "jackknife", "placebo"])
def test_variance_and_each_draw_match_independent_r(reference, multi_panel, method):
    options = {}
    if method != "jackknife":
        options = {"resamples": reference[f"{method}_resamples"], "replications": 12}
    with warnings.catch_warnings(record=True) as emitted:
        result = SyntheticDID().inference(multi_panel, method=method, **options)
    np.testing.assert_allclose(result.draws, reference[method]["draws"], rtol=0, atol=1e-10)
    assert result.variance == pytest.approx(reference[method]["variance"], abs=1e-10)
    assert result.estimate == pytest.approx(reference["att"], abs=1e-10)
    assert result.standard_error**2 == pytest.approx(result.variance)
    capped = any(
        not (d["omega"]["stopping_rule_met"] and d["lambda"]["stopping_rule_met"])
        for d in result.metadata["solver_diagnostics"]
    )
    assert any("draws reached max_iter" in str(w.message) for w in emitted) == capped
    low, high = result.confidence_interval()
    assert (low + high) / 2 == pytest.approx(result.estimate)
    assert (high - low) / 2 == pytest.approx(1.959963984540054 * result.standard_error)


def test_reference_sc_and_did_are_the_r_estimators(reference, multi_panel):
    result = SyntheticDID().fit_sc(multi_panel)
    assert result.att == pytest.approx(reference["sc"], abs=1e-10)
    np.testing.assert_array_equal(result.time_weights, 0)
    assert result.intercept == 0
    assert DifferenceInDifferences().fit(multi_panel).att == pytest.approx(
        reference["did"], abs=1e-10
    )


def test_jackknife_retains_author_software_centering_convention(reference, multi_panel):
    """Paper Algorithm 3's full-ATT center is not the R var(draws) center."""
    result = SyntheticDID().inference(multi_panel, method="jackknife")
    n = len(result.draws)
    paper_variance = (n - 1) * np.mean((result.draws - result.estimate) ** 2)
    centering_difference = (n - 1) * (result.draws.mean() - result.estimate) ** 2
    assert result.metadata["variance_centering"] == "leave_one_out_mean"
    assert result.variance == pytest.approx(reference["jackknife"]["variance"], abs=1e-10)
    assert paper_variance - result.variance == pytest.approx(centering_difference, abs=1e-14)
    # Nonuniform fitted weights make the finite-sample distinction observable.
    assert centering_difference > 1e-3


def test_seeded_placebo_draws_can_be_replayed(multi_panel):
    model = SyntheticDID()
    estimate = model.fit(multi_panel)
    first = model.inference(
        multi_panel, method="placebo", replications=3, random_state=3, estimate=estimate
    )
    replay = model.inference(
        multi_panel,
        method="placebo",
        replications=3,
        resamples=first.resamples,
        estimate=estimate,
    )
    np.testing.assert_array_equal(first.draws, replay.draws)
    np.testing.assert_array_equal(
        np.sort(first.resamples, axis=1), np.tile(np.arange(multi_panel.n_control), (3, 1))
    )
    assert first.metadata["settings"]["zeta_omega"] == estimate.metadata["zeta_omega"]


@pytest.mark.parametrize("method", ["bootstrap", "jackknife"])
def test_single_treated_requires_placebo(panel, method):
    with pytest.raises(ValueError, match="at least two treated units"):
        SyntheticDID().inference(panel, method=method)


def test_inference_rejects_fit_from_changed_donor_panel(multi_panel):
    model = SyntheticDID()
    estimate = model.fit(multi_panel)
    changed = multi_panel.y.copy()
    changed[:, 0] += 0.25
    other = Panel(changed, multi_panel.n_control, multi_panel.n_pre)
    with pytest.raises(ValueError, match="fit of this panel"):
        model.inference(other, method="placebo", estimate=estimate)


@pytest.mark.parametrize("count", [True, 1, 1.5])
def test_invalid_replication_count(multi_panel, count):
    with pytest.raises(ValueError, match="replications must be an integer >= 2"):
        SyntheticDID().inference(multi_panel, replications=count)


@pytest.mark.parametrize(
    ("method", "indices", "message"),
    [
        ("bootstrap", [[0] * 9] * 2, "both treatment groups"),
        ("bootstrap", [[0, 1, 2, 3, 4, 5, 6, 7, 9]] * 2, "both treatment groups"),
        ("placebo", [[0, 1, 2, 3, 4, 4]] * 2, "permute all original controls"),
        ("placebo", [[0, 1, 2]] * 2, "integer matrix"),
    ],
)
def test_invalid_resampling_plan(multi_panel, method, indices, message):
    with pytest.raises(ValueError, match=message):
        SyntheticDID().inference(multi_panel, method=method, replications=2, resamples=indices)


def test_single_treated_placebo_matches_independent_r(reference, multi_panel):
    panel = Panel(
        multi_panel.y[:, : multi_panel.n_control + 1], multi_panel.n_control, multi_panel.n_pre
    )
    expected = reference["single_treated_placebo"]
    result = SyntheticDID().inference(
        panel, method="placebo", replications=12, resamples=reference["placebo_resamples"]
    )
    np.testing.assert_allclose(result.draws, expected["draws"], rtol=0, atol=1e-10)
    assert result.estimate == pytest.approx(expected["att"], abs=1e-10)
    assert result.variance == pytest.approx(expected["variance"], abs=1e-10)


def test_inference_rejects_reordered_time_weights(multi_panel):
    from dataclasses import replace

    estimate = SyntheticDID().fit(multi_panel)
    reordered = replace(estimate, time_weights=estimate.time_weights.iloc[::-1])
    with pytest.raises(ValueError, match="fit of this panel"):
        SyntheticDID().inference(multi_panel, method="jackknife", estimate=reordered)


@pytest.mark.parametrize("weight_type", ["weights", "time_weights"])
@pytest.mark.parametrize("bad_value", [-0.1, np.nan, np.inf, 2])
def test_inference_rejects_invalid_fitted_simplex(multi_panel, weight_type, bad_value):
    from dataclasses import replace

    estimate = SyntheticDID().fit(multi_panel)
    changed = getattr(estimate, weight_type).copy()
    changed.iloc[0] = bad_value
    invalid = replace(estimate, **{weight_type: changed})
    with pytest.raises(ValueError, match="finite simplex"):
        SyntheticDID().inference(multi_panel, method="jackknife", estimate=invalid)


def test_modified_path_cannot_center_a_valid_sdid_interval(multi_panel):
    from dataclasses import replace

    model = SyntheticDID()
    fit = model.fit(multi_panel)
    other = replace(fit, counterfactual=fit.counterfactual + 1)
    with pytest.raises(ValueError, match="agree with the fitted SDID weights"):
        model.inference(multi_panel, estimate=other, method="jackknife")


def test_normal_interval_at_largest_representable_coverage(multi_panel):
    result = SyntheticDID().inference(multi_panel, method="jackknife")
    bounds = result.confidence_interval(np.nextafter(1.0, 0.0))
    assert np.isfinite(bounds).all()
    assert bounds[0] < result.estimate < bounds[1]
