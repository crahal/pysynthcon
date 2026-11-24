"""Explicit resample budgets preserve the SDID reference algorithm and draws."""

import hashlib
import json
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from pysynth import Panel, SyntheticDID
from pysynth.sdid import _fw


@pytest.fixture(scope="module")
def reference():
    return json.loads((Path(__file__).parent / "fixtures/sdid_inference_r.json").read_text())


@pytest.fixture(scope="module")
def multi_panel(reference):
    return Panel(reference["y"], reference["n_control"], reference["n_pre"])


@pytest.fixture(scope="module")
def short_fit(multi_panel):
    with pytest.warns(RuntimeWarning, match="SDID reached max_iter"):
        return SyntheticDID(max_iter=2).fit(multi_panel)


def _options(reference, fit):
    return {
        "method": "placebo",
        "replications": 2,
        "resamples": np.array(reference["placebo_resamples"][:2]),
        "estimate": fit,
    }


def test_explicit_budget_overrides_supplied_fit_but_retains_other_settings(
    reference, multi_panel, short_fit
):
    options = _options(reference, short_fit)
    # Constructor settings must not silently replace those of an existing fit.
    model = SyntheticDID(max_iter=50_000, noise=10, min_decrease=1)
    with pytest.warns(RuntimeWarning, match="1 of 2 draws reached max_iter \\(2\\)"):
        inherited = model.inference(multi_panel, **options)
    extended = model.inference(
        multi_panel, **options, max_iter=np.int64(100), require_convergence=True
    )
    assert inherited.metadata["original_max_iter"] == inherited.metadata["max_iter"] == 2
    assert inherited.metadata["capped_draws"] == 1
    assert inherited.metadata["capped_solvers"] == {"omega": 0, "lambda": 1}
    assert extended.metadata["original_max_iter"] == 2
    assert extended.metadata["max_iter"] == extended.metadata["settings"]["max_iter"] == 100
    for key in ("noise", "zeta_omega", "zeta_lambda", "min_decrease"):
        assert extended.metadata["settings"][key] == short_fit.metadata[key]
    assert extended.metadata["capped_draws"] == 0
    assert extended.metadata["capped_solvers"] == {"omega": 0, "lambda": 0}
    assert extended.estimate == inherited.estimate == short_fit.att
    assert abs(extended.draws[1] - inherited.draws[1]) > 1e-3
    np.testing.assert_array_equal(extended.resamples, inherited.resamples)
    diagnostics = extended.metadata["solver_diagnostics"]
    assert max(d["lambda"]["iterations"] for d in diagnostics) > 2
    assert all(d[side]["max_iter"] == 100 for d in diagnostics for side in ("omega", "lambda"))


def test_strict_failure_is_actionable_and_does_not_warn(reference, multi_panel, short_fit):
    with warnings.catch_warnings(record=True) as emitted:
        with pytest.raises(RuntimeError) as error:
            SyntheticDID().inference(
                multi_panel, **_options(reference, short_fit), require_convergence=True
            )
    message = str(error.value)
    assert "1 of 2 draws reached max_iter (2)" in message
    assert "omega: 0, lambda: 1" in message
    assert "inference(max_iter=...)" in message
    assert "budget is inherited from that fit" in message
    assert not emitted


def test_inference_override_without_supplied_fit_changes_only_resample_budget(
    reference, multi_panel, short_fit
):
    with pytest.warns(RuntimeWarning, match="SDID reached max_iter"):
        result = SyntheticDID(max_iter=2).inference(
            multi_panel,
            method="placebo",
            resamples=reference["placebo_resamples"][:2],
            replications=2,
            max_iter=100,
            require_convergence=True,
        )
    assert result.estimate == short_fit.att
    assert result.metadata["original_max_iter"] == 2
    assert result.metadata["max_iter"] == 100
    assert result.metadata["capped_draws"] == 0


def test_strict_success_checks_resamples_only(reference, multi_panel, short_fit):
    assert not short_fit.metadata["omega_solver"]["stopping_rule_met"]
    result = SyntheticDID().inference(
        multi_panel,
        **_options(reference, short_fit),
        max_iter=100,
        require_convergence=np.bool_(True),
    )
    assert result.metadata["require_convergence"] is True
    assert result.metadata["capped_draws"] == 0
    assert result.estimate == short_fit.att
    assert all(
        diagnostic[side]["stopping_rule_met"]
        for diagnostic in result.metadata["solver_diagnostics"]
        for side in ("omega", "lambda")
    )


@pytest.mark.parametrize("budget", [True, np.bool_(True), 0, 1, -2, 2.5, "100", np.nan, np.inf])
def test_invalid_explicit_budget(budget, multi_panel, short_fit):
    with pytest.raises(ValueError, match="max_iter must be an integer >= 2"):
        SyntheticDID().inference(multi_panel, estimate=short_fit, method="placebo", max_iter=budget)


def test_explicit_budget_cannot_reduce_original_budget(multi_panel):
    fit = SyntheticDID().fit(multi_panel)
    with pytest.raises(ValueError, match="at least the original fit's iteration budget"):
        SyntheticDID().inference(multi_panel, estimate=fit, max_iter=9999)


@pytest.mark.parametrize("strict", [None, 0, 1, "False"])
def test_convergence_requirement_must_be_boolean(strict, multi_panel, short_fit):
    with pytest.raises(ValueError, match="require_convergence must be a boolean"):
        SyntheticDID().inference(multi_panel, estimate=short_fit, require_convergence=strict)


def test_jackknife_rejects_unused_iteration_override(multi_panel, short_fit):
    with pytest.raises(ValueError, match="unavailable for fixed-weight jackknife"):
        SyntheticDID().inference(multi_panel, estimate=short_fit, method="jackknife", max_iter=100)
    result = SyntheticDID().inference(
        multi_panel, estimate=short_fit, method="jackknife", require_convergence=True
    )
    assert result.metadata["capped_draws"] == 0
    assert result.metadata["solver_diagnostics"] == []


@pytest.mark.parametrize("method", ["bootstrap", "placebo", "jackknife"])
def test_default_draws_still_match_independent_r(reference, multi_panel, method):
    options = {"method": method}
    if method != "jackknife":
        options.update(resamples=reference[f"{method}_resamples"], replications=12)
    with warnings.catch_warnings(record=True):
        default = SyntheticDID().inference(multi_panel, **options)
        explicit_defaults = SyntheticDID().inference(
            multi_panel, **options, max_iter=None, require_convergence=False
        )
    np.testing.assert_array_equal(default.draws, explicit_defaults.draws)
    np.testing.assert_allclose(default.draws, reference[method]["draws"], rtol=0, atol=1e-10)
    assert default.variance == pytest.approx(reference[method]["variance"], rel=0, abs=1e-10)
    assert default.metadata["original_max_iter"] == default.metadata["max_iter"] == 10_000


@pytest.mark.parametrize("method", ["bootstrap", "placebo"])
def test_extended_budget_matches_independent_original_r(multi_panel, method):
    fixtures = Path(__file__).parent / "fixtures"
    reference = json.loads((fixtures / "sdid_extended_r.json").read_text())
    original_bytes = (fixtures / reference["input_fixture"]).read_bytes()
    assert hashlib.sha256(original_bytes).hexdigest() == reference["input_fixture_sha256"]
    fit = SyntheticDID().fit(multi_panel)
    original_metadata = deepcopy(fit.metadata)
    result = SyntheticDID().inference(
        multi_panel,
        estimate=fit,
        method=method,
        replications=3,
        resamples=reference[f"{method}_resamples"],
        max_iter=reference["max_iter"],
        require_convergence=True,
    )
    np.testing.assert_allclose(result.draws, reference[method]["draws"], rtol=0, atol=1e-10)
    assert result.variance == pytest.approx(reference[method]["variance"], rel=0, abs=1e-10)
    assert result.estimate == pytest.approx(reference["att"], rel=0, abs=1e-10)
    assert result.estimate == fit.att
    assert fit.metadata == original_metadata
    np.testing.assert_allclose(fit.weights, reference["omega"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(fit.time_weights, reference["lambda"], rtol=0, atol=1e-10)
    assert result.metadata["original_max_iter"] == reference["original_max_iter"]
    assert result.metadata["max_iter"] == reference["max_iter"]
    assert result.metadata["capped_draws"] == 0
    for actual, expected in zip(
        result.metadata["solver_diagnostics"], reference[method]["solver_diagnostics"], strict=True
    ):
        for side in ("omega", "lambda"):
            assert actual[side]["iterations"] == expected[side]["iterations"]
            assert actual[side]["stopping_rule_met"]
            assert actual[side]["last_objective_decrease"] == pytest.approx(
                expected[side]["objective_decrease"], abs=5e-15
            )
            assert actual[side]["stopping_threshold"] == expected[side]["threshold"]
    if method == "bootstrap":
        assert result.metadata["solver_diagnostics"][1]["lambda"]["iterations"] > 100_000


def test_inference_does_not_mutate_fit_or_resampling_plan(reference, multi_panel, short_fit):
    options = _options(reference, short_fit)
    original_metadata = deepcopy(short_fit.metadata)
    original_omega = short_fit.weights.copy()
    original_lambda = short_fit.time_weights.copy()
    original_indices = options["resamples"].copy()
    with pytest.raises(RuntimeError, match="draws reached max_iter"):
        SyntheticDID().inference(multi_panel, **options, require_convergence=True)
    result = SyntheticDID().inference(multi_panel, **options, max_iter=100)
    assert short_fit.metadata == original_metadata
    np.testing.assert_array_equal(short_fit.weights, original_omega)
    np.testing.assert_array_equal(short_fit.time_weights, original_lambda)
    np.testing.assert_array_equal(options["resamples"], original_indices)
    result.metadata["settings"]["max_iter"] = 20
    result.resamples[0, 0] = -1
    assert short_fit.metadata == original_metadata
    np.testing.assert_array_equal(options["resamples"], original_indices)


@pytest.mark.parametrize("method", ["bootstrap", "placebo"])
def test_seeded_extended_draws_are_replayable(multi_panel, short_fit, method):
    model = SyntheticDID()
    options = {
        "method": method,
        "estimate": short_fit,
        "replications": 4,
        "max_iter": 10_000,
    }
    with warnings.catch_warnings(record=True):
        first = model.inference(multi_panel, **options, random_state=7)
        repeated = model.inference(multi_panel, **options, random_state=7)
        replayed = model.inference(multi_panel, **options, resamples=first.resamples)
    np.testing.assert_array_equal(first.draws, repeated.draws)
    np.testing.assert_array_equal(first.draws, replayed.draws)
    np.testing.assert_array_equal(first.resamples, repeated.resamples)
    assert first.variance == replayed.variance


def test_frank_wolfe_diagnostics_report_actual_objective_decrease():
    x = np.array([[0, 1, 2], [1, 0, 1], [2, 3, 0], [4, 1, 2]], dtype=float)
    y = np.array([2, 0.5, 1, 3])
    initial = np.full(3, 1 / 3)
    _, previous = _fw(x, y, 0.5, initial, 2, 0)
    _, current = _fw(x, y, 0.5, initial, 3, 0)
    assert previous["termination_reason"] == current["termination_reason"] == "max_iter"
    assert current["last_objective_decrease"] == previous["objective"] - current["objective"]
    assert current["last_objective_decrease"] > current["stopping_threshold"] == 0
    assert current["max_iter"] == current["iterations"] == 3
    _, stopped = _fw(x, y, 0.5, initial, 100, 1)
    assert stopped["termination_reason"] == "objective_decrease"
    assert stopped["stopping_rule_met"]
    assert stopped["last_objective_decrease"] <= stopped["stopping_threshold"] == 1
    assert stopped["iterations"] == 2
