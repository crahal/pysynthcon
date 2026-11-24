"""Original SCUL preprocessing rules, including rare-value predictors."""

import numpy as np
import pytest

from pysynth import SCUL, Panel
from pysynth.scul import _near_zero_variance, _reference_preprocessing


@pytest.mark.parametrize(
    ("counts", "rejected"),
    [([20], True), ([19, 1], False), ([20, 1], True), ([18, 1, 1], False), ([20, 1, 1], False)],
)
def test_caret_default_frequency_and_unique_percentage_thresholds(counts, rejected):
    # The frequency cutoff is strictly greater than 19; both tests must pass.
    values = np.repeat(np.arange(len(counts)), counts)[:, None]
    assert bool(_near_zero_variance(values)[0]) is rejected
    np.testing.assert_array_equal(_near_zero_variance(values * 1e-180), [rejected])


def _screening_panel():
    rng = np.random.default_rng(731)
    continuous = rng.normal(size=(40, 3))
    rare_late = np.zeros(40)
    rare_late[0] = 1
    rare_late[25:] = np.arange(15)
    target = continuous @ [0.2, 0.5, 0.3]
    return Panel(np.column_stack([continuous, rare_late, target]), 4, 30)


def test_donor_can_become_near_constant_only_in_a_later_training_window():
    panel = _screening_panel()
    assert not _near_zero_variance(panel.control)[3]
    assert not _near_zero_variance(panel.control[:10])[3]
    assert _near_zero_variance(panel.control[:21])[3]
    keep, constant = _reference_preprocessing(panel, range(10, 26))
    np.testing.assert_array_equal(keep, [True, True, True, False])
    assert not constant.any()


def test_original_full_series_screen_can_reject_a_donor_before_fold_checks():
    rng = np.random.default_rng(22)
    continuous = rng.normal(size=(100, 3))
    full_rare = np.zeros(100)
    full_rare[0] = 1
    panel = Panel(np.column_stack([continuous, full_rare, continuous[:, 0]]), 4, 20)
    assert not _near_zero_variance(panel.control[:20])[3]
    keep, _ = _reference_preprocessing(panel, range(5, 19))
    np.testing.assert_array_equal(keep, [True, True, True, False])


def test_near_zero_variance_target_is_rejected_before_fitting():
    original = _screening_panel()
    changed = original.y.copy()
    changed[:, -1] = 0
    changed[0, -1] = 1
    with pytest.raises(ValueError, match="near-zero-variance target"):
        SCUL(initial_window=10, horizon=5).fit(Panel(changed, 4, 30))


@pytest.mark.parametrize("backend", ["python", "r"])
def test_single_validation_fold_cannot_define_the_sample_sd_rule(panel, backend):
    with pytest.raises(ValueError, match="1se rule needs at least two"):
        SCUL(backend=backend, initial_window=18, horizon=2, selection="1se").fit(panel)


def test_python_scul_preserves_rolling_cv_after_changing_outcome_units(panel):
    model = SCUL(backend="python", n_alphas=8, selection="min")
    original = model.fit(panel)
    rescaled = model.fit(Panel(1e-18 * panel.y, panel.n_control, panel.n_pre))
    np.testing.assert_allclose(rescaled.weights, original.weights, atol=1e-8)
    assert rescaled.att / 1e-18 == pytest.approx(original.att, abs=1e-8)
    np.testing.assert_allclose(
        np.array(rescaled.metadata["alphas"]) / 1e-18, original.metadata["alphas"]
    )
