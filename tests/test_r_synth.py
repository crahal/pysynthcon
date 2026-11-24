"""The Python bridge preserves the native R Synth design and output labels."""

from importlib.util import find_spec

import numpy as np
import pytest

from pysynth import Panel
from pysynth._r_synth import ReferenceSynthDataprep
from pysynth.datasets import DATASETS, load_dataset
from pysynth.replication import california_tobacco_estimator


def design(panel, donor_order, *, predictor_times=None):
    upstream = pytest.importorskip("pysyncon")
    pre = list(panel.times[: panel.n_pre])
    return upstream.Dataprep(
        foo=panel.to_frame(),
        predictors=["outcome"],
        predictors_op="mean",
        dependent="outcome",
        unit_variable="unit",
        time_variable="time",
        treatment_identifier=panel.units[-1],
        controls_identifier=donor_order,
        time_predictors_prior=pre if predictor_times is None else predictor_times,
        time_optimize_ssr=pre,
    )


def test_original_solver_order_and_panel_outcome_alignment(panel, monkeypatch):
    donors = list(panel.units[2:-1] + panel.units[:2])
    dp = design(panel, donors)
    pre = panel.y[: panel.n_pre]
    indexes = [panel.units.index(u) for u in donors]
    by_donor = dict(zip(donors, np.arange(1, 7) / 21))

    def run(matrices, library):
        np.testing.assert_array_equal(matrices["Z0"], pre[:, indexes])
        np.testing.assert_array_equal(matrices["Z1"], pre[:, -1, None])
        np.testing.assert_allclose(matrices["X0"], pre[:, indexes].mean(axis=0, keepdims=True))
        np.testing.assert_allclose(matrices["X1"], [[pre[:, -1].mean()]])
        return {
            "weights": list(by_donor.values()),
            "predictor_weights": [1.0],
            "loss_v": 0.4,
            "loss_w": 0.1,
            "r_console_diagnostics": "Warning: numerical convergence",
        }

    monkeypatch.setattr("pysynth._r_synth._run_synth", run)
    fit = ReferenceSynthDataprep(dp).fit(panel)
    assert fit.weights.to_dict() == by_donor
    assert tuple(fit.weights.index) == panel.units[:-1]
    np.testing.assert_allclose(fit.counterfactual, panel.control @ fit.weights)
    assert fit.metadata["r_console_diagnostics"] == "Warning: numerical convergence"


def test_post_outcome_mismatch_and_post_predictors_rejected_before_running_r(panel):
    donors = list(panel.units[:-1])
    model = ReferenceSynthDataprep(design(panel, donors))
    changed = panel.y.copy()
    changed[-1, -1] += 1
    with pytest.raises(ValueError, match="outcomes must match"):
        model.fit(Panel(changed, panel.n_control, panel.n_pre, panel.times, panel.units))
    model = ReferenceSynthDataprep(design(panel, donors, predictor_times=[panel.times[-1]]))
    with pytest.raises(ValueError, match="post-treatment"):
        model.fit(panel)


@pytest.mark.r_reference
def test_local_original_synth_fit_on_two_donors():
    if find_spec("rpy2") is None:
        pytest.skip("The optional rpy2 bridge is unavailable")
    panel = Panel(
        np.array([[4, 0, 1], [5, 1, 2], [6, 2, 3], [8, 4, 7]]), 2, 3, units=("B", "A", "T")
    )
    try:
        fit = ReferenceSynthDataprep(design(panel, ["A", "B"])).fit(panel)
    except ImportError as exc:
        pytest.skip(str(exc))
    # The exact population predictor match is a quarter B and three quarters A.
    # Synth's native ipop solution deliberately retains its own finite tolerance.
    np.testing.assert_allclose(fit.weights, [0.25, 0.75], atol=3e-5)
    assert fit.att == pytest.approx(2.0, abs=1e-4)
    assert fit.metadata["predictor_weights"] == [1.0]
    assert fit.metadata["loss_v"] == pytest.approx(fit.pre_rmse**2, abs=1e-12)
    assert fit.metadata["reference_package"] == "Synth"
    assert fit.metadata["reference_version"]


@pytest.mark.network
@pytest.mark.r_reference
def test_california_matches_independent_native_r_dataprep_and_synth():
    pytest.importorskip("pysyncon")
    if find_spec("rpy2") is None:
        pytest.skip("The optional rpy2 bridge is unavailable")
    tobacco = load_dataset("california_tobacco")
    panel = DATASETS["california_tobacco"].panel(tobacco)
    try:
        fit = california_tobacco_estimator(tobacco, panel, backend="r-synth").fit(panel)
    except ImportError as exc:
        pytest.skip(str(exc))
    if fit.metadata["reference_version"] != "1.1-10":
        pytest.skip("These numerical targets were recorded with Synth 1.1-10")
    # Native R dataprep and synth, with the seven predictor rows in the
    # authors' MATLAB example order. Allow for platform-specific nested fits.
    assert fit.metadata["loss_v"] == pytest.approx(3.20907826377028, abs=1e-4)
    assert fit.effects.loc[2000] == pytest.approx(-25.4345586536677, abs=5e-3)
    post_mspe = np.mean(fit.effects.iloc[panel.n_pre :] ** 2)
    assert post_mspe / fit.pre_rmse**2 == pytest.approx(120.492665748795, abs=0.05)
