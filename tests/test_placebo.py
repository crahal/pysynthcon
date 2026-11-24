"""Unit-invariant RMSPE-rank diagnostics, including exact-fit edge cases."""

import numpy as np
import pytest

from pysynth import DifferenceInDifferences, Panel, placebo_test


@pytest.mark.parametrize("scale", [1e-15, 1e-200, 1e200])
def test_placebo_rank_and_rmspe_are_invariant_to_outcome_units(scale):
    rng = np.random.default_rng(82)
    outcomes = rng.normal(size=(12, 5)).cumsum(axis=0)
    outcomes[8:, -1] += 5
    original = placebo_test(
        DifferenceInDifferences(), Panel(outcomes, 4, 8), max_pre_rmse_ratio=1.1
    )
    scaled = placebo_test(
        DifferenceInDifferences(), Panel(outcomes * scale, 4, 8), max_pre_rmse_ratio=1.1
    )
    assert np.isfinite(scaled.treated.pre_rmse)
    assert scaled.treated.pre_rmse / scale == pytest.approx(original.treated.pre_rmse)
    np.testing.assert_allclose(scaled.table.rmspe_ratio, original.table.rmspe_ratio, atol=1e-13)
    np.testing.assert_array_equal(scaled.table.eligible, original.table.eligible)
    assert scaled.p_value == original.p_value


@pytest.mark.parametrize("effect", [0, 3])
def test_exact_zero_pre_fit_convention_and_rank(effect):
    outcomes = np.tile(np.arange(6, dtype=float)[:, None], (1, 4))
    outcomes[4:, -1] += effect
    diagnostic = placebo_test(DifferenceInDifferences(), Panel(outcomes, 3, 4))
    assert diagnostic.treated.pre_rmse == 0
    assert diagnostic.table.iloc[0].rmspe_ratio == (np.inf if effect else 0)
    assert (diagnostic.table.iloc[1:].rmspe_ratio == 0).all()
    assert diagnostic.p_value == (0.25 if effect else 1)


def test_missing_placebo_prediction_cannot_count_as_an_exact_fit(panel):
    class DropsLastPlaceboPrediction:
        def fit(self, supplied):
            fit = DifferenceInDifferences().fit(supplied)
            if supplied.units[-1] != panel.units[-1]:
                fit.counterfactual = fit.counterfactual.iloc[:-1]
            return fit

    with pytest.raises(ValueError, match="counterfactual labels"):
        placebo_test(DropsLastPlaceboPrediction(), panel)


@pytest.mark.parametrize(
    "pre, pseudo_post, treated_post",
    [
        (1e-300, 1e280, 1e289),  # both displayed quotients overflow
        (1e300, 1e-280, 1e-270),  # both displayed quotients underflow
    ],
)
def test_extreme_rmspe_ratios_do_not_turn_distinct_fits_into_ties(pre, pseudo_post, treated_post):
    from pysynth.results import make_estimate

    panel = Panel(np.zeros((4, 4)), 3, 2)

    class ExtremeFit:
        def fit(self, sample):
            post = treated_post if sample.units[-1] == panel.units[-1] else pseudo_post
            return make_estimate(
                sample, "extreme", np.zeros(sample.n_control), counterfactual=[pre, pre, post, post]
            )

    result = placebo_test(ExtremeFit(), panel)
    assert result.table.rmspe_ratio.nunique() == 1
    assert result.table.at_least_treated_ratio.tolist() == [True, False, False, False]
    assert result.p_value == 0.25


def test_reassignment_rebuilds_each_pool_and_predictor_specification(panel):
    samples = []

    def factory(assignment):
        samples.append(assignment)
        return DifferenceInDifferences()

    result = placebo_test(factory, panel, donor_pool="reassign")
    assert len(samples) == len(panel.units)
    assert result.donor_pool == "reassign"
    for sample in samples:
        unit = sample.units[-1]
        assert sample.units[:-1] == tuple(other for other in panel.units if other != unit)
        assert sample.n_pre == panel.n_pre
        assert sample.times.equals(panel.times)
        for i, name in enumerate(sample.units):
            np.testing.assert_array_equal(sample.y[:, i], panel.y[:, panel.units.index(name)])
    assert all(panel.units[-1] in fit.weights.index for fit in result.placebos.values())
    default = placebo_test(DifferenceInDifferences(), panel)
    assert all(panel.units[-1] not in fit.weights.index for fit in default.placebos.values())
    assert result.gaps.columns.tolist() == [panel.units[-1], *panel.units[:-1]]
    for unit, fit in result.placebos.items():
        np.testing.assert_allclose(result.gaps[unit], fit.effects)


@pytest.mark.parametrize("scale", [1, 1e-200, 1e200])
def test_mspe_cutoff_rescreens_all_fits_and_ties_without_refitting(scale):
    from pysynth.results import make_estimate

    units = ("A", "B", "C", "T")
    panel = Panel(np.zeros((4, 4)), 3, 2, units=units)
    errors = {"A": (2, 4), "B": (3, 6), "C": (1, 0.5), "T": (1, 2)}
    calls = []

    class Fit:
        def fit(self, sample):
            unit = sample.units[-1]
            pre, post = errors[unit]
            gaps = np.array([pre, pre, post, post]) * scale
            return make_estimate(
                sample, "fixed gaps", np.zeros(sample.n_control), counterfactual=-gaps
            )

    def factory(sample):
        calls.append(sample.units[-1])
        return Fit()

    result = placebo_test(factory, panel, donor_pool="reassign")
    assert calls == ["T", "A", "B", "C"]
    assert result.table.mspe_ratio.to_dict() == {"A": 4, "B": 4, "C": 0.25, "T": 4}
    assert result.rank == 3 and result.p_value == 0.75
    screened = result.with_pre_mspe_cutoff(4)
    assert screened.table.index[screened.table.eligible].tolist() == ["T", "A", "C"]
    assert screened.p_value == pytest.approx(2 / 3)
    tight = screened.with_pre_mspe_cutoff(2)
    assert tight.table.index[tight.table.eligible].tolist() == ["T", "C"]
    assert tight.p_value == 0.5
    assert tight.with_pre_mspe_cutoff(None).p_value == result.p_value
    assert len(calls) == len(units)
    assert tight.with_pre_mspe_cutoff(4).table.equals(screened.table)
    assert tight.gaps.equals(result.gaps) and result.table.eligible.all()
    direct = placebo_test(Fit(), panel, max_pre_mspe_ratio=4)
    assert direct.table.equals(screened.table)
    rmspe = placebo_test(Fit(), panel, max_pre_rmse_ratio=2)
    assert rmspe.table.equals(screened.table)
    assert rmspe.max_pre_rmse_ratio == 2 and rmspe.max_pre_mspe_ratio is None


@pytest.mark.parametrize(
    "options",
    [
        {"donor_pool": "unknown"},
        {"max_pre_mspe_ratio": -1},
        {"max_pre_mspe_ratio": False},
        {"max_pre_rmse_ratio": float("inf")},
        {"max_pre_rmse_ratio": 2, "max_pre_mspe_ratio": 4},
    ],
)
def test_invalid_placebo_options_fail_before_refitting(panel, options):
    def factory(_):
        pytest.fail("invalid options should fail before refitting")

    with pytest.raises(ValueError):
        placebo_test(factory, panel, **options)


def test_two_assignments_retain_one_donor_each_with_tuple_labels():
    panel = Panel(np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]), 1, 2, units=(("D", 0), ("T", 1)))
    with pytest.raises(ValueError, match="donor"):
        placebo_test(DifferenceInDifferences(), panel)
    diagnostic = placebo_test(DifferenceInDifferences(), panel, donor_pool="reassign")
    assert diagnostic.table.index.tolist() == [("T", 1), ("D", 0)]
    assert diagnostic.gaps.columns.tolist() == [("T", 1), ("D", 0)]
    assert diagnostic.p_value == 1 and diagnostic.rank == 2


def test_plot_placebo_gaps_retains_time_axis_and_marks_eligible_sample(panel, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from pysynth.placebo_plotting import plot_placebo_gaps
    from pysynth.plotting import save_figure

    result = placebo_test(DifferenceInDifferences(), panel, donor_pool="reassign")
    screened = result.with_pre_mspe_cutoff(1)
    fig, ax = plot_placebo_gaps(screened, treated_label="Treated region", ylabel="Gap (packs)")
    try:
        assert ax.get_ylabel() == "Gap (packs)"
        legend = [t.get_text() for t in ax.get_legend().get_texts()]
        n = int(screened.table.eligible.sum())
        assert legend == [
            f"Treated region · post/pre MSPE rank {screened.rank}/{n}",
            f"{n - 1} of {len(screened.placebos)} placebo paths · pre-MSPE ≤ 1× treated",
        ]
        np.testing.assert_array_equal(ax.lines[-1].get_xdata(), panel.times)
        np.testing.assert_allclose(ax.lines[-1].get_ydata(), result.treated.effects)
        eligible = set(screened.table.index[screened.table.eligible & ~screened.table.treated])
        assert {
            line.get_label() for line in ax.lines if line.get_label() in result.placebos
        } == eligible
        paths = save_figure(fig, tmp_path / "placebo", formats=("svg",))
        assert "post/pre MSPE rank" in paths[0].read_text()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("policy", ["reassign", "untreated"])
def test_parallel_and_serial_preserve_results_and_assignment_order(panel, policy):
    def factory(sample):
        return DifferenceInDifferences()

    options = dict(max_pre_mspe_ratio=4, donor_pool=policy)
    serial = placebo_test(factory, panel, **options)
    parallel = placebo_test(factory, panel, n_jobs=2, **options)
    assert parallel.table.equals(serial.table)
    assert parallel.gaps.equals(serial.gaps)
    assert parallel.p_value == serial.p_value
    assert list(parallel.placebos) == list(serial.placebos)


def test_parallel_predictor_factory_preserves_fits_and_optimizer_diagnostics():
    upstream = pytest.importorskip("pysyncon")
    from pysynth import PredictorSyntheticControl

    y = np.random.default_rng(15).normal(size=(9, 4)).cumsum(axis=0)
    panel = Panel(y, 3, 6, units=tuple("ABCZ"))
    frame = panel.to_frame()

    def factory(assignment):
        dataprep = upstream.Dataprep(
            foo=frame,
            predictors=["outcome"],
            predictors_op="mean",
            dependent="outcome",
            unit_variable="unit",
            time_variable="time",
            treatment_identifier=assignment.units[-1],
            controls_identifier=list(assignment.units[:-1]),
            time_predictors_prior=range(6),
            time_optimize_ssr=range(6),
            special_predictors=[("outcome", [1, 4], "mean")],
        )
        return PredictorSyntheticControl(dataprep=dataprep)

    serial = placebo_test(factory, panel, donor_pool="reassign")
    parallel = placebo_test(factory, panel, donor_pool="reassign", n_jobs=2)
    np.testing.assert_allclose(parallel.gaps, serial.gaps, rtol=1e-9, atol=1e-9)
    assert parallel.rank == serial.rank
    for first, second in zip(
        [serial.treated, *serial.placebos.values()], [parallel.treated, *parallel.placebos.values()]
    ):
        assert first.weights.index.equals(second.weights.index)
        assert second.metadata["backend_method"] == "synth"
        assert second.metadata["optimization"] == first.metadata["optimization"]
        assert second.metadata["optimization"]["outer_attempts"][-1]["success"]


def test_parallel_fits_report_warnings_and_failures_to_caller():
    import warnings

    panel = Panel(np.zeros((4, 3)), 2, 2, units=tuple("ABZ"))

    def warn_assignment(sample):
        warnings.warn(f"assignment {sample.units[-1]}", RuntimeWarning, stacklevel=1)
        return DifferenceInDifferences()

    with pytest.warns(RuntimeWarning) as caught:
        placebo_test(warn_assignment, panel, n_jobs=2)
    assert [str(w.message) for w in caught] == ["assignment Z", "assignment A", "assignment B"]

    def fail_assignment(sample):
        if sample.units[-1] == "A":
            raise ValueError("unrecoverable fit A")
        return DifferenceInDifferences()

    with pytest.raises(ValueError, match="unrecoverable fit A"):
        placebo_test(fail_assignment, panel, n_jobs=2)


@pytest.mark.parametrize("invalid", [True, -1, 0, 1.5])
def test_invalid_worker_count_is_rejected(panel, invalid):
    with pytest.raises(ValueError, match="n_jobs"):
        placebo_test(DifferenceInDifferences(), panel, n_jobs=invalid)
