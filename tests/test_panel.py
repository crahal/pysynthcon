import numpy as np
import pandas as pd
import pytest

from pysynth import Panel


def test_long_data_round_trip_and_shuffling(panel):
    frame = panel.to_frame().sample(frac=1, random_state=2)
    actual = Panel.from_frame(
        frame,
        unit="unit",
        time="time",
        outcome="outcome",
        treated_unit="Z",
        treatment_time=20,
        controls=list("ABCDEF"),
    )
    np.testing.assert_array_equal(actual.y, panel.y)
    assert not actual.y.flags.writeable


@pytest.mark.parametrize("problem", ["duplicate", "missing", "infinite", "absent", "timing"])
def test_bad_frame_rejected(panel, problem):
    frame = panel.to_frame()
    timing = 20
    if problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    if problem == "missing":
        frame = frame.iloc[1:]
    if problem == "infinite":
        frame.loc[0, "outcome"] = np.inf
    if problem == "absent":
        frame = frame[frame.unit != "Z"]
    if problem == "timing":
        timing = 20.5
    with pytest.raises(ValueError):
        Panel.from_frame(
            frame,
            unit="unit",
            time="time",
            outcome="outcome",
            treated_unit="Z",
            treatment_time=timing,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_control": 0},
        {"n_control": 7},
        {"n_pre": 1},
        {"n_pre": 30},
        {"n_pre": 2.5},
        {"times": list(range(29, -1, -1))},
        {"units": tuple("AAAAAAZ")},
    ],
)
def test_bad_matrix_rejected(panel, kwargs):
    options = dict(n_control=6, n_pre=20)
    options.update(kwargs)
    with pytest.raises(ValueError):
        Panel(panel.y, **options)


def test_simultaneous_treated_and_copy(panel):
    source = panel.y.copy()
    p = Panel(source, 5, 20)
    source[:] = 0
    assert p.n_treated == 2
    np.testing.assert_allclose(p.treated, panel.y[:, 5:].mean(axis=1))


@pytest.mark.parametrize(
    "units", [("time", "outcome", "unit"), (False, True), (("a", 1), ("b", 1))]
)
def test_round_trip_preserves_labels_that_collide_with_dataframe_conventions(units):
    values = np.arange(5 * len(units)).reshape(5, len(units))
    original = Panel(values, len(units) - 1, 3, units=units)
    recovered = Panel.from_frame(
        original.to_frame(),
        unit="unit",
        time="time",
        outcome="outcome",
        treatment_time=3,
        treated_unit=units[-1],
        controls=units[:-1],
    )
    assert recovered.units == original.units
    np.testing.assert_equal(recovered.y, original.y)


@pytest.mark.parametrize(
    "outcomes",
    [
        np.array([[1 + 1j, 0], [2 + 2j, 1], [1 + 1j, 1]]),
        np.array([[np.complex128(1j), 0], [1, 2], [3, 4]], dtype=object),
        np.full((3, 2), np.datetime64("NaT", "D")),
        np.ma.array([[1.0, 0], [2, 1], [3, 1]], mask=[[0, 1], [0, 0], [0, 0]]),
    ],
)
def test_nonreal_or_masked_outcomes_cannot_be_silently_coerced(outcomes):
    with pytest.raises(ValueError, match="observed, real"):
        Panel(outcomes, 1, 2)


def test_multitreated_mean_with_overflowing_sum():
    outcomes = np.tile([0.0, 1e308, 1e308], (3, 1))
    np.testing.assert_equal(Panel(outcomes, 1, 2).treated, np.full(3, 1e308))


@pytest.mark.parametrize("choice", ["treated_units", "controls"])
def test_plural_arguments_never_split_a_treatment_or_donor_label(choice):
    # CA and its two characters all exist, so guessing would fit another design.
    frame = Panel(np.arange(16).reshape(4, 4), 3, 2, units=("CA", "C", "A", "X")).to_frame()
    kwargs = (
        {"treated_units": "CA"}
        if choice == "treated_units"
        else {"treated_unit": "X", "controls": "CA"}
    )
    with pytest.raises(ValueError, match=choice):
        Panel.from_frame(
            frame, unit="unit", time="time", outcome="outcome", treatment_time=2, **kwargs
        )
    correctly_labelled = Panel.from_frame(
        frame,
        unit="unit",
        time="time",
        outcome="outcome",
        treatment_time=2,
        treated_unit="X",
        controls=["CA"],
    )
    assert correctly_labelled.units == ("CA", "X")


def test_unit_axis_rejects_a_single_string_even_if_its_length_matches():
    with pytest.raises(ValueError, match="units must be a sequence"):
        Panel(np.zeros((4, 2)), 1, 2, units="CA")
