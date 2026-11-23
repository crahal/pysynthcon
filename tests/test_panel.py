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
