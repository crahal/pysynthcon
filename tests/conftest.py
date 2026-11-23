import numpy as np
import pytest

from pysynth import Panel


@pytest.fixture
def panel():
    rng = np.random.default_rng(82)
    x = rng.normal(size=(30, 6)).cumsum(axis=0) + np.arange(30)[:, None] * 0.2 + 20
    y = x @ np.array([0.2, 0.5, 0.3, 0, 0, 0])
    y[20:] += 3
    return Panel(np.column_stack([x, y]), 6, 20, units=tuple("ABCDEFZ"))
