"""Analytic optimality checks for failed-MSCMT quadratic repairs."""

import numpy as np
import pytest

from pysynth._mscmt_qp import solve


def test_weighted_quadratic_analytic_solution_and_scale_invariance():
    design = np.array([[-1.0, 1.0], [-3.0, 1.0]])
    weights, diagnostics = solve(design)
    np.testing.assert_allclose(weights, [0.3, 0.7], atol=1e-10)
    assert diagnostics["objective"] == pytest.approx(0.1)
    assert diagnostics["dual_gap"] < 1e-10
    scaled, other = solve(design * 1e6)
    np.testing.assert_allclose(weights, scaled, atol=1e-10)
    assert other["objective"] == pytest.approx(diagnostics["objective"] * 1e12)


def test_duplicate_sampled_donors_keep_identical_quadratic():
    design = np.array([[-1.0, 1.0, 1.0], [-3.0, 1.0, 1.0]])
    weights, diagnostics = solve(design)
    assert weights[0] == pytest.approx(0.3, abs=1e-10)
    assert weights[1:].sum() == pytest.approx(0.7, abs=1e-10)
    assert diagnostics["objective"] == pytest.approx(0.1)
    assert diagnostics["dual_gap"] < 1e-10
