"""Independent derivative and degeneracy checks for the nested SCM objective."""

import numpy as np
import pytest

from pysynth import SCM, Panel
from pysynth._nested import _DerivativeUnavailable, _outer_derivative, _polish_weights


@pytest.mark.parametrize("pen", [0.0, 0.7])
def test_implicit_gradient_matches_central_difference_of_closed_form_qp(pen):
    # The third donor lies below the first two donors' convex segment and
    # remains inactive. The two active donors give a scalar quadratic with
    # an independently available minimizer t, the weight on donor two.
    x = np.array([[0.0, 2.0, 4.0], [0.0, 1.0, -2.0]])
    target = np.array([0.8, 0.7])
    a = x - target[:, None]
    pairwise = a**2
    v = np.array([0.4, 0.6])
    z = np.array([[0.0, 2.0, 8.0], [1.0, 3.0, -3.0], [4.0, 2.0, 1.0]])
    y = np.array([-0.2, 1.8, 3.1])

    def exact_weights(candidate):
        t = (
            1.6 * candidate[0]
            + 0.7 * candidate[1]
            - pen * (0.8 * candidate[0] - 0.4 * candidate[1]) / 2
        ) / (4 * candidate[0] + candidate[1])
        return np.array([1 - t, t, 0.0])

    def exact_outer_loss(candidate):
        residual = y - z @ exact_weights(candidate)
        return residual @ residual

    # Deliberately include a tiny spurious positive interior-point weight.
    polished, active, kkt, dual, inactive = _polish_weights(
        a, pairwise, v, pen, np.array([0.5, 0.499999, 0.000001])
    )
    np.testing.assert_allclose(polished, exact_weights(v), atol=1e-12)
    np.testing.assert_array_equal(active, [0, 1])
    analytic = _outer_derivative(
        a, pairwise, pen, z, y - z @ polished, polished, active, kkt, dual, inactive
    )
    h = 1e-6
    numeric = np.array(
        [
            (exact_outer_loss(v + h * direction) - exact_outer_loss(v - h * direction)) / (2 * h)
            for direction in np.eye(2)
        ]
    )
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)


def test_singular_donor_problem_declines_implicit_derivative():
    a = np.array([[1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(_DerivativeUnavailable, match="singular"):
        _polish_weights(a, a**2, np.array([0.4, 0.6]), 0.0, np.array([0.5, 0.5]))


def test_duplicate_donors_use_checked_derivative_free_recovery():
    pytest.importorskip("SyntheticControlMethods")
    control = np.arange(8, dtype=float)
    treated = control + 0.2 * np.sin(control)
    treated[4:] += 2
    panel = Panel(np.column_stack([control, control, treated]), 2, 4)
    fit = SCM(n_optim=1).fit(panel)
    np.testing.assert_allclose(fit.counterfactual, control, atol=1e-10)
    attempts = fit.metadata["optimization"][0]["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["derivative"] == "active-set KKT adjoint"
    assert not attempts[0]["success"] and "singular" in attempts[0]["message"]
    assert attempts[1]["solver"] == "Powell" and attempts[1]["success"]
