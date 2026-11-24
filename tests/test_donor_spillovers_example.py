"""Check the paired experimental design in the executable simulation example."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysynth import (
    DifferenceInDifferences,
    Panel,
    SyntheticControl,
    backtest,
    conformal_test,
    donor_sensitivity,
)

nbformat = pytest.importorskip("nbformat")


@pytest.fixture(scope="module")
def design():
    notebook = Path(__file__).parents[1] / "simulated_examples" / "donor_spillovers.ipynb"
    cells = {cell.id: cell.source for cell in nbformat.read(notebook, as_version=4).cells}
    ns = {
        "np": np,
        "pd": pd,
        "Panel": Panel,
        "SyntheticControl": SyntheticControl,
        "DifferenceInDifferences": DifferenceInDifferences,
        "donor_sensitivity": donor_sensitivity,
        "backtest": backtest,
        "conformal_test": conformal_test,
    }
    for identifier in ("generate-panel", "paired-fits", "conformal-paired-draw"):
        exec(compile(cells[identifier], f"{notebook.name}#{identifier}", "exec"), ns)
    return ns


def test_spillover_moves_only_neighbor_outcomes_after_policy(design):
    clean, exposed, treated_y0 = design["generate_pair"](72)
    next_clean, _, _ = design["generate_pair"](73)
    np.testing.assert_array_equal(clean.y, design["generate_pair"](72)[0].y)
    assert not np.array_equal(clean.y, next_clean.y)
    expected = np.zeros_like(clean.y)
    expected[clean.n_pre :, : len(design["NEAR"])] = design["SPILLOVER"]
    np.testing.assert_allclose(exposed.y - clean.y, expected, rtol=0, atol=1e-12)
    truth = np.where(clean.times < 0, 0, design["TRUE_ATT"])
    np.testing.assert_allclose(clean.treated - treated_y0, truth, rtol=0, atol=1e-12)


def test_both_refits_match_did_spillover_identity(design):
    rows = pd.DataFrame(design["analyze_draw"](72, 1)).set_index(["method", "donor_pool"])
    assert len(rows) == 4
    fraction = len(design["NEAR"]) / (len(design["NEAR"]) + len(design["FAR"]))
    assert rows.loc[("DID", "All donors"), "paired_att_shift"] == pytest.approx(
        -design["SPILLOVER"] * fraction
    )
    np.testing.assert_allclose(rows.xs("Distant only", level="donor_pool").paired_att_shift, 0)
    np.testing.assert_allclose(
        rows.paired_att_shift, -design["SPILLOVER"] * rows.neighbor_weight, atol=1e-9, rtol=0
    )


def test_conformal_paired_design_recovers_the_same_distant_donor_tests(design):
    rows = pd.DataFrame(design["test_effect_paths"](72, 1))
    assert len(rows) == 8
    assert set(rows.null_effect) == {0, design["TRUE_ATT"]}
    assert (rows.n_rotations == design["N_PRE"] + design["N_POST"]).all()
    np.testing.assert_array_equal(rows.rejected, rows.p_value <= design["ALPHA"])
    distant = rows[rows.donor_pool == "Distant only"].set_index(["regime", "null_effect"])
    np.testing.assert_array_equal(
        distant.loc["No spillover", ["statistic", "p_value"]],
        distant.loc["Spillover", ["statistic", "p_value"]],
    )
