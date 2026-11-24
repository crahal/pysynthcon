import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysynth import Estimate, reference_comparison_table
from pysynth.datasets import DATASETS, load_dataset
from pysynth.replication import (
    _basque_schooling_shares,
    check_replication,
    reference_comparisons,
    replicate,
)


@pytest.mark.network
@pytest.mark.parametrize("name", list(DATASETS))
def test_academic_reference_specification(name, tmp_path):
    if name != "california":
        pytest.importorskip("pysyncon")
    frame = load_dataset(name, cache_dir=tmp_path)
    result = replicate(name, frame)
    check_replication(name, result)
    if name == "basque":
        original = json.loads((Path(__file__).parent / "fixtures/basque_synth_r.json").read_text())
        assert result.weights.index.tolist() == original["donor_names"]
        np.testing.assert_allclose(result.weights, original["weights"], atol=1e-4, rtol=0)
        np.testing.assert_allclose(
            result.counterfactual, original["counterfactual"], atol=1e-4, rtol=0
        )
        prepared = _basque_schooling_shares(frame)
        means = prepared.loc[
            prepared.year.between(1964, 1969)
            & (prepared.regionname == "Basque Country (Pais Vasco)")
        ].mean(numeric_only=True)
        for column in ["school.illit", "school.prim", "school.med", "school.high"]:
            assert means[column] == pytest.approx(original["treated_predictors"][column], abs=1e-12)
        # Independent paper-printed values distinguish annual-share means
        # from the later software manual's shares of mean counts.
        np.testing.assert_array_equal(
            means[["school.illit", "school.prim", "school.med", "school.high"]].round(2),
            [3.32, 85.97, 7.46, 3.26],
        )


def reference_fit(times, n_pre, effect=-1.0):
    observed = pd.Series(np.linspace(5, 10, len(times)), index=times)
    return Estimate(
        "Historical", observed, observed - effect, pd.Series([1.0], index=["donor"]), n_pre=n_pre
    )


def test_basque_schooling_averages_annual_shares_and_combines_top_categories():
    frame = pd.DataFrame(
        {
            "regionname": ["region"] * 3,
            "year": [1964, 1965, 1970],
            "school.illit": [10.0, 0.0, 1000.0],
            "school.prim": [0.0, 30.0, 1000.0],
            "school.med": [0.0, 10.0, 1000.0],
            "school.high": [0.0, 5.0, 1000.0],
            "school.post.high": [0.0, 5.0, 1000.0],
        }
    )
    original = frame.copy(deep=True)
    prepared = _basque_schooling_shares(frame)
    columns = ["school.illit", "school.prim", "school.med", "school.high"]
    expected = np.array([[100.0, 0.0, 0.0, 0.0], [0.0, 60.0, 20.0, 20.0]])
    np.testing.assert_allclose(prepared.loc[:1, columns], expected)
    pd.testing.assert_series_equal(prepared.loc[2], original.loc[2])
    pd.testing.assert_frame_equal(frame, original)


def test_basque_paper_percentage_is_separate_from_original_r_att():
    fit = reference_fit(pd.Index(range(1960, 1998)), n_pre=10)
    comparisons = reference_comparisons("basque", fit)
    table = reference_comparison_table(comparisons)
    assert table.window_start.tolist() == [1980, 1975]
    assert table.statistic.tolist() == ["mean_percent_effect", "mean_effect"]
    assert table.approximate.tolist() == [True, False]
    assert table.reference.tolist() == [-10, -0.6995627075801111]
    expected = (-100 / fit.observed.loc[1980:1997]).mean()
    assert table.loc[0, "reproduced"] == pytest.approx(expected)
    assert pd.isna(table.loc[0, "difference"]) and pd.isna(table.loc[0, "difference_units"])
    assert not comparisons[0].comparable and comparisons[1].comparable
    assert table.loc[1, "reproduced"] == -1
    assert "Costs%20of%20Conflict.pdf#page=4" in table.loc[1, "source_url"]


def test_german_figure_pairs_reproduced_mean_with_authors_rounded_estimate():
    fit = reference_fit(pd.Index(range(1960, 2004)), n_pre=30)
    (paper,) = reference_comparisons("germany", fit)
    assert paper.reference == -1600 and paper.approximate
    assert paper.reference_precision == 0
    row = reference_comparison_table((paper,)).iloc[0]
    assert row.reproduced == fit.att
    assert "2025 erratum" in paper.context
    assert "PPP current" in paper.units


def test_california_table_one_references_are_bound_to_distinct_specifications():
    sdid = reference_fit(pd.Index(range(1970, 2001)), n_pre=19, effect=-15.6038278727)
    sc = reference_fit(sdid.observed.index, n_pre=19, effect=-19.61966347)
    did = reference_fit(sdid.observed.index, n_pre=19, effect=-27.34911108)
    comparisons = reference_comparisons("california", sdid, additional={"SC": sc, "DID": did})
    assert [ref.reference for ref in comparisons] == [-15.6, -19.6, -27.3]
    assert all(ref.estimate is fit for ref, fit in zip(comparisons, (sdid, sc, did)))
    assert all(ref.window == (1989, 2000) for ref in comparisons)
    with pytest.raises(ValueError, match="additional references"):
        reference_comparisons("california", sdid, additional={"Arbitrary SCM": sc})


def test_kansas_uses_author_vignette_not_a_paper_printed_headline():
    fit = reference_fit(pd.Index(np.arange(1990, 2016.25, 0.25)), n_pre=89, effect=-0.0401)
    (comparison,) = reference_comparisons("kansas", fit)
    assert comparison.source_label == "Authors’ vignette"
    assert comparison.window_label == "2012 Q2–2016 Q1"
    assert comparison.reference_precision == 4
    assert "7e70072232fe75057aa8b5d7af08c7e1aef53ae8" in comparison.source_url


@pytest.mark.parametrize("alteration", ["shifted", "missing", "nonfinite", "boundary"])
def test_matching_california_att_requires_complete_original_sample(alteration):
    times = np.arange(1970, 2001)
    if alteration == "shifted":
        times += 1
    elif alteration == "missing":
        times = np.delete(times, 24)
    fit = reference_fit(
        pd.Index(times), 18 if alteration == "boundary" else 19, -15.603827872733849
    )
    if alteration == "nonfinite":
        fit.counterfactual.iloc[-1] = np.nan
    error = ValueError if alteration == "nonfinite" else AssertionError
    message = "finite" if alteration == "nonfinite" else "complete historical outcome window"
    with pytest.raises(error, match=message):
        check_replication("california", fit)


def test_correct_mean_does_not_mask_wrong_kansas_quarterly_effects():
    fit = reference_fit(pd.Index(np.arange(1990, 2016.25, 0.25)), 89, -0.0401)
    with pytest.raises(AssertionError, match="Kansas post-treatment effects"):
        check_replication("kansas", fit)
