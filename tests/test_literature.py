import numpy as np
import pandas as pd
import pytest

from pysynth import Estimate, reference_comparison_table
from pysynth.datasets import DATASETS, load_dataset
from pysynth.replication import check_replication, reference_comparisons, replicate


@pytest.mark.network
@pytest.mark.parametrize("name", list(DATASETS))
def test_academic_reference_specification(name, tmp_path):
    if name != "california":
        pytest.importorskip("pysyncon")
    frame = load_dataset(name, cache_dir=tmp_path)
    result = replicate(name, frame)
    check_replication(name, result)


def reference_fit(times, n_pre, effect=-1.0):
    observed = pd.Series(np.linspace(5, 10, len(times)), index=times)
    return Estimate(
        "Historical", observed, observed - effect, pd.Series([1.0], index=["donor"]), n_pre=n_pre
    )


def test_basque_paper_percentage_is_separate_from_upstream_att():
    fit = reference_fit(pd.Index(range(1960, 1998)), n_pre=10)
    comparisons = reference_comparisons("basque", fit)
    table = reference_comparison_table(comparisons)
    assert table.window_start.tolist() == [1980, 1975]
    assert table.statistic.tolist() == ["mean_percent_effect", "mean_effect"]
    assert table.approximate.tolist() == [True, False]
    assert table.reference.tolist() == [-10, -0.6995647842110987]
    expected = (-100 / fit.observed.loc[1980:1997]).mean()
    assert table.loc[0, "reproduced"] == pytest.approx(expected)
    assert table.loc[1, "reproduced"] == -1
    assert "parity_tests/test_synth_basque.py" in table.loc[1, "source_url"]


def test_german_paper_and_software_targets_retain_different_precision_and_status():
    fit = reference_fit(pd.Index(range(1960, 2004)), n_pre=30)
    paper, code = reference_comparisons("germany", fit)
    assert paper.reference == -1600 and paper.approximate
    assert paper.reference_precision == 0
    assert code.reference == -1555.1346777620479 and not code.approximate
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
