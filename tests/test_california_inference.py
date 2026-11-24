"""Reproduce the paper's California placebo standard error and RNG stream."""

from dataclasses import replace

import pytest

from pysynth import Panel, SyntheticDID
from pysynth._r import reference_directory, require_rscript
from pysynth.datasets import DATASETS, load_dataset
from pysynth.replication import california_paper_inference


def test_california_paper_inference_rejects_other_designs_before_running_r(panel):
    with pytest.raises(ValueError, match="original 38 alphabetically ordered"):
        california_paper_inference(panel, None)


@pytest.mark.network
@pytest.mark.r_reference
def test_california_paper_standard_error_and_exact_ordering(tmp_path):
    try:
        reference_directory("california_rng.R")
        require_rscript()
    except ImportError as error:
        pytest.skip(str(error))
    panel = DATASETS["california"].panel(load_dataset("california", cache_dir=tmp_path))
    estimate = SyntheticDID().fit(panel)
    result = california_paper_inference(panel, estimate)
    assert result.standard_error == pytest.approx(8.36799308138054, abs=1e-9)
    assert round(result.standard_error, 1) == 8.4
    assert result.resamples.shape == (200, 38)
    provenance = result.metadata["paper_replication"]
    assert provenance["seed"] == 12345
    assert provenance["rng_kind"] == ["Mersenne-Twister", "Inversion", "Rejection"]
    assert "paper-results.Rmd" in provenance["source_url"]
    order = [1, 0, *range(2, 39)]
    reordered = Panel(panel.y[:, order], 38, 19, panel.times, tuple(panel.units[i] for i in order))
    with pytest.raises(ValueError, match="alphabetically ordered"):
        california_paper_inference(reordered, estimate)

    changed = replace(estimate, metadata={**estimate.metadata, "max_iter": 20})
    with pytest.raises(ValueError, match="original default SDID"):
        california_paper_inference(panel, changed)
