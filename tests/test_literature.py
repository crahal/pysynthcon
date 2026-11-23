import pytest

from pysynth.datasets import DATASETS, load_dataset
from pysynth.replication import check_replication, replicate


@pytest.mark.network
@pytest.mark.parametrize("name", list(DATASETS))
def test_academic_reference_specification(name, tmp_path):
    if name != "california":
        pytest.importorskip("pysyncon")
    frame = load_dataset(name, cache_dir=tmp_path)
    result = replicate(name, frame)
    check_replication(name, result)
