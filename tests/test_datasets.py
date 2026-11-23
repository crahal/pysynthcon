import hashlib
from dataclasses import replace
from unittest.mock import Mock

import pytest
import requests

from pysynth.datasets import DATASETS, load_dataset


@pytest.fixture
def response(monkeypatch):
    content = b"State;Year;PacksPerCapita;treated\nCalifornia;1989;1;1\n"
    monkeypatch.setitem(
        DATASETS,
        "california",
        replace(DATASETS["california"], sha256=hashlib.sha256(content).hexdigest()),
    )
    response = Mock(content=content)
    get = Mock(return_value=response)
    monkeypatch.setattr(requests, "get", get)
    return get, response


def test_requests_timeout_hash_cache(response, tmp_path):
    get, reply = response
    frame = load_dataset("california", cache_dir=tmp_path, timeout=15)
    assert frame.iloc[0].State == "California"
    get.assert_called_once_with(DATASETS["california"].url, timeout=15)
    reply.raise_for_status.assert_called_once()
    load_dataset("california", cache_dir=tmp_path)
    assert get.call_count == 1
    next(tmp_path.glob("*.csv")).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Checksum"):
        load_dataset("california", cache_dir=tmp_path)


def test_http_errors_not_suppressed(response):
    response[1].raise_for_status.side_effect = requests.HTTPError("not found")
    with pytest.raises(requests.HTTPError):
        load_dataset("california")


def test_download_hash_mismatch(response):
    response[1].content = b"wrong"
    with pytest.raises(ValueError, match="Checksum"):
        load_dataset("california")


def test_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown"):
        load_dataset("unknown")
