"""Unit tests for the legacy ``mars_rec.data.utils`` download helpers (no network)."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from mars_rec.data import utils as data_utils
from mars_rec.utils import download

CSV = b"a,b\n1,2\n3,4\n"


class _FakeResponse:
    """Minimal stand-in for the object ``urllib.request.urlopen`` returns."""

    def __init__(self, body: bytes, content_length=None):
        self._body = body
        self._pos = 0
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}

    def read(self, size=-1):
        end = len(self._body) if size < 0 else self._pos + size
        chunk = self._body[self._pos : end]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_get(monkeypatch):
    calls = []

    def install(body=CSV, content_length=len(CSV)):
        def _urlopen(request, timeout=None, context=None):
            calls.append(request.full_url)
            return _FakeResponse(body, content_length)

        monkeypatch.setattr(download.urllib.request, "urlopen", _urlopen)
        return calls

    return install


def test_datasets_lists_registry_keys():
    assert data_utils.datasets() == list(data_utils.DATASETS.keys())
    assert "random" in data_utils.datasets()


def test_load_dataset_downloads_then_reads_from_cache(tmp_path, fake_get):
    calls = fake_get()
    (df,) = data_utils.load_dataset("random", output_path=str(tmp_path))
    pd.testing.assert_frame_equal(df, pd.DataFrame({"a": [1, 3], "b": [2, 4]}))
    assert os.path.isfile(tmp_path / "random" / "dataset.csv")
    assert len(calls) == 1

    data_utils.load_dataset("random", output_path=str(tmp_path))
    assert len(calls) == 1  # served from disk

    data_utils.load_dataset("random", cache=False, output_path=str(tmp_path))
    assert len(calls) == 2  # cache bypassed


def test_load_dataset_one_frame_per_file_and_forwards_kwargs(tmp_path, fake_get):
    fake_get(content_length=None)  # missing content-length → size check skipped
    frames = data_utils.load_dataset("trivago_rio", output_path=str(tmp_path), usecols=["a"])
    assert len(frames) == 2
    assert all(list(f.columns) == ["a"] for f in frames)


def test_load_dataset_raises_on_truncated_download(tmp_path, fake_get):
    fake_get(content_length=len(CSV) + 10)
    with pytest.raises(ConnectionError):
        data_utils.load_dataset("random", output_path=str(tmp_path))


def test_download_task_outputs_and_run(tmp_path, fake_get):
    fake_get()
    task = data_utils.DownloadDataset(output_path=str(tmp_path), dataset="trivago_rio")
    paths = [t.path for t in task.output()]
    assert paths == [
        os.path.join(str(tmp_path), "trivago_rio", "train.csv"),
        os.path.join(str(tmp_path), "trivago_rio", "item_metadata.csv"),
    ]
    assert not task.complete()
    task.run()
    assert task.complete()
