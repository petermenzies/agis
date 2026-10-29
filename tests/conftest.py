import pytest

from agis.sample_data import write_sample_data


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return write_sample_data(tmp_path)
