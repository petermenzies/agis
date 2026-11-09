import pytest

from agis import config
from agis.setup import Hardware, is_installed, recommend_model


@pytest.mark.parametrize("memory,apple", [(8, True), (16, True), (36, True), (64, False)])
def test_recommend_model(memory, apple):
    choice = recommend_model(Hardware("chip", memory, apple))
    assert choice.name == "qwen2.5:7b"
    assert choice.download_gb == 4.7


def test_is_installed_handles_latest_tag():
    assert is_installed("qwen2.5-coder", ["qwen2.5-coder:latest"])
    assert not is_installed("qwen2.5-coder:7b", ["qwen2.5-coder:latest"])


def test_config_roundtrip_and_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("AGIS_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert not config.load().from_file
    config.save("qwen2.5-coder:3b", "http://localhost:11434")
    loaded = config.load()
    assert (loaded.model, loaded.host, loaded.from_file) == (
        "qwen2.5-coder:3b", "http://localhost:11434", True)
    monkeypatch.setenv("AGIS_MODEL", "other")
    assert config.load().model == "other"
