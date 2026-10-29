import pytest

from agis import config
from agis.setup import Hardware, is_installed, recommend_model


@pytest.mark.parametrize("memory,apple,expected", [
    (8, True, "qwen2.5-coder:3b"),
    (16, True, "qwen2.5-coder:7b"),
    (36, True, "qwen2.5-coder:14b"),
    (64, False, "qwen2.5-coder:3b"),
])
def test_recommend_model(memory, apple, expected):
    assert recommend_model(Hardware("chip", memory, apple)).name == expected


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
