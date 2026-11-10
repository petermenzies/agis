import pytest

from agis.llm import model_ready


@pytest.mark.parametrize("models,wanted,ok", [
    ([], "qwen2.5:7b", False),
    ([{"model": "qwen2.5:7b", "context_length": 2048}], "qwen2.5:7b", False),
    ([{"model": "qwen2.5:7b", "context_length": 8192}], "qwen2.5:7b", True),
    ([{"model": "qwen2.5:7b", "context_length": 16384}], "qwen2.5:7b", True),
    ([{"model": "qwen2.5:7b"}], "qwen2.5:7b", True),
    ([{"model": "other:7b", "context_length": 8192}], "qwen2.5:7b", False),
    ([{"name": "qwen2.5:latest"}], "qwen2.5", True),
])
def test_model_ready(models, wanted, ok):
    assert model_ready(models, wanted, 8192) is ok
