import os
from pathlib import Path

from finchat.env import load_env_file


def test_load_env_file_sets_missing_values_only(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        'OPENAI_API_KEY="test-key"\n'
        "OPENAI_MODEL=gpt-4o-mini\n"
        "# comment\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "existing-model")

    loaded = load_env_file(env_path)

    assert loaded["OPENAI_API_KEY"] == "test-key"
    assert loaded["OPENAI_MODEL"] == "gpt-4o-mini"
    assert os.environ["OPENAI_API_KEY"] == "test-key"
    assert os.environ["OPENAI_MODEL"] == "existing-model"


def test_load_env_file_ignores_missing_file(tmp_path):
    loaded = load_env_file(Path(tmp_path / ".env"))

    assert loaded == {}
