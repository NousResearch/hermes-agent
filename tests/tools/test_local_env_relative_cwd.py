"""Regression tests for local terminal initial cwd normalization."""

from pathlib import Path

from tools.environments.local import LocalEnvironment, _resolve_local_initial_cwd


def test_relative_initial_cwd_resolves_from_parent(tmp_path, monkeypatch):
    project = tmp_path / "hermes-agent"
    project.mkdir()
    monkeypatch.chdir(tmp_path)

    assert _resolve_local_initial_cwd("hermes-agent") == str(project)


def test_relative_initial_cwd_matching_current_dir_uses_current_dir(tmp_path, monkeypatch):
    project = tmp_path / "hermes-agent"
    project.mkdir()
    monkeypatch.chdir(project)

    assert _resolve_local_initial_cwd("hermes-agent") == str(project)


def test_local_environment_does_not_cd_into_nested_matching_relative_cwd(tmp_path, monkeypatch):
    project = tmp_path / "hermes-agent"
    project.mkdir()
    monkeypatch.chdir(project)

    env = LocalEnvironment(cwd="hermes-agent", timeout=5)
    try:
        result = env.execute("pwd", timeout=5)
    finally:
        env.cleanup()

    assert result["returncode"] == 0
    assert result["output"].strip() == str(project)
    assert "cd: hermes-agent" not in result["output"]


def test_local_environment_keeps_existing_relative_child_cwd(tmp_path, monkeypatch):
    project = tmp_path / "hermes-agent"
    project.mkdir()
    monkeypatch.chdir(tmp_path)

    env = LocalEnvironment(cwd="hermes-agent", timeout=5)
    try:
        result = env.execute("pwd", timeout=5)
    finally:
        env.cleanup()

    assert result["returncode"] == 0
    assert result["output"].strip() == str(project)
