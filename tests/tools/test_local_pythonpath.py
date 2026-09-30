"""Regression tests for Hermes-owned child PYTHONPATH filtering."""

from pathlib import Path

from tools.environments import local
from tools.environments import local_pythonpath


def test_strips_site_packages_selected_before_a_new_generation(monkeypatch, tmp_path):
    old_generation = tmp_path / "old" / "venv" / "lib" / "python3.14" / "site-packages"
    user_path = tmp_path / "user"
    old_generation.parent.mkdir(parents=True)
    user_path.mkdir()

    monkeypatch.setattr(local, "_hermes_site_packages", [])
    monkeypatch.setattr(local, "_startup_pythonpath_site_packages", (old_generation,))
    monkeypatch.setattr(local_pythonpath, "_validated_runtime_venv", lambda env: None)
    monkeypatch.setattr(local, "_hermes_repo_root_aliases", ())

    env = {"PYTHONPATH": ":".join((str(old_generation), str(user_path)))}
    local_pythonpath._strip_hermes_owned_pythonpath(env)

    assert env["PYTHONPATH"] == str(user_path)
