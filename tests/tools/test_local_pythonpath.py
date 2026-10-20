"""Regression tests for Hermes-owned child PYTHONPATH filtering."""

import os
from pathlib import Path

import pytest

from tools.environments import local
from tools.environments import local_pythonpath


def test_strips_site_packages_selected_before_a_new_generation(monkeypatch, tmp_path):
    installs = tmp_path / "installs"
    monkeypatch.setattr(local, "_startup_dependency_installs_root", installs)
    old_generation = installs / "id" / "environments" / "old" / "venv" / "lib" / "python3.14" / "site-packages"
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


@pytest.mark.skipif(os.name == "nt", reason="Symlinks require elevated privileges on Windows")
def test_capture_does_not_claim_store_lookalikes_or_escaping_links(tmp_path):
    installs = tmp_path / "installs"
    installs.mkdir()
    user = tmp_path / "installs-other" / "site-packages"
    user.mkdir(parents=True)
    alias = installs / "site-packages"
    alias.symlink_to(user, target_is_directory=True)
    assert local_pythonpath._capture_startup_site_packages(str(user), installs) == ()
    assert local_pythonpath._capture_startup_site_packages(str(alias), installs) == ()
