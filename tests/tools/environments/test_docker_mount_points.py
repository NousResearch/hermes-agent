"""Regression for #141911: bind-mount points nested inside a sandbox bind are created as the user
BEFORE docker run. dockerd auto-creates missing mount points inside a bind as root, and those
root-owned dirs under sandboxes/docker/*/home made Docker-sandboxed profiles un-deletable
(rmtree: Permission denied on home/.hermes/*)."""

from __future__ import annotations

import os

from tools.environments.docker import _precreate_nested_mount_points


def test_nested_mount_points_are_user_owned_dirs_and_files(tmp_path):
    home = tmp_path / "sandbox" / "home"
    home.mkdir(parents=True)
    src_dir = tmp_path / "profile" / "skills"
    src_dir.mkdir(parents=True)
    src_file = tmp_path / "profile" / ".env"
    src_file.write_text("X=1\n")
    _precreate_nested_mount_points(
        [(str(home), "/root")],
        ["-v", f"{src_dir}:/root/.hermes/skills:ro",
         "-v", f"{src_file}:/root/.hermes/.env:ro",
         "-v", f"{tmp_path / 'elsewhere'}:/data:ro"])
    skills_point = home / ".hermes" / "skills"
    env_point = home / ".hermes" / ".env"
    assert skills_point.is_dir() and env_point.is_file()
    for point in (home / ".hermes", skills_point, env_point):
        assert point.stat().st_uid == os.getuid(), "mount points must never be dockerd's root-owned dirs"
    assert not (home / "data").exists(), "binds outside the sandbox binds are not mount points to create"
    assert not (tmp_path / "elsewhere").exists(), "mount SOURCES are their owners' business, not this"


def test_binds_at_the_sandbox_root_are_left_alone(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    _precreate_nested_mount_points([(str(home), "/root")],
                                   ["-v", f"{tmp_path / 'other'}:/root:ro"])
    assert list(home.iterdir()) == []
