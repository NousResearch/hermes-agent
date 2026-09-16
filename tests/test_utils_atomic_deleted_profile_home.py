"""Atomic writers must never recreate a deleted named profile home."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from hermes_constants import mark_named_profile_deleted
from utils import (
    atomic_json_write,
    atomic_roundtrip_yaml_save,
    atomic_roundtrip_yaml_update,
    atomic_write_bytes,
    atomic_write_text,
    atomic_yaml_write,
)


@pytest.fixture
def named_profile_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    home = tmp_path / ".hermes"
    profiles = home / "profiles"
    profile = profiles / "worker"
    profiles.mkdir(parents=True)
    return home, profiles, profile


def _tombstone_and_remove(profile: Path) -> None:
    profile.mkdir(parents=True)
    mark_named_profile_deleted(profile)
    shutil.rmtree(profile)


@pytest.mark.parametrize(
    "writer",
    [
        lambda profile: atomic_json_write(
            profile / "cache" / "x.json", {"a": 1}
        ),
        lambda profile: atomic_write_text(
            profile / "cache" / "x.txt", "x"
        ),
        lambda profile: atomic_write_bytes(
            profile / "cache" / "x.bin", b"x"
        ),
        lambda profile: atomic_yaml_write(
            profile / "config.yaml", {"a": 1}
        ),
        lambda profile: atomic_roundtrip_yaml_update(
            profile / "config.yaml", "model.default", "x"
        ),
        lambda profile: atomic_roundtrip_yaml_save(
            profile / "config.yaml", {"a": 1}
        ),
    ],
    ids=[
        "json",
        "text",
        "bytes",
        "yaml",
        "roundtrip-update",
        "roundtrip-save",
    ],
)
def test_atomic_writer_does_not_recreate_deleted_profile_home(
    named_profile_layout,
    writer,
):
    _home, _profiles, profile = named_profile_layout
    _tombstone_and_remove(profile)

    with pytest.raises(
        FileNotFoundError,
        match="Named profile home does not exist",
    ):
        writer(profile)

    assert not profile.exists()


def test_atomic_writer_refuses_never_created_named_profile(
    named_profile_layout,
):
    _home, profiles, _profile = named_profile_layout
    never_created = profiles / "never"

    with pytest.raises(
        FileNotFoundError,
        match="Named profile home does not exist",
    ):
        atomic_json_write(
            never_created / "cache" / "x.json",
            {},
        )

    assert not never_created.exists()


def test_atomic_writer_creates_missing_parents_for_ordinary_path(
    tmp_path: Path,
):
    target = tmp_path / "ordinary" / "nested" / "x.json"

    atomic_json_write(target, {"a": 1})

    assert target.is_file()
    assert target.parent.is_dir()


def test_atomic_writer_writes_under_default_home(
    named_profile_layout,
):
    home, _profiles, _profile = named_profile_layout
    target = home / "cache" / "x.json"

    atomic_json_write(target, {})

    assert target.is_file()


def test_atomic_writer_writes_under_hidden_profile_staging_name(
    named_profile_layout,
):
    _home, profiles, _profile = named_profile_layout
    staging = profiles / ".building-worker"
    target = staging / "cache" / "x.json"

    atomic_json_write(target, {})

    assert target.is_file()


def test_atomic_writer_writes_under_live_named_profile(
    named_profile_layout,
):
    _home, _profiles, profile = named_profile_layout
    profile.mkdir()
    target = profile / "cache" / "x.json"

    atomic_json_write(target, {})

    assert target.is_file()


def test_atomic_writer_refuses_tombstoned_existing_profile_home(
    named_profile_layout,
):
    _home, _profiles, profile = named_profile_layout
    profile.mkdir()
    mark_named_profile_deleted(profile)

    with pytest.raises(
        FileNotFoundError,
        match="Named profile home does not exist",
    ):
        atomic_json_write(profile / "cache" / "x.json", {})

    assert profile.is_dir()
    assert not any(profile.iterdir())
