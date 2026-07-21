"""Per-skill experience memory and lifecycle integration tests."""

from __future__ import annotations

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from tools.skill_manager_tool import _create_skill, skill_manage
from tools.skills_tool import skill_view


VALID_SKILL = """\
---
name: test-skill
description: A test skill.
---

# Test Skill

Follow the procedure.
"""


def _append_in_process(item: tuple[str, int]) -> None:
    from tools.skill_experience import append_skill_experience

    skill_dir, index = item
    append_skill_experience(Path(skill_dir), f"process-experience-{index}")


@contextmanager
def isolated_skills(tmp_path: Path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    with (
        patch("tools.skill_manager_tool.SKILLS_DIR", skills_dir),
        patch("tools.skills_tool.SKILLS_DIR", skills_dir),
        patch("agent.skill_utils.get_all_skills_dirs", return_value=[skills_dir]),
        patch("agent.skill_utils.get_external_skills_dirs", return_value=[]),
    ):
        yield skills_dir


def test_remember_appends_timestamped_experience_and_skill_view_surfaces_it(tmp_path):
    with isolated_skills(tmp_path) as skills_dir:
        assert _create_skill("test-skill", VALID_SKILL)["success"]

        with patch(
            "tools.skill_experience._utc_now",
            return_value=datetime(2026, 7, 21, 12, 30, tzinfo=timezone.utc),
        ):
            result = json.loads(
                skill_manage(
                    action="remember",
                    name="test-skill",
                    experience="PDFs over 100 MB need chunked extraction.",
                )
            )

        viewed = json.loads(skill_view("test-skill"))

    assert result["success"] is True
    assert result["memory_file"].endswith(".memory.md")
    assert "2026-07-21 12:30:00 UTC" in viewed["skill_memory"]
    assert "PDFs over 100 MB need chunked extraction." in viewed["skill_memory"]
    assert viewed["content"] == VALID_SKILL
    assert (skills_dir / "test-skill" / ".memory.md").exists()


def test_remember_rejects_empty_or_oversized_experience(tmp_path):
    with isolated_skills(tmp_path):
        assert _create_skill("test-skill", VALID_SKILL)["success"]
        empty = json.loads(
            skill_manage(action="remember", name="test-skill", experience="  ")
        )
        large = json.loads(
            skill_manage(action="remember", name="test-skill", experience="x" * 8193)
        )

    assert empty["success"] is False
    assert "non-empty" in empty["error"]
    assert large["success"] is False
    assert "8,192" in large["error"]


def test_remember_requires_existing_directory_skill(tmp_path):
    with isolated_skills(tmp_path):
        result = json.loads(
            skill_manage(action="remember", name="missing", experience="lesson")
        )

    assert result["success"] is False
    assert "not found" in result["error"]


def test_concurrent_experience_appends_preserve_every_entry(tmp_path):
    with isolated_skills(tmp_path):
        assert _create_skill("test-skill", VALID_SKILL)["success"]

        def remember(index: int) -> bool:
            result = json.loads(
                skill_manage(
                    action="remember",
                    name="test-skill",
                    experience=f"experience-{index}",
                )
            )
            return result["success"]

        with ThreadPoolExecutor(max_workers=12) as executor:
            assert all(executor.map(remember, range(50)))

        viewed = json.loads(skill_view("test-skill"))

    memory = viewed["skill_memory"]
    for index in range(50):
        assert f"experience-{index}\n" in memory


def test_cross_process_experience_appends_preserve_every_entry(tmp_path):
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    work = [(str(skill_dir), index) for index in range(30)]

    with ProcessPoolExecutor(
        max_workers=6, mp_context=mp.get_context("spawn")
    ) as executor:
        list(executor.map(_append_in_process, work))

    memory = (skill_dir / ".memory.md").read_text(encoding="utf-8")
    for index in range(30):
        assert f"process-experience-{index}\n" in memory


def test_short_append_write_rolls_back_partial_entry(tmp_path, monkeypatch):
    with isolated_skills(tmp_path) as skills_dir:
        assert _create_skill("test-skill", VALID_SKILL)["success"]
        import tools.skill_sidecar_io as sidecar_io

        real_write = sidecar_io.os.write
        writes = 0

        def short_then_fail(fd, data):
            nonlocal writes
            if b"## " in bytes(data) or writes:
                writes += 1
                if writes == 1:
                    return real_write(fd, bytes(data)[:5])
                raise OSError("simulated disk failure")
            return real_write(fd, data)

        monkeypatch.setattr(sidecar_io.os, "write", short_then_fail)
        result = json.loads(
            skill_manage(action="remember", name="test-skill", experience="lesson")
        )
        memory = skills_dir / "test-skill" / ".memory.md"

    assert result["success"] is False
    assert not memory.exists() or memory.read_bytes() == b""


def test_directory_fsync_storage_error_is_propagated(tmp_path, monkeypatch):
    import errno
    import stat

    with isolated_skills(tmp_path):
        assert _create_skill("test-skill", VALID_SKILL)["success"]
        import tools.skill_sidecar_io as sidecar_io

        real_fsync = sidecar_io.os.fsync

        def failing_directory_fsync(fd):
            if stat.S_ISDIR(sidecar_io.os.fstat(fd).st_mode):
                raise OSError(errno.EIO, "simulated directory fsync failure")
            return real_fsync(fd)

        monkeypatch.setattr(sidecar_io.os, "fsync", failing_directory_fsync)
        result = json.loads(
            skill_manage(action="remember", name="test-skill", experience="lesson")
        )

    assert result["success"] is False
    assert "fsync failure" in result["error"]


def test_skill_memory_is_bounded_on_read_without_corrupting_file(tmp_path):
    with isolated_skills(tmp_path) as skills_dir:
        assert _create_skill("test-skill", VALID_SKILL)["success"]
        memory_path = skills_dir / "test-skill" / ".memory.md"
        original = "old\n" + ("x" * 40_000) + "\nrecent lesson\n"
        memory_path.write_text(original, encoding="utf-8")

        viewed = json.loads(skill_view("test-skill"))

    assert viewed["skill_memory_truncated"] is True
    assert viewed["skill_memory"].startswith("[Earlier skill experience omitted")
    assert viewed["skill_memory"].endswith("recent lesson\n")
    assert memory_path.read_text(encoding="utf-8") == original


def test_memory_sidecar_symlink_cannot_escape_skill_directory(tmp_path):
    with isolated_skills(tmp_path) as skills_dir:
        assert _create_skill("test-skill", VALID_SKILL)["success"]
        outside = tmp_path / "outside.md"
        outside.write_text("untouched", encoding="utf-8")
        sidecar = skills_dir / "test-skill" / ".memory.md"
        try:
            sidecar.symlink_to(outside)
        except OSError:
            return

        result = json.loads(
            skill_manage(action="remember", name="test-skill", experience="escape")
        )

    assert result["success"] is False
    assert "symlink" in result["error"].lower()
    assert outside.read_text(encoding="utf-8") == "untouched"


def test_skill_directory_symlink_swap_cannot_redirect_memory_write(
    tmp_path, monkeypatch
):
    with isolated_skills(tmp_path) as skills_dir:
        assert _create_skill("test-skill", VALID_SKILL)["success"]
        skill_dir = skills_dir / "test-skill"
        original_dir = skills_dir / "original"
        outside = tmp_path / "outside"
        outside.mkdir()

        import tools.skill_sidecar_io as sidecar_io

        real_open = sidecar_io.os.open
        swapped = False

        def swapping_open(path, flags, *args, **kwargs):
            nonlocal swapped
            if not swapped and Path(path) == skill_dir:
                swapped = True
                skill_dir.rename(original_dir)
                skill_dir.symlink_to(outside, target_is_directory=True)
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(sidecar_io.os, "open", swapping_open)
        result = json.loads(
            skill_manage(action="remember", name="test-skill", experience="escape")
        )

    assert result["success"] is False
    assert not (outside / ".memory.md").exists()


def test_remember_redacts_secrets_before_persistence(tmp_path):
    secrets = [
        "MY_API_TOKEN=abcdefghijklmnopqrstuv",
        "password: hunter2",
        "https://example.test/?access_token=opaquevalue123456789",
    ]
    with isolated_skills(tmp_path) as skills_dir:
        assert _create_skill("test-skill", VALID_SKILL)["success"]
        result = json.loads(
            skill_manage(
                action="remember",
                name="test-skill",
                experience="\n".join(secrets),
            )
        )
        persisted = (skills_dir / "test-skill" / ".memory.md").read_text(
            encoding="utf-8"
        )

    assert result["success"] is True
    assert all(secret not in persisted for secret in secrets)
    assert "REDACTED" in persisted


def test_no_dir_fd_platform_fails_closed(tmp_path, monkeypatch):
    with isolated_skills(tmp_path) as skills_dir:
        assert _create_skill("test-skill", VALID_SKILL)["success"]
        import tools.skill_sidecar_io as sidecar_io

        monkeypatch.setattr(sidecar_io, "_DIR_FD_SUPPORTED", False)
        remembered = json.loads(
            skill_manage(action="remember", name="test-skill", experience="fallback")
        )
        tested = json.loads(
            skill_manage(
                action="write_file",
                name="test-skill",
                file_path="tests/test_behavior.py",
                file_content="def test_behavior():\n    assert True\n",
            )
        )
        validated = json.loads(skill_manage(action="validate", name="test-skill"))
        legacy_tests = skills_dir / "test-skill" / "tests"
        legacy_tests.mkdir(exist_ok=True)
        (legacy_tests / "test_legacy.py").write_text(
            "def test_ok():\n    assert True\n"
        )
        from tools.skill_validation import validation_allows_discovery

        discoverable = validation_allows_discovery(skills_dir / "test-skill")

    assert remembered["success"] is False
    assert tested["success"] is False
    assert validated["success"] is False
    assert discoverable is False
    assert not (skills_dir / "test-skill" / "tests" / "test_behavior.py").exists()
    assert not (skills_dir / "test-skill" / ".memory.md").exists()
    assert not (skills_dir / "test-skill" / ".validation.json").exists()


def test_skill_manage_schema_exposes_remember_action():
    from tools.skill_manager_tool import SKILL_MANAGE_SCHEMA

    properties = SKILL_MANAGE_SCHEMA["parameters"]["properties"]
    assert "remember" in properties["action"]["enum"]
    assert properties["experience"]["type"] == "string"
