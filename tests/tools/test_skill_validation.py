"""Skill package validation and refinement-signal tests."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from tools.skill_manager_tool import _create_skill, skill_manage


VALID_SKILL = """\
---
name: validated-skill
description: A code-backed skill with tests.
---

# Validated Skill

Run `scripts/add.py` and verify the result.
"""

PASSING_SCRIPT = "def add(left, right):\n    return left + right\n"
PASSING_TEST = """\
from scripts.add import add


def test_add():
    assert add(2, 3) == 5
"""
FAILING_TEST = """\
from scripts.add import add


def test_add():
    assert add(2, 3) == 6
"""


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


def create_code_skill(test_source: str, tmp_path: Path) -> Path:
    assert _create_skill("validated-skill", VALID_SKILL)["success"]
    script = json.loads(
        skill_manage(
            action="write_file",
            name="validated-skill",
            file_path="scripts/add.py",
            file_content=PASSING_SCRIPT,
        )
    )
    test = json.loads(
        skill_manage(
            action="write_file",
            name="validated-skill",
            file_path="tests/test_add.py",
            file_content=test_source,
        )
    )
    assert script["success"] and test["success"]
    return tmp_path / "skills" / "validated-skill"


def evidence(skill_dir: Path, exit_code: int, output: str) -> dict:
    from tools.skill_validation import skill_content_digest

    pending = json.loads((skill_dir / ".validation.json").read_text(encoding="utf-8"))
    return {
        "content_digest": skill_content_digest(skill_dir),
        "validation_token": pending["validation_token"],
        "command": "python -m pytest -q tests",
        "exit_code": exit_code,
        "output": output,
    }


def _record_in_process(item: tuple[str, dict]) -> dict:
    from tools.skill_validation import record_skill_validation

    skill_dir, submitted = item
    return record_skill_validation(Path(skill_dir), submitted)


def test_validate_records_digest_bound_evidence_and_rejects_token_replay(tmp_path):
    with isolated_skills(tmp_path):
        skill_dir = create_code_skill(PASSING_TEST, tmp_path)
        submitted = evidence(skill_dir, 0, "1 passed")
        result = json.loads(
            skill_manage(
                action="validate",
                name="validated-skill",
                validation=submitted,
            )
        )
        replay = json.loads(
            skill_manage(
                action="validate",
                name="validated-skill",
                validation=submitted,
            )
        )

    record = json.loads((skill_dir / ".validation.json").read_text(encoding="utf-8"))
    assert result["success"] is True
    assert result["validation_status"] == "passed"
    assert result["tests_collected"] == 1
    assert record["status"] == "passed"
    assert record["content_digest"] == result["content_digest"]
    assert replay["success"] is False
    assert "validation_token" in replay["error"]


def test_content_digest_binds_executable_mode_bits(tmp_path):
    from tools.skill_validation import skill_content_digest

    with isolated_skills(tmp_path):
        skill_dir = create_code_skill(PASSING_TEST, tmp_path)
        script = skill_dir / "scripts" / "add.py"
        script.chmod(0o644)
        non_executable = skill_content_digest(skill_dir)
        script.chmod(0o755)
        executable = skill_content_digest(skill_dir)

    assert executable != non_executable


def test_created_code_skill_is_hidden_until_validated(tmp_path):
    from tools.skill_validation import read_skill_validation, validation_allows_discovery

    with isolated_skills(tmp_path):
        skill_dir = create_code_skill(PASSING_TEST, tmp_path)
        # A freshly created, still-untested package must never be discoverable,
        # even for the brief window before the pending sidecar exists.
        record = read_skill_validation(skill_dir)
        assert record is not None
        assert record["status"] in {"draft", "pending"}
        assert validation_allows_discovery(skill_dir) is False


def test_background_skill_draft_is_stamped_before_atomic_publication(
    tmp_path, monkeypatch
):
    import tools.skill_validation as validation
    from tools.skill_provenance import (
        reset_current_write_origin,
        set_current_write_origin,
    )

    with isolated_skills(tmp_path) as skills_dir:
        final_dir = skills_dir / "validated-skill"
        observed = {}
        real_record = validation.record_draft_validation

        def record_before_publish(draft_dir):
            draft_dir = Path(draft_dir)
            observed["draft_dir"] = draft_dir
            observed["final_visible"] = final_dir.exists()
            assert draft_dir.name.startswith(".validated-skill.draft-")
            real_record(draft_dir)

        monkeypatch.setattr(validation, "record_draft_validation", record_before_publish)
        token = set_current_write_origin("background_review")
        try:
            created = _create_skill("validated-skill", VALID_SKILL)
        finally:
            reset_current_write_origin(token)

        assert created["success"] is True
        assert observed["final_visible"] is False
        assert final_dir.is_dir()
        assert not observed["draft_dir"].exists()
        assert validation.read_skill_validation(final_dir)["status"] == "draft"


def test_foreground_create_is_serialized_with_background_draft_publish(
    tmp_path, monkeypatch
):
    import threading
    import time

    import tools.skill_validation as validation
    from tools.skill_provenance import (
        reset_current_write_origin,
        set_current_write_origin,
    )

    with isolated_skills(tmp_path):
        entered = threading.Event()
        release = threading.Event()
        real_record = validation.record_draft_validation
        calls = 0

        def blocking_record(draft_dir):
            nonlocal calls
            real_record(Path(draft_dir))
            calls += 1
            if calls == 1:
                entered.set()
                assert release.wait(5)

        monkeypatch.setattr(validation, "record_draft_validation", blocking_record)
        results = {}

        def background_create():
            token = set_current_write_origin("background_review")
            try:
                results["background"] = _create_skill(
                    "validated-skill", VALID_SKILL, category="alpha"
                )
            finally:
                reset_current_write_origin(token)

        def foreground_create():
            results["foreground"] = _create_skill(
                "validated-skill", VALID_SKILL, category="beta"
            )

        background = threading.Thread(target=background_create)
        foreground = threading.Thread(target=foreground_create)
        background.start()
        assert entered.wait(5)
        foreground.start()
        time.sleep(0.1)
        assert foreground.is_alive(), "foreground create bypassed parent create lock"
        release.set()
        background.join(5)
        foreground.join(5)

        assert results["background"]["success"] is True
        assert results["foreground"]["success"] is False


def test_draft_survives_intermediate_script_write(tmp_path):
    from tools.skill_validation import (
        read_skill_validation,
        record_draft_validation,
        validation_allows_discovery,
    )

    with isolated_skills(tmp_path):
        assert _create_skill("validated-skill", VALID_SKILL)["success"]
        skill_dir = tmp_path / "skills" / "validated-skill"
        record_draft_validation(skill_dir)

        # Adding a non-test file must not promote the draft to a discoverable
        # static skill.
        assert skill_manage(
            action="write_file",
            name="validated-skill",
            file_path="scripts/add.py",
            file_content=PASSING_SCRIPT,
        )
        record = read_skill_validation(skill_dir)
        assert record["status"] == "draft"
        assert validation_allows_discovery(skill_dir) is False



def test_tested_package_without_validation_sidecar_is_hidden(tmp_path):
    from tools.skill_validation import validation_allows_discovery

    with isolated_skills(tmp_path):
        assert _create_skill("validated-skill", VALID_SKILL)["success"]
        skill_dir = tmp_path / "skills" / "validated-skill"
        tests_dir = skill_dir / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_add.py").write_text(PASSING_TEST, encoding="utf-8")

        assert not (skill_dir / ".validation.json").exists()
        assert validation_allows_discovery(skill_dir) is False


def test_validation_token_is_consumed_atomically_across_threads(tmp_path):
    with isolated_skills(tmp_path):
        skill_dir = create_code_skill(PASSING_TEST, tmp_path)
        submitted = evidence(skill_dir, 0, "1 passed")
        barrier = threading.Barrier(2)

        def submit_once():
            barrier.wait()
            return json.loads(
                skill_manage(
                    action="validate",
                    name="validated-skill",
                    validation=submitted,
                )
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: submit_once(), range(2)))

    assert sum(result["success"] is True for result in results) == 1
    assert sum("validation_token" in result.get("error", "") for result in results) == 1


def test_validation_token_is_consumed_atomically_across_processes(tmp_path):
    with isolated_skills(tmp_path):
        skill_dir = create_code_skill(PASSING_TEST, tmp_path)
        submitted = evidence(skill_dir, 0, "1 passed")
        context = mp.get_context("spawn")
        work = [(str(skill_dir), submitted)] * 2
        with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
            results = list(pool.map(_record_in_process, work))

    assert sum(result["success"] is True for result in results) == 1
    assert sum("validation_token" in result.get("error", "") for result in results) == 1


def test_validate_tested_skill_without_evidence_returns_digest(tmp_path):
    with isolated_skills(tmp_path):
        skill_dir = create_code_skill(PASSING_TEST, tmp_path)
        result = json.loads(skill_manage(action="validate", name="validated-skill"))

    assert result["success"] is False
    assert result["validation_status"] == "pending"
    assert len(result["validation_token"]) >= 24
    assert result["content_digest"] == evidence(skill_dir, 0, "")["content_digest"]


def test_validate_failure_emits_refinement_signal_and_records_failure(tmp_path):
    with isolated_skills(tmp_path):
        skill_dir = create_code_skill(FAILING_TEST, tmp_path)
        result = json.loads(
            skill_manage(
                action="validate",
                name="validated-skill",
                validation=evidence(skill_dir, 1, "1 failed"),
            )
        )

    record = json.loads((skill_dir / ".validation.json").read_text(encoding="utf-8"))
    assert result["success"] is False
    assert result["validation_status"] == "failed"
    assert result["refinement_required"] is True
    assert "1 failed" in result["test_output"]
    assert record["status"] == "failed"


def test_validate_text_only_skill_records_static_validation(tmp_path):
    with isolated_skills(tmp_path) as skills_dir:
        assert _create_skill("validated-skill", VALID_SKILL)["success"]
        result = json.loads(skill_manage(action="validate", name="validated-skill"))

    record = json.loads(
        (skills_dir / "validated-skill" / ".validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["success"] is True
    assert result["validation_status"] == "static"
    assert result["tests_collected"] == 0
    assert record["status"] == "static"


def test_mutating_executable_skill_content_invalidates_prior_validation(tmp_path):
    with isolated_skills(tmp_path):
        skill_dir = create_code_skill(PASSING_TEST, tmp_path)
        assert json.loads(
            skill_manage(
                action="validate",
                name="validated-skill",
                validation=evidence(skill_dir, 0, "passed"),
            )
        )["success"]

        changed = json.loads(
            skill_manage(
                action="write_file",
                name="validated-skill",
                file_path="scripts/add.py",
                file_content="def add(left, right):\n    return left - right\n",
            )
        )

    assert changed["success"] is True
    record = json.loads((skill_dir / ".validation.json").read_text(encoding="utf-8"))
    assert record["status"] == "pending"
    assert record["reason"] == "skill package changed"


def test_validate_rechecks_skill_frontmatter_before_recording(tmp_path):
    with isolated_skills(tmp_path) as skills_dir:
        assert _create_skill("validated-skill", VALID_SKILL)["success"]
        (skills_dir / "validated-skill" / "SKILL.md").write_text(
            "# invalid skill\n", encoding="utf-8"
        )
        result = json.loads(skill_manage(action="validate", name="validated-skill"))

    assert result["success"] is False
    assert "frontmatter" in result["error"].lower()
    assert not (skills_dir / "validated-skill" / ".validation.json").exists()


def test_tested_skill_is_hidden_from_catalog_until_validation_passes(tmp_path):
    from tools.skills_tool import skills_list

    with isolated_skills(tmp_path):
        skill_dir = create_code_skill(PASSING_TEST, tmp_path)
        pending = json.loads(skills_list())

        assert json.loads(
            skill_manage(
                action="validate",
                name="validated-skill",
                validation=evidence(skill_dir, 0, "passed"),
            )
        )["success"]
        validated = json.loads(skills_list())

    assert "validated-skill" not in {item["name"] for item in pending["skills"]}
    assert "validated-skill" in {item["name"] for item in validated["skills"]}


def test_text_only_static_validation_remains_discoverable_after_edit(tmp_path):
    from tools.skills_tool import skills_list

    with isolated_skills(tmp_path):
        assert _create_skill("validated-skill", VALID_SKILL)["success"]
        assert json.loads(skill_manage(action="validate", name="validated-skill"))[
            "success"
        ]
        edited = VALID_SKILL.replace(
            "Follow the procedure.", "Follow the revised procedure."
        )
        assert json.loads(
            skill_manage(action="edit", name="validated-skill", content=edited)
        )["success"]

        listed = json.loads(skills_list())

    assert "validated-skill" in {item["name"] for item in listed["skills"]}


def test_validation_rejects_evidence_for_an_older_package_digest(tmp_path):
    with isolated_skills(tmp_path):
        skill_dir = create_code_skill(PASSING_TEST, tmp_path)
        stale = evidence(skill_dir, 0, "passed")
        assert json.loads(
            skill_manage(
                action="write_file",
                name="validated-skill",
                file_path="scripts/add.py",
                file_content="def add(left, right):\n    return left - right\n",
            )
        )["success"]

        result = json.loads(
            skill_manage(
                action="validate",
                name="validated-skill",
                validation=stale,
            )
        )

    assert result["success"] is False
    assert result["validation_status"] == "pending"
    assert "stale" in result["error"]
    assert result["content_digest"] != stale["content_digest"]


def test_validation_digest_ignores_test_runner_artifacts(tmp_path):
    from tools.skill_validation import skill_content_digest

    with isolated_skills(tmp_path):
        skill_dir = create_code_skill(PASSING_TEST, tmp_path)
        before = skill_content_digest(skill_dir)
        env = os.environ.copy()
        env.pop("PYTHONDONTWRITEBYTECODE", None)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests"],
            cwd=skill_dir,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        after = skill_content_digest(skill_dir)

    assert result.returncode == 0, result.stdout
    assert (skill_dir / ".pytest_cache").exists()
    assert list(skill_dir.rglob("*.pyc"))
    assert after == before


def test_validation_digest_covers_nested_sidecar_names_and_rejects_symlinks(tmp_path):
    from tools.skill_validation import skill_content_digest

    with isolated_skills(tmp_path):
        skill_dir = create_code_skill(PASSING_TEST, tmp_path)
        nested = skill_dir / "tests" / ".memory.md"
        nested.write_text("first", encoding="utf-8")
        first = skill_content_digest(skill_dir)
        nested.write_text("second", encoding="utf-8")
        second = skill_content_digest(skill_dir)
        assert first != second

        outside = tmp_path / "outside.py"
        outside.write_text("print('outside')", encoding="utf-8")
        link = skill_dir / "scripts" / "linked.py"
        try:
            link.symlink_to(outside)
        except OSError:
            return

        result = json.loads(skill_manage(action="validate", name="validated-skill"))

    assert result["success"] is False
    assert "symlink" in result["error"]


def test_validation_gates_system_prompt_and_invalidates_cached_pass(tmp_path):
    import importlib

    prompt_builder = importlib.import_module("agent.prompt_builder")

    with isolated_skills(tmp_path) as skills_dir:
        skill_dir = create_code_skill(PASSING_TEST, tmp_path)
        with (
            patch.object(prompt_builder, "get_skills_dir", return_value=skills_dir),
            patch.object(
                prompt_builder, "get_all_skills_dirs", return_value=[skills_dir]
            ),
            patch.object(
                prompt_builder, "get_disabled_skill_names", return_value=set()
            ),
            patch.object(
                prompt_builder,
                "_skills_prompt_snapshot_path",
                return_value=tmp_path / "snapshot.json",
            ),
        ):
            prompt_builder.clear_skills_system_prompt_cache(clear_snapshot=True)
            pending_prompt = prompt_builder.build_skills_system_prompt()

            pending_record = json.loads(
                (skill_dir / ".validation.json").read_text(encoding="utf-8")
            )
            passed = json.loads(
                skill_manage(
                    action="validate",
                    name="validated-skill",
                    validation={
                        "content_digest": pending_record["content_digest"],
                        "validation_token": pending_record["validation_token"],
                        "command": "pytest",
                        "exit_code": 0,
                        "output": "passed",
                    },
                )
            )
            assert passed["success"]
            passed_prompt = prompt_builder.build_skills_system_prompt()
            passed_record = json.loads(
                (skill_dir / ".validation.json").read_text(encoding="utf-8")
            )

            challenge = json.loads(
                skill_manage(action="validate", name="validated-skill")
            )
            failed = json.loads(
                skill_manage(
                    action="validate",
                    name="validated-skill",
                    validation={
                        "content_digest": challenge["content_digest"],
                        "validation_token": challenge["validation_token"],
                        "command": "pytest",
                        "exit_code": 1,
                        "output": "failed",
                    },
                )
            )
            assert failed["success"] is False
            failed_prompt = prompt_builder.build_skills_system_prompt()

    assert "validated-skill" not in pending_prompt
    assert "validated-skill" in passed_prompt, passed_record
    assert "validated-skill" not in failed_prompt


def test_cross_process_validation_change_invalidates_prompt_cache(tmp_path):
    import importlib

    prompt_builder = importlib.import_module("agent.prompt_builder")

    with isolated_skills(tmp_path) as skills_dir:
        skill_dir = create_code_skill(PASSING_TEST, tmp_path)
        passed = json.loads(
            skill_manage(
                action="validate",
                name="validated-skill",
                validation=evidence(skill_dir, 0, "passed"),
            )
        )
        assert passed["success"]
        with (
            patch.object(prompt_builder, "get_skills_dir", return_value=skills_dir),
            patch.object(
                prompt_builder, "get_all_skills_dirs", return_value=[skills_dir]
            ),
            patch.object(
                prompt_builder, "get_disabled_skill_names", return_value=set()
            ),
            patch.object(
                prompt_builder,
                "_skills_prompt_snapshot_path",
                return_value=tmp_path / "snapshot.json",
            ),
        ):
            prompt_builder.clear_skills_system_prompt_cache(clear_snapshot=True)
            cached_prompt = prompt_builder.build_skills_system_prompt()
            record_path = skill_dir / ".validation.json"
            original_stat = record_path.stat()
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["status"] = "failed"
            record["exit_code"] = 1
            record_path.write_text(
                json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            assert record_path.stat().st_size == original_stat.st_size
            os.utime(
                record_path,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )

            refreshed_prompt = prompt_builder.build_skills_system_prompt()

    assert "validated-skill" in cached_prompt
    assert "validated-skill" not in refreshed_prompt


def test_cross_process_validation_change_invalidates_catalog_cache(tmp_path):
    from tools.skills_tool import skills_list

    with isolated_skills(tmp_path):
        skill_dir = create_code_skill(PASSING_TEST, tmp_path)
        passed = json.loads(
            skill_manage(
                action="validate",
                name="validated-skill",
                validation=evidence(skill_dir, 0, "passed"),
            )
        )
        assert passed["success"]
        cached = json.loads(skills_list())
        record_path = skill_dir / ".validation.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["status"] = "failed"
        record["exit_code"] = 1
        record["output"] = "failed in another process"
        record_path.write_text(json.dumps(record), encoding="utf-8")

        refreshed = json.loads(skills_list())

    assert "validated-skill" in {item["name"] for item in cached["skills"]}
    assert "validated-skill" not in {item["name"] for item in refreshed["skills"]}


def test_cross_process_package_mutation_invalidates_discovery_caches(tmp_path):
    import importlib

    prompt_builder = importlib.import_module("agent.prompt_builder")
    from tools.skills_tool import skills_list

    with isolated_skills(tmp_path) as skills_dir:
        skill_dir = create_code_skill(PASSING_TEST, tmp_path)
        passed = json.loads(
            skill_manage(
                action="validate",
                name="validated-skill",
                validation=evidence(skill_dir, 0, "passed"),
            )
        )
        assert passed["success"]
        with (
            patch.object(prompt_builder, "get_skills_dir", return_value=skills_dir),
            patch.object(
                prompt_builder, "get_all_skills_dirs", return_value=[skills_dir]
            ),
            patch.object(
                prompt_builder, "get_disabled_skill_names", return_value=set()
            ),
            patch.object(
                prompt_builder,
                "_skills_prompt_snapshot_path",
                return_value=tmp_path / "snapshot.json",
            ),
        ):
            prompt_builder.clear_skills_system_prompt_cache(clear_snapshot=True)
            cached_prompt = prompt_builder.build_skills_system_prompt()
            cached_catalog = json.loads(skills_list())
            package_file = skill_dir / "scripts" / "add.py"
            original_stat = package_file.stat()
            package_file.write_text(
                "def add(left, right):\n    return left - right\n",
                encoding="utf-8",
            )
            assert package_file.stat().st_size == original_stat.st_size
            os.utime(
                package_file,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )

            refreshed_prompt = prompt_builder.build_skills_system_prompt()
            refreshed_catalog = json.loads(skills_list())

    assert "validated-skill" in cached_prompt
    assert "validated-skill" not in refreshed_prompt
    assert "validated-skill" in {item["name"] for item in cached_catalog["skills"]}
    assert "validated-skill" not in {
        item["name"] for item in refreshed_catalog["skills"]
    }


def test_validation_signature_limit_disables_cache_reuse(tmp_path):
    import tools.skill_validation as validation

    with isolated_skills(tmp_path) as skills_dir:
        create_code_skill(PASSING_TEST, tmp_path)
        with patch.object(validation, "MAX_VALIDATION_SIGNATURE_ENTRIES", 1):
            first = validation.validation_sidecar_signature([skills_dir])
            second = validation.validation_sidecar_signature([skills_dir])

    assert first != second
    assert any("validation_signature_" in str(entry[0]) for entry in first)


def test_validation_signature_enforces_per_package_bound(tmp_path):
    import tools.skill_validation as validation

    with isolated_skills(tmp_path) as skills_dir:
        create_code_skill(PASSING_TEST, tmp_path)
        with patch.object(validation, "MAX_VALIDATED_PACKAGE_FILES", 1):
            signature = validation.validation_sidecar_signature([skills_dir])

    assert any(
        len(entry) > 1 and entry[1] == "package_limit_exceeded"
        for entry in signature
    )


def test_validation_signature_does_not_follow_symlink_cycles(tmp_path):
    import tools.skill_validation as validation

    with isolated_skills(tmp_path) as skills_dir:
        skill_dir = create_code_skill(PASSING_TEST, tmp_path)
        cycle = skill_dir / "scripts" / "cycle"
        try:
            cycle.symlink_to(skill_dir, target_is_directory=True)
        except OSError:
            return

        signature = validation.validation_sidecar_signature([skills_dir])

    assert any(str(cycle) == entry[0] for entry in signature)
    assert len(signature) < 100


def test_validation_sidecar_symlink_is_not_read(tmp_path):
    from tools.skill_validation import read_skill_validation

    with isolated_skills(tmp_path) as skills_dir:
        assert _create_skill("validated-skill", VALID_SKILL)["success"]
        outside = tmp_path / "outside.json"
        outside.write_text(
            json.dumps({"status": "passed", "secret": "must-not-leak"}),
            encoding="utf-8",
        )
        sidecar = skills_dir / "validated-skill" / ".validation.json"
        try:
            sidecar.symlink_to(outside)
        except OSError:
            return

        record = read_skill_validation(skills_dir / "validated-skill")

    assert record == {
        "status": "invalid",
        "reason": "validation record cannot be a symlink",
    }


def test_skill_directory_symlink_swap_cannot_redirect_validation_write(
    tmp_path, monkeypatch
):
    with isolated_skills(tmp_path) as skills_dir:
        assert _create_skill("validated-skill", VALID_SKILL)["success"]
        skill_dir = skills_dir / "validated-skill"
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
        result = json.loads(skill_manage(action="validate", name="validated-skill"))

    assert result["success"] is False
    assert not (outside / ".validation.json").exists()


def test_unknown_validation_schema_or_status_is_fail_closed(tmp_path):
    from tools.skill_validation import (
        read_skill_validation,
        validation_allows_discovery,
    )

    with isolated_skills(tmp_path):
        skill_dir = create_code_skill(PASSING_TEST, tmp_path)
        record_path = skill_dir / ".validation.json"
        base = json.loads(record_path.read_text(encoding="utf-8"))

        base["schema"] = "unknown"
        base["status"] = "passed"
        record_path.write_text(json.dumps(base), encoding="utf-8")
        assert read_skill_validation(skill_dir)["status"] == "invalid"
        assert validation_allows_discovery(skill_dir) is False

        base["schema"] = "hermes.skill-validation.v1"
        base["status"] = "surprise"
        record_path.write_text(json.dumps(base), encoding="utf-8")
        assert read_skill_validation(skill_dir)["status"] == "invalid"
        assert validation_allows_discovery(skill_dir) is False


def test_status_specific_validation_schema_is_fail_closed(tmp_path):
    from tools.skill_validation import (
        VALIDATION_SCHEMA,
        read_skill_validation,
        skill_content_digest,
        validation_allows_discovery,
    )

    with isolated_skills(tmp_path):
        skill_dir = create_code_skill(PASSING_TEST, tmp_path)
        record_path = skill_dir / ".validation.json"
        digest = skill_content_digest(skill_dir)
        malformed = [
            {
                "schema": VALIDATION_SCHEMA,
                "status": "static",
                "tests_collected": 1,
                "content_digest": digest,
                "validated_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "schema": VALIDATION_SCHEMA,
                "status": "passed",
                "tests_collected": 0,
                "content_digest": digest,
                "command": "pytest",
                "exit_code": 0,
                "output": "passed",
                "validated_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "schema": VALIDATION_SCHEMA,
                "status": "passed",
                "tests_collected": 1,
                "content_digest": digest,
                "command": "pytest",
                "exit_code": True,
                "output": "passed",
                "validated_at": "2026-01-01T00:00:00+00:00",
            },
        ]
        for record in malformed:
            record_path.write_text(json.dumps(record), encoding="utf-8")
            assert read_skill_validation(skill_dir)["status"] == "invalid"
            assert validation_allows_discovery(skill_dir) is False


def test_oversized_validation_record_is_not_loaded(tmp_path):
    from tools.skill_validation import read_skill_validation

    with isolated_skills(tmp_path) as skills_dir:
        assert _create_skill("validated-skill", VALID_SKILL)["success"]
        record_path = skills_dir / "validated-skill" / ".validation.json"
        record_path.write_text("x" * 70_000, encoding="utf-8")
        record = read_skill_validation(skills_dir / "validated-skill")

    assert record == {"status": "invalid", "reason": "validation record is too large"}


def test_validation_output_redacts_secrets_before_persistence(tmp_path):
    secret = "MY_API_TOKEN=abcdefghijklmnopqrstuv"
    with isolated_skills(tmp_path):
        skill_dir = create_code_skill(PASSING_TEST, tmp_path)
        result = json.loads(
            skill_manage(
                action="validate",
                name="validated-skill",
                validation=evidence(skill_dir, 1, f"request failed with {secret}"),
            )
        )
        persisted = (skill_dir / ".validation.json").read_text(encoding="utf-8")

    assert result["success"] is False
    assert secret not in result["test_output"]
    assert secret not in persisted
    assert "REDACTED" in persisted


def test_validate_schema_and_tests_directory_are_supported():
    from tools.skill_manager_tool import SKILL_MANAGE_SCHEMA, _validate_file_path

    properties = SKILL_MANAGE_SCHEMA["parameters"]["properties"]
    assert "validate" in properties["action"]["enum"]
    assert "content_digest" in properties["validation"]["required"]
    assert "validation_token" in properties["validation"]["required"]
    assert _validate_file_path("tests/test_behavior.py") is None
