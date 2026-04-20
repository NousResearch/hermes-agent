import hashlib
import json
from pathlib import Path

import tools.file_tools as file_tools


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_replace_mode_hashline_race_never_mutates_file(tmp_path, monkeypatch):
    target = tmp_path / "sample.txt"
    target.write_text("hello world\n", encoding="utf-8")
    original_hash = _sha256(target)

    original_conditional_write = file_tools.ShellFileOperations.conditional_write_file
    mutated_once = False

    def mutate_before_conditional_write(self, path: str, content: str, expected_sha256: str):
        nonlocal mutated_once
        if str(path) == str(target) and not mutated_once:
            mutated_once = True
            Path(path).write_text("hello world\nINTRUDE\n", encoding="utf-8")
        return original_conditional_write(self, path, content, expected_sha256)

    monkeypatch.setattr(
        file_tools.ShellFileOperations,
        "conditional_write_file",
        mutate_before_conditional_write,
    )

    result = json.loads(
        file_tools.patch_tool(
            mode="replace",
            path=str(target),
            old_string=f"# HASHLINE sha256:{original_hash}\nhello world",
            new_string="hello hermes",
            task_id="hashline-race-replace",
        )
    )

    assert "HASHLINE validation failed" in result["error"]
    assert target.read_text(encoding="utf-8") == "hello world\nINTRUDE\n"



def test_patch_mode_hashline_race_fails_without_partial_mutation(tmp_path, monkeypatch):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("alpha\n", encoding="utf-8")
    second.write_text("beta\n", encoding="utf-8")

    patch_text = f"""*** Begin Patch
*** Hashline: {first} sha256:{_sha256(first)}
*** Update File: {first}
@@
-alpha
+alpha patched
*** Hashline: {second} sha256:{_sha256(second)}
*** Update File: {second}
@@
-beta
+beta patched
*** End Patch
"""

    original_conditional_write = file_tools.ShellFileOperations.conditional_write_file
    mutated_once = False

    def mutate_second_before_conditional_write(self, path: str, content: str, expected_sha256: str):
        nonlocal mutated_once
        if str(path) == str(second) and not mutated_once:
            mutated_once = True
            second.write_text("beta\nINTRUDE\n", encoding="utf-8")
        return original_conditional_write(self, path, content, expected_sha256)

    monkeypatch.setattr(
        file_tools.ShellFileOperations,
        "conditional_write_file",
        mutate_second_before_conditional_write,
    )

    result = json.loads(file_tools.patch_tool(mode="patch", patch=patch_text, task_id="hashline-race-patch"))

    assert "HASHLINE validation failed" in result["error"]
    assert first.read_text(encoding="utf-8") == "alpha\n"
    assert second.read_text(encoding="utf-8") == "beta\nINTRUDE\n"



def test_reread_then_reapply_succeeds_after_hashline_race_failure(tmp_path, monkeypatch):
    target = tmp_path / "sample.txt"
    target.write_text("hello world\n", encoding="utf-8")
    stale_hash = _sha256(target)

    original_conditional_write = file_tools.ShellFileOperations.conditional_write_file
    mutated_once = False

    def mutate_once_before_conditional_write(self, path: str, content: str, expected_sha256: str):
        nonlocal mutated_once
        if str(path) == str(target) and not mutated_once:
            mutated_once = True
            Path(path).write_text("hello world\nINTRUDE\n", encoding="utf-8")
        return original_conditional_write(self, path, content, expected_sha256)

    monkeypatch.setattr(
        file_tools.ShellFileOperations,
        "conditional_write_file",
        mutate_once_before_conditional_write,
    )

    first_result = json.loads(
        file_tools.patch_tool(
            mode="replace",
            path=str(target),
            old_string=f"# HASHLINE sha256:{stale_hash}\nhello world",
            new_string="hello hermes",
            task_id="hashline-race-reread",
        )
    )

    assert "HASHLINE validation failed" in first_result["error"]
    assert target.read_text(encoding="utf-8") == "hello world\nINTRUDE\n"

    fresh_hash = _sha256(target)
    second_result = json.loads(
        file_tools.patch_tool(
            mode="replace",
            path=str(target),
            old_string=f"# HASHLINE sha256:{fresh_hash}\nhello world\nINTRUDE",
            new_string="hello hermes\nINTRUDE",
            task_id="hashline-race-reread",
        )
    )

    assert not second_result.get("error")
    assert target.read_text(encoding="utf-8") == "hello hermes\nINTRUDE\n"



def test_patch_mode_rollback_skips_restore_when_post_apply_state_changed(tmp_path, monkeypatch):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("alpha\n", encoding="utf-8")
    second.write_text("beta\n", encoding="utf-8")

    patch_text = f"""*** Begin Patch
*** Hashline: {first} sha256:{_sha256(first)}
*** Update File: {first}
@@
-alpha
+alpha patched
*** Hashline: {second} sha256:{_sha256(second)}
*** Update File: {second}
@@
-beta
+beta patched
*** End Patch
"""

    original_conditional_write = file_tools.ShellFileOperations.conditional_write_file
    intruded = False

    def mutate_first_before_second_write(self, path: str, content: str, expected_sha256: str):
        nonlocal intruded
        if str(path) == str(second) and not intruded:
            intruded = True
            first.write_text("alpha patched\nTHIRD PARTY\n", encoding="utf-8")
            second.write_text("beta\nINTRUDE\n", encoding="utf-8")
        return original_conditional_write(self, path, content, expected_sha256)

    monkeypatch.setattr(
        file_tools.ShellFileOperations,
        "conditional_write_file",
        mutate_first_before_second_write,
    )

    result = json.loads(file_tools.patch_tool(mode="patch", patch=patch_text, task_id="hashline-rollback-clobber"))

    assert "HASHLINE validation failed" in result["error"]
    assert "rollback could not safely proceed for" in result["error"]
    assert str(first) in result["error"]
    assert first.read_text(encoding="utf-8") == "alpha patched\nTHIRD PARTY\n"
    assert second.read_text(encoding="utf-8") == "beta\nINTRUDE\n"
