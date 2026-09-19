"""Real-path regressions for V4A candidate preflight."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.patch_parser import apply_v4a_operations, parse_v4a_patch


def _apply(patch: str, root: Path):
    operations, error = parse_v4a_patch(patch)
    assert error is None
    file_ops = ShellFileOperations(LocalEnvironment(cwd=str(root)), cwd=str(root))
    return apply_v4a_operations(operations, file_ops)


@pytest.mark.parametrize("phase", ["batch", "candidate", "apply"])
def test_existing_binary_add_is_rejected(tmp_path: Path, phase: str):
    from tools.patch_parser import _apply_add, _validate_operations

    good = tmp_path / "good.txt"
    binary = tmp_path / "existing.bin"
    original = b"\x00ORIGINAL-BYTES\xff"
    good.write_text("old\n")
    binary.write_bytes(original)
    file_ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)), cwd=str(tmp_path))
    read = file_ops.read_file_raw(str(binary))
    assert read.is_binary and read.error
    operations, error = parse_v4a_patch(
        f"*** Begin Patch\n*** Update File: {good}\n@@\n-old\n+new\n"
        f"*** Add File: {binary}\n+replacement text\n*** End Patch\n"
    )
    assert error is None
    if phase == "batch":
        result = file_ops.patch_v4a(
            f"*** Begin Patch\n*** Update File: {good}\n@@\n-old\n+new\n"
            f"*** Add File: {binary}\n+replacement text\n*** End Patch\n"
        )
        assert result.success is False
        assert "no files were modified" in (result.error or "")
        assert not result.files_created and not result.files_modified
    elif phase == "candidate":
        assert _validate_operations(operations, file_ops)
    else:
        assert _apply_add(operations[-1], file_ops)[0] is False
    assert good.read_text() == "old\n"
    assert binary.read_bytes() == original


@pytest.mark.parametrize("error", [
    "Failed to read file: Permission denied",
    "Terminal environment unavailable: could not stat target",
    "Not a regular file: target",
    "Failed to read file: File not found: target",
])
def test_add_read_failures_do_not_authorize_writes(error):
    from tools.file_operations_common import ReadResult
    from tools.patch_parser import _apply_add, _validate_operations

    operations, parse_error = parse_v4a_patch("*** Add File: target\n+new")
    assert parse_error is None
    file_ops = MagicMock()
    file_ops.read_file_raw.return_value = ReadResult(error=error)
    assert error in "\n".join(_validate_operations(operations, file_ops))
    assert error in _apply_add(operations[0], file_ops)[1]
    file_ops.write_file.assert_not_called()
    file_ops.validate_write_candidate.assert_not_called()


def test_missing_add_target_is_created(tmp_path: Path):
    target = tmp_path / "new.txt"
    file_ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)), cwd=str(tmp_path))
    assert file_ops.read_file_raw(str(target)).error == f"File not found: {target}"
    result = file_ops.patch_v4a(f"*** Add File: {target}\n+new")
    assert result.success
    assert target.read_text() == "new"


def test_invalid_structured_candidate_blocks_entire_batch(tmp_path: Path):
    good = tmp_path / "good.txt"
    structured = tmp_path / "config.json"
    good.write_text("old\n")
    structured.write_text('{"ok": true}\n')

    result = _apply(
        f"""*** Begin Patch
*** Update File: {good}
@@
-old
+new
*** Update File: {structured}
@@
-{{"ok": true}}
+{{"ok":
*** End Patch
""",
        tmp_path,
    )

    assert result.success is False
    assert "no files were modified" in (result.error or "")
    assert good.read_text() == "old\n"
    assert structured.read_text() == '{"ok": true}\n'


@pytest.mark.parametrize(
    "filename, content, error_fragment",
    [("new.json", '{"ok":', "syntax validation"),
     ("new.txt", "\ud800", "lone surrogate")],
)
def test_invalid_add_candidate_blocks_earlier_update(
    tmp_path: Path, filename: str, content: str, error_fragment: str,
):
    good = tmp_path / "good.txt"
    good.write_text("old\n")
    added = tmp_path / filename
    result = _apply(
        f"*** Begin Patch\n*** Update File: {good}\n@@\n-old\n+new\n"
        f"*** Add File: {added}\n+{content}\n*** End Patch\n",
        tmp_path,
    )
    assert result.success is False
    assert "no files were modified" in (result.error or "")
    assert error_fragment in (result.error or "")
    assert good.read_text() == "old\n"
    assert not added.exists()


def test_write_policy_denial_blocks_entire_batch(
    tmp_path: Path,
    monkeypatch,
):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    protected = hermes_home / ".env"
    protected.write_text("SECRET=unchanged\n")
    good = tmp_path / "good.txt"
    good.write_text("old\n")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    result = _apply(
        f"""*** Begin Patch
*** Update File: {good}
@@
-old
+new
*** Update File: {protected}
@@
-SECRET=unchanged
+SECRET=overwritten
*** End Patch
""",
        tmp_path,
    )

    assert result.success is False
    assert "no files were modified" in (result.error or "")
    assert "Write denied" in (result.error or "")
    assert good.read_text() == "old\n"
    assert protected.read_text() == "SECRET=unchanged\n"


def test_non_string_backend_preflight_result_is_not_a_rejection(tmp_path: Path):
    target = tmp_path / "plain.txt"
    target.write_text("old\n")
    file_ops = MagicMock()
    file_ops.read_file_raw.return_value.content = "old\n"
    file_ops.read_file_raw.return_value.error = None
    file_ops.validate_write_candidate.return_value = True
    file_ops.write_file.return_value.error = None
    file_ops.write_file.return_value.lsp_diagnostics = None

    operations, error = parse_v4a_patch(
        f"""*** Begin Patch
*** Update File: {target}
@@
-old
+new
*** End Patch
"""
    )
    assert error is None

    result = apply_v4a_operations(operations, file_ops)

    assert result.success is True
    file_ops.write_file.assert_called_once_with(str(target), "new\n", pre_content="old\n")


def test_missing_addition_only_hint_rejects_without_appending(tmp_path: Path):
    target = tmp_path / "plain.py"
    target.write_text("x = 1\n")

    result = _apply(
        f"""*** Begin Patch
*** Update File: {target}
@@ missing_anchor @@
+y = 2
*** End Patch
""",
        tmp_path,
    )

    assert result.success is False
    assert "missing_anchor" in (result.error or "")
    assert "not found" in (result.error or "")
    assert target.read_text() == "x = 1\n"
