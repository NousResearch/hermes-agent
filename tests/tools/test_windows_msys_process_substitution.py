"""Tests for the Windows-only MSYS process substitution guard.

On Windows git-bash, bash process substitution ``<(cmd)`` / ``>(cmd)``
expands to ``/dev/fd/N`` — an MSYS pipe handle that native Windows programs
(python, git, curl, uv, ...) cannot open. The guard in
``check_all_command_guards`` blocks these commands unconditionally on win32
(before the yolo / mode=off bypass) and must stay completely inert on POSIX,
where process substitution is a legitimate, working feature.

v2 (2026-08-09): shell-aware state machine (quotes / comments / escapes).
Quoted literals, escaped ``\<(``, and comment bodies are DATA, not process
substitution — v1's bare substring regex false-positived on all three.
"""

import os

import pytest

import tools.approval as approval_module
from tools.approval import check_all_command_guards


@pytest.fixture(autouse=True)
def _clean_state():
    """Clear approval state and relevant env vars between tests."""
    approval_module._session_approved.clear()
    approval_module._pending.clear()
    approval_module._permanent_approved.clear()
    saved = {}
    for k in ("HERMES_INTERACTIVE", "HERMES_GATEWAY_SESSION",
              "HERMES_EXEC_ASK", "HERMES_YOLO_MODE"):
        if k in os.environ:
            saved[k] = os.environ.pop(k)
    yield
    approval_module._session_approved.clear()
    approval_module._pending.clear()
    approval_module._permanent_approved.clear()
    for k, v in saved.items():
        os.environ[k] = v
    for k in ("HERMES_INTERACTIVE", "HERMES_GATEWAY_SESSION",
              "HERMES_EXEC_ASK", "HERMES_YOLO_MODE"):
        os.environ.pop(k, None)


class TestWindowsBlocksProcessSubstitution:
    """On win32 the guard fires for code-position process substitution."""

    @pytest.fixture(autouse=True)
    def _force_windows(self, monkeypatch):
        monkeypatch.setattr(approval_module, "_IS_WINDOWS", True)

    def test_blocks_input_process_substitution(self):
        # The cst_runtime --args-file <(...) failure class: native consumer
        # receiving an MSYS pipe handle as a path argument.
        result = check_all_command_guards("python x.py <(cat a)", "local")
        assert result["approved"] is False
        assert "MSYS process substitution" in result["message"]
        # Message must teach the L0 R1 alternative (write a real file).
        assert "mktemp" in result["message"] or "write_file" in result["message"]

    def test_blocks_output_process_substitution(self):
        result = check_all_command_guards("diff >(sort b) other.txt", "local")
        assert result["approved"] is False

    def test_blocks_dev_fd_reference(self):
        result = check_all_command_guards("grep -l pattern /dev/fd/63", "local")
        assert result["approved"] is False

    def test_blocks_nested_mid_argument(self):
        # Not just at command position: nested mid-argument is also doomed.
        result = check_all_command_guards(
            "cst_runtime --args-file <(cat /tmp/x.json)", "local")
        assert result["approved"] is False

    def test_blocks_after_command_separator(self):
        result = check_all_command_guards("ls; python x.py <(cat a)", "local")
        assert result["approved"] is False

    def test_blocks_parameter_expansion_default(self):
        # ${x:-<(a)} genuinely executes process substitution in bash — the
        # state machine has no way to know x is unset, so this stays blocked.
        result = check_all_command_guards("echo ${x:-<(a)}", "local")
        assert result["approved"] is False

    def test_blocks_standalone_output_substitution_word(self):
        # Bare >(b) with no quotes is a real output process substitution.
        result = check_all_command_guards("echo a >(b)", "local")
        assert result["approved"] is False

    def test_blocks_even_when_yolo_frozen(self):
        # Like the hardline floor: a structurally doomed command is not a
        # risk trade-off the user opted into, so yolo cannot bypass it.
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", True)
        try:
            result = check_all_command_guards("python x.py <(cat a)", "local")
        finally:
            monkeypatch.undo()
        assert result["approved"] is False

    def test_allows_plain_input_redirection(self):
        # `< file` (no paren) is ordinary redirection — must not be caught.
        result = check_all_command_guards("python x.py < input.txt", "local")
        assert result["approved"] is True

    def test_allows_normal_commands(self):
        result = check_all_command_guards("ls -la", "local")
        assert result["approved"] is True

    def test_allows_windows_native_flags(self):
        # #56700 regression: tasklist /FO must stay untouched.
        result = check_all_command_guards("tasklist /FO CSV", "local")
        assert result["approved"] is True


class TestWindowsAllowsLiteralData:
    """Quoted / escaped / comment occurrences are DATA, not process substitution (v2)."""

    @pytest.fixture(autouse=True)
    def _force_windows(self, monkeypatch):
        monkeypatch.setattr(approval_module, "_IS_WINDOWS", True)

    def test_allows_double_quoted_literal(self):
        result = check_all_command_guards('echo "literal <( text"', "local")
        assert result["approved"] is True

    def test_allows_single_quoted_grep_pattern(self):
        result = check_all_command_guards("grep '<(' file.txt", "local")
        assert result["approved"] is True

    def test_allows_escaped_literal(self):
        # v1 blocked `\<(` — wrong: bash treats the backslash-escaped <( as
        # a literal. v2 correctly allows it (escape consumed before match).
        result = check_all_command_guards(r"python x.py \<(cat a)", "local")
        assert result["approved"] is True

    def test_allows_comment_occurrence(self):
        result = check_all_command_guards(
            "# comment with <( here\necho ok", "local")
        assert result["approved"] is True

    def test_allows_piped_single_quoted_pattern(self):
        result = check_all_command_guards("echo hi | grep '<('", "local")
        assert result["approved"] is True


class TestPosixUnaffected:
    """On POSIX the guard must be completely inert."""

    @pytest.fixture(autouse=True)
    def _force_posix(self, monkeypatch):
        monkeypatch.setattr(approval_module, "_IS_WINDOWS", False)

    def test_process_substitution_allowed_on_posix(self):
        # On Linux <(...) is a legitimate feature — never blocked.
        result = check_all_command_guards("diff <(sort a) <(sort b)", "local")
        assert result["approved"] is True

    def test_dev_fd_reference_allowed_on_posix(self):
        result = check_all_command_guards("grep -l pattern /dev/fd/63", "local")
        assert result["approved"] is True
