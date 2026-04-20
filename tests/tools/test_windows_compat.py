"""Tests for Windows compatibility of process management code.

Verifies that os.setsid and os.killpg are never called unconditionally,
and that each module uses a platform guard before invoking POSIX-only functions.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

# Files that must have Windows-safe process management
GUARDED_FILES = [
    "tools/environments/local.py",
    "tools/process_registry.py",
    "tools/code_execution_tool.py",
    "gateway/platforms/whatsapp.py",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _get_preexec_fn_values(filepath: Path) -> list:
    """Find all preexec_fn= keyword arguments in Popen calls."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))
    values = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "preexec_fn":
            values.append(ast.dump(node.value))
    return values


class TestNoUnconditionalSetsid:
    """preexec_fn must never be a bare os.setsid reference."""

    @pytest.mark.parametrize("relpath", GUARDED_FILES)
    def test_preexec_fn_is_guarded(self, relpath):
        filepath = PROJECT_ROOT / relpath
        if not filepath.exists():
            pytest.skip(f"{relpath} not found")
        values = _get_preexec_fn_values(filepath)
        for val in values:
            # A bare os.setsid would be: Attribute(value=Name(id='os'), attr='setsid')
            assert "attr='setsid'" not in val or "IfExp" in val or "None" in val, (
                f"{relpath} has unconditional preexec_fn=os.setsid"
            )


class TestIsWindowsConstant:
    """Each guarded file must define _IS_WINDOWS."""

    @pytest.mark.parametrize("relpath", GUARDED_FILES)
    def test_has_is_windows(self, relpath):
        filepath = PROJECT_ROOT / relpath
        if not filepath.exists():
            pytest.skip(f"{relpath} not found")
        source = filepath.read_text(encoding="utf-8")
        assert "_IS_WINDOWS" in source, (
            f"{relpath} missing _IS_WINDOWS platform guard"
        )


class TestKillpgGuarded:
    """os.killpg must always be behind a platform check."""

    @pytest.mark.parametrize("relpath", GUARDED_FILES)
    def test_no_unguarded_killpg(self, relpath):
        filepath = PROJECT_ROOT / relpath
        if not filepath.exists():
            pytest.skip(f"{relpath} not found")
        source = filepath.read_text(encoding="utf-8")
        lines = source.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if "os.killpg" in stripped or "os.getpgid" in stripped:
                # Check that there's an _IS_WINDOWS guard in the surrounding context
                context = "\n".join(lines[max(0, i - 15):i + 1])
                assert "_IS_WINDOWS" in context or "else:" in context, (
                    f"{relpath}:{i + 1} has unguarded os.killpg/os.getpgid call"
                )


class TestWindowsBashSelection:
    """Windows should use Git Bash for local shell execution, not WSL bash."""

    def test_prefers_git_bash_over_wsl_bash_on_path(self, tmp_path, monkeypatch):
        from tools.environments import local as local_env

        git_bash = tmp_path / "Git" / "bin" / "bash.exe"
        git_bash.parent.mkdir(parents=True)
        git_bash.write_text("", encoding="utf-8")

        monkeypatch.setattr(local_env, "_IS_WINDOWS", True)
        monkeypatch.setenv("ProgramFiles", str(tmp_path))
        monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path / "missing-x86"))
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "missing-local"))
        monkeypatch.setattr(
            local_env.shutil,
            "which",
            lambda name: r"C:\Windows\System32\bash.EXE" if name == "bash" else None,
        )

        assert local_env._find_bash() == str(git_bash)

    def test_rejects_wsl_bash_when_git_bash_is_missing(self, tmp_path, monkeypatch):
        from tools.environments import local as local_env

        monkeypatch.setattr(local_env, "_IS_WINDOWS", True)
        monkeypatch.setenv("ProgramFiles", str(tmp_path / "missing"))
        monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path / "missing-x86"))
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "missing-local"))
        monkeypatch.setenv("WINDIR", r"C:\Windows")
        monkeypatch.setattr(
            local_env.shutil,
            "which",
            lambda name: r"C:\Windows\System32\bash.exe" if name == "bash" else None,
        )

        with pytest.raises(RuntimeError, match="Git Bash not found"):
            local_env._find_bash()

    def test_allows_non_wsl_bash_from_path(self, tmp_path, monkeypatch):
        from tools.environments import local as local_env

        path_bash = tmp_path / "tools" / "bash.exe"
        path_bash.parent.mkdir(parents=True)
        path_bash.write_text("", encoding="utf-8")

        monkeypatch.setattr(local_env, "_IS_WINDOWS", True)
        monkeypatch.setenv("ProgramFiles", str(tmp_path / "missing"))
        monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path / "missing-x86"))
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "missing-local"))
        monkeypatch.setattr(
            local_env.shutil,
            "which",
            lambda name: str(path_bash) if name == "bash" else None,
        )

        assert local_env._find_bash() == str(path_bash)


class TestWindowsPipeDrain:
    """Windows pipe draining cannot rely on select.select()."""

    def test_wait_for_process_drains_stdout_with_windows_path(self, monkeypatch):
        import tools.environments.base as base_env
        from tools.environments.base import BaseEnvironment

        class DummyEnvironment(BaseEnvironment):
            def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
                raise NotImplementedError

            def cleanup(self):
                pass

        monkeypatch.setattr(base_env.os, "name", "nt", raising=False)
        env = DummyEnvironment(cwd=".", timeout=5)
        proc = subprocess.Popen(
            [sys.executable, "-c", "print('hermes-local-test')"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        result = env._wait_for_process(proc, timeout=5)

        assert result["returncode"] == 0
        assert "hermes-local-test" in result["output"]
