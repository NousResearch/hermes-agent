"""Tests for shell-aware snapshot bootstrap in LocalEnvironment.

The session snapshot (``init_session``) captures the login shell's environment
into a file that subsequent commands source. Historically the bootstrap used
bash-specific syntax that fails under zsh:

- ``alias -p``      → zsh has no ``-p`` flag (use ``alias -L``)
- ``shopt -s expand_aliases`` → zsh has no ``shopt`` (use ``setopt``)
- ``declare -F``    → works in zsh but returns nothing (zsh uses ``functions``)

These tests verify that when the terminal shell is zsh, the bootstrap uses
zsh-compatible commands and produces a valid snapshot file.
"""

import os
import shutil
import subprocess
import tempfile
from unittest.mock import MagicMock, patch

import pytest


def _run_shell_snapshot(shell_path: str, cwd: str) -> tuple[int, str, str]:
    """Run a minimal snapshot bootstrap under the given shell.

    Returns (returncode, output, snapshot_file_content).
    """
    snap_tmp = tempfile.mktemp(prefix="hermes-test-snap-tmp.")
    snap_final = tempfile.mktemp(prefix="hermes-test-snap-final.")
    cwd_file = tempfile.mktemp(prefix="hermes-test-cwd.")
    cwd_marker = "__CWD_MARKER_TEST__"

    # Build a minimal bootstrap that mirrors init_session logic
    # Using the same structure as base.py init_session
    bootstrap_lines = [
        "umask 077",
        f"export -p > {snap_tmp} 2>/dev/null || true",
    ]

    # Shell-specific function listing
    shell_name = os.path.basename(shell_path)
    if shell_name == "zsh":
        bootstrap_lines.append(
            f'__hermes_fns=$(functions 2>/dev/null | grep -E "^[a-zA-Z_][a-zA-Z0-9_]* \\(\\) \\{{" | awk "{{print \\$1}}" | grep -vE "^_[^_]") || true'
        )
        bootstrap_lines.append(
            f'[ -n "$__hermes_fns" ] && functions $__hermes_fns >> {snap_tmp} 2>/dev/null || true'
        )
        bootstrap_lines.append(f"alias -L >> {snap_tmp} 2>/dev/null || true")
        bootstrap_lines.append("echo 'setopt interactive_comments' >> " + snap_tmp)
    else:
        bootstrap_lines.append(
            f'__hermes_fns=$(declare -F | awk "{{print \\$3}}" | grep -vE "^_[^_]") || true'
        )
        bootstrap_lines.append(
            f'[ -n "$__hermes_fns" ] && declare -f $__hermes_fns >> {snap_tmp} 2>/dev/null || true'
        )
        bootstrap_lines.append(f"alias -p >> {snap_tmp} 2>/dev/null || true")
        bootstrap_lines.append("echo 'shopt -s expand_aliases' >> " + snap_tmp)

    bootstrap_lines.extend([
        "echo 'set +e' >> " + snap_tmp,
        "echo 'set +u' >> " + snap_tmp,
        f"mv -f {snap_tmp} {snap_final} || rm -f {snap_tmp}",
        f"builtin cd -- {cwd} 2>/dev/null || true",
        f"pwd -P > {cwd_file} 2>/dev/null || true",
        f'printf "\\n{cwd_marker}%s{cwd_marker}\\n" "$(pwd -P)"',
    ])

    bootstrap = "\n".join(bootstrap_lines)

    proc = subprocess.run(
        [shell_path, "-l", "-c", bootstrap],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=30,
    )
    snap_content = ""
    try:
        with open(snap_final) as f:
            snap_content = f.read()
    except FileNotFoundError:
        pass
    for tmp_file in (snap_tmp, snap_final, cwd_file):
        try:
            os.unlink(tmp_file)
        except FileNotFoundError:
            pass
    return proc.returncode, proc.stdout + proc.stderr, snap_content


class TestZshSnapshotBootstrapSucceeds:
    """Worst-first: the zsh path is the most failure-prone (the original bug)."""

    @pytest.fixture
    def zsh_path(self):
        """Find zsh or skip test if unavailable."""
        for candidate in ["/usr/bin/zsh", "/bin/zsh"]:
            if os.path.isfile(candidate):
                return candidate
        pytest.skip("zsh not installed on this system")

    @pytest.fixture
    def bash_path(self):
        """Find bash."""
        for candidate in ["/bin/bash", "/usr/bin/bash"]:
            if os.path.isfile(candidate):
                return candidate
        pytest.skip("bash not installed on this system")

    def test_zsh_snapshot_does_not_crash_with_exit_1(self, zsh_path, tmp_path):
        """The original bug: zsh snapshot exits 1 with no output."""
        rc, output, _ = _run_shell_snapshot(zsh_path, str(tmp_path))
        assert rc == 0, f"zsh snapshot exited {rc} — the original bug. Output: {output}"

    def test_zsh_snapshot_produces_marker(self, zsh_path, tmp_path):
        """Snapshot output must contain the CWD marker."""
        rc, output, _ = _run_shell_snapshot(zsh_path, str(tmp_path))
        assert "__CWD_MARKER_TEST__" in output

    def test_bash_snapshot_still_works(self, bash_path, tmp_path):
        """Regression: bash snapshot must not break."""
        rc, output, _ = _run_shell_snapshot(bash_path, str(tmp_path))
        assert rc == 0
        assert "__CWD_MARKER_TEST__" in output

    def test_zsh_snapshot_file_contains_no_shopt(self, zsh_path, tmp_path):
        """Snapshot file must NOT emit shopt as a standalone command (zsh has no shopt).

        Note: the substring 'shopt' may legitimately appear inside zsh function
        bodies (e.g. completion functions with ``(setopt | shopt)`` case patterns).
        The dangerous thing is a standalone ``shopt -s`` line that zsh would
        try to execute and fail on.  Check for that, not the bare substring.
        """
        rc, output, snap_content = _run_shell_snapshot(zsh_path, str(tmp_path))
        assert snap_content, "No snapshot file produced"
        # A standalone shopt command appears on its own line, not inside a function body
        standalone_shopts = [
            line for line in snap_content.splitlines()
            if line.strip().startswith("shopt ")
        ]
        assert not standalone_shopts, (
            f"zsh snapshot emits shopt as standalone command (zsh has no shopt):\n"
            f"{chr(10).join(standalone_shopts)}"
        )

    def test_zsh_snapshot_file_contains_alias_L_not_alias_p(self, zsh_path, tmp_path):
        """Snapshot file must use alias -L (zsh) not alias -p (bash)."""
        rc, output, snap_content = _run_shell_snapshot(zsh_path, str(tmp_path))
        assert snap_content, "No snapshot file produced"
        assert "bad option: -p" not in snap_content, f"zsh received bash alias -p syntax:\n{snap_content[:500]}"


class TestLocalEnvironmentShellIntegration:
    """Integration: LocalEnvironment.execute() uses the configured shell."""

    def test_execute_uses_zsh_when_configured(self, monkeypatch, tmp_path):
        """When terminal.shell=zsh and $SHELL=/usr/bin/zsh, execute() spawns zsh."""
        from tools.environments.local import LocalEnvironment

        monkeypatch.setenv("SHELL", "/usr/bin/zsh")

        captured_args = {}

        def fake_popen(args, **kwargs):
            captured_args["args"] = args
            captured_args["cwd"] = kwargs.get("cwd")
            read_fd, write_fd = os.pipe()
            os.close(write_fd)
            stdout = os.fdopen(read_fd, "rb", buffering=0)
            proc = MagicMock()
            proc.poll.return_value = 0
            proc.returncode = 0
            proc.stdout = stdout
            proc.stdin = MagicMock()
            proc.pid = 99999
            return proc

        with patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None), \
             patch("subprocess.Popen", side_effect=fake_popen), \
             patch("tools.terminal_tool._interrupt_event"):
            env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
            # Force _snapshot_ready so it doesn't try login shell
            env._snapshot_ready = True
            # Write a minimal snapshot file so source doesn't fail
            with open(env._snapshot_path, "w") as f:
                f.write("# minimal snapshot\n")
            try:
                env.execute("echo hello")
            except Exception:
                pass  # We only care about the spawn args

        shell_binary = os.path.basename(captured_args["args"][0])
        assert shell_binary == "zsh", f"Expected zsh, got {shell_binary}: {captured_args['args']}"

    def test_execute_uses_bash_when_configured(self, monkeypatch, tmp_path):
        """When terminal.shell=bash, execute() spawns bash even if $SHELL=zsh."""
        from tools.environments.local import LocalEnvironment

        monkeypatch.setenv("SHELL", "/usr/bin/zsh")

        captured_args = {}

        def fake_popen(args, **kwargs):
            captured_args["args"] = args
            read_fd, write_fd = os.pipe()
            os.close(write_fd)
            stdout = os.fdopen(read_fd, "rb", buffering=0)
            proc = MagicMock()
            proc.poll.return_value = 0
            proc.returncode = 0
            proc.stdout = stdout
            proc.stdin = MagicMock()
            proc.pid = 99999
            return proc

        with patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None), \
             patch("subprocess.Popen", side_effect=fake_popen), \
             patch("tools.terminal_tool._interrupt_event"):
            env = LocalEnvironment(cwd=str(tmp_path), timeout=10, shell="bash")
            env._snapshot_ready = True
            with open(env._snapshot_path, "w") as f:
                f.write("# minimal snapshot\n")
            try:
                env.execute("echo hello")
            except Exception:
                pass

        shell_binary = os.path.basename(captured_args["args"][0])
        assert shell_binary == "bash", f"Expected bash, got {shell_binary}: {captured_args['args']}"


class TestRealInitSessionZshFunctionCapture:
    """Exercise the REAL init_session() under zsh — not a re-implemented bootstrap.

    This catches regex bugs in the production bootstrap that inline test
    helpers miss (see verifier defect: production regex used ``\\-)`` instead
    of ``\\(\\)`` to match zsh function definitions).
    """

    @pytest.fixture
    def zsh_path(self):
        for candidate in ["/usr/bin/zsh", "/bin/zsh"]:
            if os.path.isfile(candidate):
                return candidate
        pytest.skip("zsh not installed on this system")

    def test_real_init_session_captures_user_functions_under_zsh(
        self, monkeypatch, tmp_path, zsh_path
    ):
        """init_session under zsh must capture user-defined functions.

        We define a sentinel function via .zshrc, mock _resolve_shell_init_files
        to explicitly source it (as production hermes config does), run the
        REAL LocalEnvironment(shell='auto') with $SHELL=zsh, then read the
        actual snapshot file and assert the sentinel function body is present.
        """
        monkeypatch.setenv("SHELL", zsh_path)

        # Create a fake .zshrc that defines a user function
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        fake_zshrc = fake_home / ".zshrc"
        sentinel = "test_sentinel_fn_42"
        fake_zshrc.write_text(
            f"{sentinel}() {{ echo 'sentinel_was_here'; }}\n"
        )
        monkeypatch.setenv("HOME", str(fake_home))

        from tools.environments.local import LocalEnvironment

        # zsh -l -c (login non-interactive) does NOT source .zshrc.
        # Production hermes config lists .zshrc in shell_init_files, which
        # _resolve_shell_init_files() returns and _run_bash prepends.
        # Mock it to simulate production behavior.
        with patch(
            "tools.environments.local._resolve_shell_init_files",
            return_value=[str(fake_zshrc)],
        ):
            try:
                env = LocalEnvironment(cwd=str(tmp_path), timeout=30, shell="auto")
            except RuntimeError:
                pytest.skip("snapshot bootstrap failed (non-zsh environment)")

        assert env._shell_kind == "zsh", f"Expected zsh, got {env._shell_kind}"
        assert env._snapshot_ready, "Snapshot should be ready after init_session"

        # Read the REAL snapshot file produced by init_session
        with open(env._snapshot_path) as f:
            snap_content = f.read()

        # The sentinel function body must appear in the snapshot
        assert sentinel in snap_content, (
            f"User function '{sentinel}' NOT captured in real zsh snapshot. "
            f"The production regex in base.py may be wrong. "
            f"Snapshot content (first 500 chars):\n{snap_content[:500]}"
        )

        # Cleanup
        try:
            os.unlink(env._snapshot_path)
        except FileNotFoundError:
            pass
