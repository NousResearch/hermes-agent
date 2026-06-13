"""Tests for tools/host_home_cli.py — host-HOME PATH shims for allowlisted CLIs.

``prepare_host_home_cli_shims`` runs on EVERY local terminal subprocess env
build (`_sanitize_subprocess_env` and `_make_run_env` in
tools/environments/local.py). The happy path is covered E2E in
tests/test_subprocess_home_isolation.py; this file pins the module's edge
contracts: the self-exec recursion guard, rewrite idempotency, no-op
fallbacks, HERMES_CLAUDE_HOME precedence, and the PATH re-prepend snippet.

POSIX-only by design: the module skips shim generation on native Windows.
"""

import os
import subprocess

import pytest

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="host-home shims are POSIX-only (module no-ops on Windows)"
)


@pytest.fixture
def shim_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME + host home + a fake `claude` binary on PATH."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    host_home = tmp_path / "host-home"
    host_home.mkdir()
    real_bin = tmp_path / "bin"
    real_bin.mkdir()
    fake_claude = real_bin / "claude"
    fake_claude.write_text("#!/bin/sh\nprintf '%s\\n' \"$HOME\"\n", encoding="utf-8")
    fake_claude.chmod(0o755)

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_HOST_HOME", str(host_home))
    return {
        "hermes_home": hermes_home,
        "host_home": host_home,
        "real_bin": real_bin,
        "fake_claude": fake_claude,
    }


class TestPrepareShims:
    def test_installs_shim_and_prepends_path(self, shim_env):
        from tools.host_home_cli import prepare_host_home_cli_shims

        env = {"PATH": str(shim_env["real_bin"])}
        result = prepare_host_home_cli_shims(env)

        assert result is env  # mutated in place and returned
        assert result["HERMES_HOST_HOME"] == str(shim_env["host_home"])
        shim_dir = result["HERMES_HOST_CLI_SHIM_DIR"]
        assert result["PATH"].split(os.pathsep)[0] == shim_dir

        shim = shim_env["hermes_home"] / "cache" / "host-cli-shims" / "claude"
        assert str(shim.parent) == shim_dir
        assert os.access(shim, os.X_OK)
        # The shim execs the resolved real binary, not a bare name.
        assert str(shim_env["fake_claude"].resolve()) in shim.read_text(encoding="utf-8")

    def test_no_claude_on_path_is_safe_noop(self, shim_env, tmp_path):
        from tools.host_home_cli import prepare_host_home_cli_shims

        empty_bin = tmp_path / "empty-bin"
        empty_bin.mkdir()
        env = {"PATH": str(empty_bin)}
        result = prepare_host_home_cli_shims(env)

        # Host home is still exported for the wrapper contract, but no shim
        # dir is created or injected into PATH.
        assert result["HERMES_HOST_HOME"] == str(shim_env["host_home"])
        assert "HERMES_HOST_CLI_SHIM_DIR" not in result
        assert result["PATH"] == str(empty_bin)

    def test_unresolvable_host_home_leaves_env_untouched(self, shim_env, monkeypatch):
        from tools import host_home_cli

        monkeypatch.setattr(host_home_cli, "get_host_user_home", lambda: None)
        env = {"PATH": str(shim_env["real_bin"])}
        result = host_home_cli.prepare_host_home_cli_shims(env)
        assert result == {"PATH": str(shim_env["real_bin"])}

    def test_second_pass_does_not_resolve_shim_as_real_binary(self, shim_env):
        """Recursion guard: with the shim dir already first on PATH, the
        shim must keep pointing at the real binary — never at itself."""
        from tools.host_home_cli import prepare_host_home_cli_shims

        env = {"PATH": str(shim_env["real_bin"])}
        prepare_host_home_cli_shims(env)
        assert env["PATH"].split(os.pathsep)[0] == env["HERMES_HOST_CLI_SHIM_DIR"]

        prepare_host_home_cli_shims(env)

        shim = shim_env["hermes_home"] / "cache" / "host-cli-shims" / "claude"
        content = shim.read_text(encoding="utf-8")
        assert str(shim_env["fake_claude"].resolve()) in content
        assert str(shim) not in content
        # PATH must not accumulate duplicate shim-dir entries.
        assert env["PATH"].split(os.pathsep).count(env["HERMES_HOST_CLI_SHIM_DIR"]) == 1

    def test_unchanged_shim_is_not_rewritten(self, shim_env):
        from tools.host_home_cli import prepare_host_home_cli_shims

        env = {"PATH": str(shim_env["real_bin"])}
        prepare_host_home_cli_shims(env)
        shim = shim_env["hermes_home"] / "cache" / "host-cli-shims" / "claude"
        first_stat = shim.stat()

        prepare_host_home_cli_shims(dict(env))
        second_stat = shim.stat()
        assert second_stat.st_mtime_ns == first_stat.st_mtime_ns


class TestShimExecution:
    def test_claude_home_override_beats_host_home(self, shim_env):
        """HERMES_CLAUDE_HOME (per-CLI override) wins over HERMES_HOST_HOME."""
        from tools.host_home_cli import prepare_host_home_cli_shims

        claude_home = shim_env["hermes_home"].parent / "claude-home"
        claude_home.mkdir()
        env = {"PATH": str(shim_env["real_bin"])}
        prepare_host_home_cli_shims(env)
        env["HERMES_CLAUDE_HOME"] = str(claude_home)
        env["HOME"] = "/profile/home"

        completed = subprocess.run(
            ["claude"], capture_output=True, text=True, env=env, check=True
        )
        assert completed.stdout.strip() == str(claude_home)


class TestShellReprepend:
    def _path_after_snippet(self, snippet_env):
        from tools.host_home_cli import shell_reprepend_host_cli_shim_path

        script = shell_reprepend_host_cli_shim_path() + '\nprintf "%s" "$PATH"\n'
        completed = subprocess.run(
            ["/bin/sh", "-c", script],
            capture_output=True,
            text=True,
            env=snippet_env,
            check=True,
        )
        return completed.stdout

    def test_prepends_shim_dir_when_missing(self, tmp_path):
        shim_dir = str(tmp_path / "shims")
        path = self._path_after_snippet(
            {"PATH": "/usr/bin:/bin", "HERMES_HOST_CLI_SHIM_DIR": shim_dir}
        )
        assert path.split(os.pathsep)[0] == shim_dir

    def test_does_not_duplicate_existing_entry(self, tmp_path):
        shim_dir = str(tmp_path / "shims")
        path = self._path_after_snippet(
            {
                "PATH": f"{shim_dir}:/usr/bin:/bin",
                "HERMES_HOST_CLI_SHIM_DIR": shim_dir,
            }
        )
        assert path.split(os.pathsep).count(shim_dir) == 1

    def test_noop_when_shim_dir_unset(self):
        path = self._path_after_snippet({"PATH": "/usr/bin:/bin"})
        assert path == "/usr/bin:/bin"
