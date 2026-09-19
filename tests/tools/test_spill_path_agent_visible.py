"""Regression for the terminal/execute_code spill-path class (#72389 family).

The truncation footers tell the agent to page the spilled output with
``read_file``/``search_files`` — tools that run INSIDE the active backend.
Two things must hold or the handle dangles on every remote backend:

1. The spill dirs (``cache/terminal-output``, ``cache/exec``) must be in
   ``credential_files._CACHE_DIRS`` so docker/modal bind-mount them and
   ssh/daytona/vercel sync them.
2. The footer path must be rendered through ``to_agent_visible_cache_path``
   (the host path is meaningless inside a container).
"""

from pathlib import Path

import pytest


@pytest.fixture
def small_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    import tools.tool_output_limits as lim
    monkeypatch.setattr(lim, "_cached_limits", {
        "max_bytes": 2000, "max_lines": 2000, "max_line_length": 2000,
    })
    return tmp_path


class TestSpillDirsMounted:
    def test_terminal_and_exec_spill_dirs_are_in_cache_mounts(self, tmp_path, monkeypatch):
        """Both spill dirs the footers point at must ride the mount/sync list —
        an unmounted spill file is unreadable from any remote backend."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        from tools.credential_files import _CACHE_DIRS, get_cache_directory_mounts

        subpaths = {new for new, _old in _CACHE_DIRS}
        assert "cache/terminal-output" in subpaths
        assert "cache/exec" in subpaths

        mounts = get_cache_directory_mounts()
        mounted = {m["container_path"] for m in mounts}
        assert "/root/.hermes/cache/terminal-output" in mounted, "terminal spill dir not mounted"
        assert "/root/.hermes/cache/exec" in mounted, "execute_code spill dir not mounted"


class TestTerminalSpillPathTranslated:
    def _spilled_file(self, tmp_path):
        """A real spilled file under the temp HERMES_HOME's terminal-output dir."""
        spill_dir = tmp_path / ".hermes" / "cache" / "terminal-output"
        spill_dir.mkdir(parents=True, exist_ok=True)
        path = spill_dir / "out-1-2-abcd.log"
        path.write_text("head\n" + "w" * 90 + "\ntail", encoding="utf-8")
        return path

    def test_docker_backend_renders_container_visible_spill_path(self, small_cap, monkeypatch):
        """With TERMINAL_ENV=docker the spill handle must carry the /root/.hermes
        path the container sees, not the host path."""
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        from tools.terminal_tool_result import _redact_spill_file

        path = self._spilled_file(small_cap)
        fields = dict(_redact_spill_file(str(path), 10_000, "some command"))
        assert fields["full_output_path"].startswith("/root/.hermes/cache/terminal-output"), (
            f"spill handle not translated for the container: {fields['full_output_path']}"
        )
        assert "/root/.hermes/cache/terminal-output" in fields["truncation_note"]
        assert "host" not in fields["truncation_note"].lower() or "/root/.hermes" in fields["truncation_note"]

    def test_local_backend_keeps_host_path(self, small_cap, monkeypatch):
        """Control: on the local backend the host path is already agent-visible."""
        monkeypatch.setenv("TERMINAL_ENV", "local")
        from tools.terminal_tool_result import _redact_spill_file

        path = self._spilled_file(small_cap)
        fields = dict(_redact_spill_file(str(path), 10_000, "some command"))
        assert str(path) == fields["full_output_path"]
        assert Path(fields["full_output_path"]).exists()
class TestExecuteCodeSpillPathTranslated:
    def test_execute_code_footer_path_is_agent_visible(self, small_cap, monkeypatch):
        """execute_code's spill footer must name the path the sandbox's read_file
        can open — under a docker backend that is the /root/.hermes mirror."""
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        from tools.code_execution_tool import _truncate_stdout_text

        text, metadata = _truncate_stdout_text("x" * 60_000)
        assert metadata["stdout_truncated"] is True
        spill = metadata.get("stdout_spill_path")
        assert spill and spill.startswith("/root/.hermes"), (
            f"execute_code spill footer not translated: {spill}"
        )
        assert "/root/.hermes" in metadata["warning"]


class TestBlockedScriptPathTranslated:
    def test_parser_limit_block_points_at_backend_visible_script(self, tmp_path, monkeypatch):
        """The parser-limit recovery hint tells the agent to `bash <path>` via the
        terminal tool — under docker that must be the /root/.hermes script."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        from tools.approval_floors import _PARSER_LIMIT_DESCRIPTION, _hardline_block_result

        result = _hardline_block_result(_PARSER_LIMIT_DESCRIPTION, command="echo " + "x" * 500)
        assert result["approved"] is False
        assert 'terminal(command="bash /root/.hermes/cache/blocked-scripts/' in result["message"], (
            f"recovery hint not translated: {result['message']}"
        )

    def test_local_block_keeps_host_path(self, tmp_path, monkeypatch):
        """Control: on the local backend the host script path is runnable as-is."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setenv("TERMINAL_ENV", "local")
        from tools.approval_floors import _PARSER_LIMIT_DESCRIPTION, _hardline_block_result

        result = _hardline_block_result(_PARSER_LIMIT_DESCRIPTION, command="echo " + "x" * 500)
        assert 'bash ' + str(tmp_path / ".hermes" / "cache" / "blocked-scripts") in result["message"]


class TestHookSpillPathTranslated:
    def test_hook_spill_preview_points_at_backend_visible_path(self, tmp_path, monkeypatch):
        """The hook spill preview points the agent at the saved content — under
        docker it must carry the /root/.hermes mirror of hook_outputs."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        from tools.hook_output_spill import spill_if_oversized

        out = spill_if_oversized("h" * 20_000, session_id="sess-1", source="hook",
                                  config={"enabled": True, "max_chars": 100, "preview_head": 10})
        assert "saved to /root/.hermes/hook_outputs/" in out, f"preview not translated: {out}"

    def test_local_hook_spill_keeps_host_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setenv("TERMINAL_ENV", "local")
        from tools.hook_output_spill import spill_if_oversized

        out = spill_if_oversized("h" * 20_000, session_id="sess-1", source="hook",
                                  config={"enabled": True, "max_chars": 100, "preview_head": 10})
        assert f"saved to {tmp_path / '.hermes' / 'hook_outputs'}" in out