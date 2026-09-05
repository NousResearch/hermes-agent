"""Truncation footers tell the AGENT where the full text lives, and the agent's read_file runs inside
the active terminal backend: under docker the mounted cache is at /root/.hermes, so a host path in the
footer is unreadable from the sandbox (#72389, #81984, #77015)."""

import os
import re

import tools.browser_tool_snapshot as browser_tool_snapshot
import tools.delegate_tool_results as delegate_tool_results
import tools.web_tools_truncate as web_tools_truncate


def _footer_path(text: str) -> str:
    """Every truncation footer tells the agent how to page the full text via ``read_file path="..."``."""
    return re.search(r'read_file path="([^"]+)"', text).group(1)


def test_truncation_footers_render_the_sandbox_visible_cache_path(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    body = "\n".join(f"row {i}" for i in range(5000))

    web_out, _ = web_tools_truncate._truncate_with_footer(body, "https://example.com/doc", 3000)
    snap_out = browser_tool_snapshot._truncate_snapshot(body, 3000)
    deleg_out, _ = delegate_tool_results._trim_summary_with_footer(body, 3000, 0)

    for text, subdir in ((web_out, "cache/web"), (snap_out, "cache/web"), (deleg_out, "cache/delegation")):
        path = _footer_path(text)
        assert path.startswith(f"/root/.hermes/{subdir}/"), path
        # The bytes still live on the host under HERMES_HOME; only the rendered path is translated.
        assert os.path.exists(str(home / subdir / os.path.basename(path)))


def test_local_backend_footer_keeps_the_host_path(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    body = "\n".join(f"row {i}" for i in range(5000))
    out, _ = web_tools_truncate._truncate_with_footer(body, "https://example.com/doc", 3000)
    assert _footer_path(out).startswith(str(home))
