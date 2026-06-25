"""Regression: tui_gateway subprocess I/O must decode child output as UTF-8,
not the OS locale codepage. On non-UTF-8 Windows consoles (e.g. cp950) the
default text-mode decode raises UnicodeDecodeError in the subprocess reader
threads, killing them and stalling the gateway on a fixed cadence (#52649).
"""

from unittest.mock import MagicMock, patch


def test_git_branch_lookup_decodes_as_utf8_replace():
    import tui_gateway.server as srv

    captured = []

    def fake_run(cmd, **kwargs):
        captured.append(kwargs)
        m = MagicMock()
        m.returncode = 0
        m.stdout = "main"
        return m

    with patch.object(srv.subprocess, "run", side_effect=fake_run):
        srv._git_branch_for_cwd("/some/repo")

    assert captured, "expected subprocess.run to be called"
    for kwargs in captured:
        assert kwargs.get("encoding") == "utf-8"
        assert kwargs.get("errors") == "replace"
