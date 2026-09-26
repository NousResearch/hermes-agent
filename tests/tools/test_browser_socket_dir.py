"""Tests for the agent-browser socket dir budget in ``_prepare_session_socket_dir`` —
the dir layout ``<root>/agent-browser-<session>/<session>.sock`` must keep the final
socket path under agent-browser's 103-byte cap (#122786)."""

from tools import browser_tool_session as session_mod


def _capture_root(monkeypatch, tmp_path):
    """Point ``_socket_safe_tmpdir`` at a controlled root, recording the requested budget."""
    seen = []

    def fake(max_len=None):
        seen.append(max_len)
        return str(tmp_path)

    monkeypatch.setattr("tools.browser_tool._socket_safe_tmpdir", fake)
    monkeypatch.setattr(
        "tools.browser_tool_lifecycle._write_owner_pid", lambda *_a: None
    )
    return seen


class TestPrepareSessionSocketDir:
    def test_short_local_session_keeps_default_budget(self, monkeypatch, tmp_path):
        seen = _capture_root(monkeypatch, tmp_path)
        socket_dir = session_mod._prepare_session_socket_dir("h_ab12cd34ef")
        assert socket_dir == str(tmp_path / "agent-browser-h_ab12cd34ef")
        # 103 - 1 - len("agent-browser-h_ab12cd34ef/h_ab12cd34ef.sock") = 58
        assert seen == [58]

    def test_long_cloud_session_name_requests_tight_budget(self, monkeypatch, tmp_path):
        seen = _capture_root(monkeypatch, tmp_path)
        # plugins/browser/_common.py:: _session_name -> f"hermes_{task_id}_{hex8}" (49 chars)
        name = "hermes_20260925_163836_07df261a_38488bd0_ab12cd34"
        session_mod._prepare_session_socket_dir(name)
        suffix = f"agent-browser-{name}/{name}.sock"
        assert seen == [103 - 1 - len(suffix)]

    def test_budget_keeps_final_socket_path_within_cap(self, monkeypatch, tmp_path):
        """Whatever root the helper picks, the daemon socket path fits the cap: local
        sessions bind ``<session>.sock`` next to their dir; a long cloud name falls back
        to the short /tmp root and binds the short CLI-default ``default.sock`` there."""
        import hermes_constants
        import shutil
        import tempfile

        monkeypatch.setattr(hermes_constants.sys, "platform", "linux")
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        names = (
            "h_ab12cd34ef",
            "hermes-real-profile",
            "hermes_20260925_163836_07df261a_38488bd0_ab12cd34",
        )
        try:
            for name in names:
                socket_dir = session_mod._prepare_session_socket_dir(name)
                sock = "default.sock" if name.startswith("hermes_") else f"{name}.sock"
                final = len(socket_dir) + 1 + len(sock)
                assert final <= session_mod._AGENT_BROWSER_SOCKET_PATH_CAP, (
                    name,
                    final,
                )
        finally:
            for name in names:
                shutil.rmtree(f"/tmp/agent-browser-{name}", ignore_errors=True)

    def test_dir_created_owner_only(self, monkeypatch, tmp_path):
        import os

        _capture_root(monkeypatch, tmp_path)
        socket_dir = session_mod._prepare_session_socket_dir("h_ab12cd34ef")
        assert os.path.isdir(socket_dir)
        assert (os.stat(socket_dir).st_mode & 0o777) == 0o700

    def test_owner_pid_claimed_under_session_name(self, monkeypatch, tmp_path):
        claims = []
        monkeypatch.setattr(
            "tools.browser_tool._socket_safe_tmpdir", lambda max_len=None: str(tmp_path)
        )
        monkeypatch.setattr(
            "tools.browser_tool_lifecycle._write_owner_pid",
            lambda dirpath, name: claims.append((dirpath, name)),
        )
        session_mod._prepare_session_socket_dir("h_ab12cd34ef")
        assert claims == [
            (str(tmp_path / "agent-browser-h_ab12cd34ef"), "h_ab12cd34ef")
        ]
