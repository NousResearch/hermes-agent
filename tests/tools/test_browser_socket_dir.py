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
    def test_short_session_budget_clamped_to_default(self, monkeypatch, tmp_path):
        seen = _capture_root(monkeypatch, tmp_path)
        socket_dir = session_mod._prepare_session_socket_dir("h_ab12cd34ef")
        assert socket_dir == str(tmp_path / "agent-browser-h_ab12cd34ef")
        # Raw budget 103 - 1 - len(suffix) = 58, clamped to SOCKET_TMPDIR_MAX_LEN = 50: a
        # short name must never LOOSEN the default budget, or a 51-58 char scratch root
        # would move local socket dirs into scratch where reaper/teardown can't find them.
        assert seen == [50]

    def test_long_cloud_session_name_requests_tight_budget(self, monkeypatch, tmp_path):
        seen = _capture_root(monkeypatch, tmp_path)
        # plugins/browser/_common.py:: _session_name -> f"hermes_{task_id}_{hex8}"
        name = "hermes_20260925_163836_07df261a_38488bd0_ab12cd34"
        session_mod._prepare_session_socket_dir(name)
        suffix = f"agent-browser-{name}/{name}.sock"
        # far below SOCKET_TMPDIR_MAX_LEN, so the clamp leaves the tight budget intact
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


class TestSessionSocketRootAgreement:
    """Producer, teardown and the reaper must agree on the socket dir root (production
    follow-up to #122786: teardown and the reaper looked under the default-budget root
    while the producer had placed long-name dirs in the /tmp fallback, so the rmtree and
    the pid-file kill silently did nothing)."""

    # Scratch root lengths reported from real fleets, including the 51-58 range where an
    # unclamped 58-char budget would diverge from the default SOCKET_TMPDIR_MAX_LEN=50.
    SCRATCH_LENGTHS = (39, 40, 41, 42, 43, 45, 46, 47, 50, 51, 57, 58)

    # Real session-name shapes: local/real-profile/cdp ``<prefix>_<hex10>``, the
    # lightpanda fallback ``h_cfb_<hex8>``, and cloud ``hermes_<task_id>_<hex8>`` where
    # subagent task ids look like ``sa-1-e74bf0ca`` — 13 chars, not 12.
    NAMES = (
        "h_ab12cd34ef",
        "rp_0011223344",
        "cdp_0011223344",
        "h_cfb_00112233",
        "hermes_sa-1-e74bf0ca_ab12cd34",                    # 13-char task id
        "hermes_sa-1-" + "a" * 17 + "_ab12cd34",            # 22-char task id
        "hermes_sa-1-" + "a" * 28 + "_ab12cd34",            # 33-char task id
    )

    def _linux_scratch(self, monkeypatch, length):
        import hermes_constants
        import tempfile

        scratch = "/" + "s" * (length - 1)
        monkeypatch.setattr(hermes_constants.sys, "platform", "linux")
        monkeypatch.setattr(tempfile, "gettempdir", lambda: scratch)

    def test_reaper_roots_cover_every_producer_root(self, monkeypatch):
        """Whatever the scratch length, each producer root is one of the two roots the
        orphan reaper scans (default-budget root or the /tmp fallback)."""
        from tools import browser_tool as bt

        for length in self.SCRATCH_LENGTHS:
            self._linux_scratch(monkeypatch, length)
            reaper_roots = {bt._socket_safe_tmpdir(), bt._socket_safe_tmpdir(max_len=0)}
            for name in self.NAMES:
                assert session_mod._session_socket_root(name) in reaper_roots, (
                    length,
                    name,
                )

    def test_matrix_keeps_socket_path_within_cap(self, monkeypatch):
        """Fleet-reported matrix of scratch lengths × session names: the composed daemon
        socket path (``<root>/agent-browser-<session>/<daemon>.sock``) always fits the
        103-byte cap. Local sessions bind ``<session>.sock``; long cloud names fall back
        to /tmp and bind the short CLI-default ``default.sock``."""
        for length in self.SCRATCH_LENGTHS:
            self._linux_scratch(monkeypatch, length)
            for name in self.NAMES:
                root = session_mod._session_socket_root(name)
                daemon_sock = "default.sock" if name.startswith("hermes_") else f"{name}.sock"
                final = len(root) + 1 + len(f"agent-browser-{name}") + 1 + len(daemon_sock)
                assert final <= session_mod._AGENT_BROWSER_SOCKET_PATH_CAP, (
                    length,
                    name,
                    final,
                )

    def test_teardown_removes_dir_from_producer_root(self, monkeypatch):
        """The 51-char scratch case from production: the clamped budget still falls back
        to /tmp, and ``_release_session_resources`` must look there too, or its rmtree
        silently no-ops and the dir leaks to the grace sweep."""
        import os
        import shutil

        from tools import browser_tool_lifecycle as lifecycle

        self._linux_scratch(monkeypatch, 51)
        monkeypatch.setattr(
            "tools.browser_tool_lifecycle._write_owner_pid", lambda *_a: None
        )
        socket_dir = session_mod._prepare_session_socket_dir("h_ab12cd34ef")
        try:
            assert socket_dir == "/tmp/agent-browser-h_ab12cd34ef"  # not under scratch
            assert os.path.isdir(socket_dir)
            lifecycle._release_session_resources(
                "t", {"session_name": "h_ab12cd34ef", "bb_session_id": None}
            )
            assert not os.path.exists(socket_dir)
        finally:
            shutil.rmtree(socket_dir, ignore_errors=True)
