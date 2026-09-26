"""Regression tests: the browser-use own-tab preamble must create its tab in
the background so daemon startup never steals desktop focus.

The preamble is a code template (executed inside the browser-use daemon),
so these tests exercise the generated-code contract: exec the template with
recording stubs and assert the CDP call shape. Sibling of PR #82482, which
fixed the same bug class on the CDP backend paths (tools/browser_supervisor.py,
tools/browser_cdp_tool.py).
"""

import os as _os


def _marker_path(tmp_path):
    """Mirror the preamble's marker path (uid-keyed; uid is absent on Windows)."""
    import tempfile as _tf

    try:
        from browser_harness import _ipc as _bipc

        dpid = _bipc.pid_path("default").read_text().strip() or "0"
    except Exception:
        dpid = "0"
    uid = _os.getuid() if hasattr(_os, "getuid") else 0
    return _os.path.join(_tf.gettempdir(), f"hermes-bu-owntab-{uid}-default-{dpid}")


def _run_preamble(cdp_calls, switch_calls):
    """Exec _OWN_TAB_PREAMBLE with recording stubs."""
    import types

    import tools.browser_use_cli as cli

    def fake_cdp(method, **kwargs):
        cdp_calls.append((method, kwargs))
        return {"targetId": "synthetic-target"}

    def fake_switch(tab_id):
        switch_calls.append(tab_id)

    stub = types.SimpleNamespace(cdp=fake_cdp, switch_tab=fake_switch)
    exec(compile(cli._OWN_TAB_PREAMBLE, "<preamble>", "exec"), stub.__dict__)


def test_preamble_creates_target_in_background(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    cdp_calls, switch_calls = [], []
    _run_preamble(cdp_calls, switch_calls)

    creates = [kw for method, kw in cdp_calls if method == "Target.createTarget"]
    assert len(creates) == 1
    assert creates[0].get("background") is True
    assert creates[0].get("url") == "about:blank"
    # The daemon must also pin itself to the new tab.
    assert switch_calls == ["synthetic-target"]


def test_preamble_skips_when_marker_exists(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    with open(_marker_path(tmp_path), "w"):
        pass

    cdp_calls, switch_calls = [], []
    _run_preamble(cdp_calls, switch_calls)

    assert cdp_calls == [] and switch_calls == []


def test_preamble_skips_cdp_when_create_fails(tmp_path, monkeypatch):
    """Best-effort: a failed create must still leave the marker, so a daemon
    restart retries but a live daemon does not hammer CDP every exec."""
    import types

    import tools.browser_use_cli as cli

    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))

    def failing_cdp(method, **kwargs):
        raise ConnectionError("cdp down")

    stub = types.SimpleNamespace(cdp=failing_cdp, switch_tab=lambda t: None)
    exec(compile(cli._OWN_TAB_PREAMBLE, "<preamble>", "exec"), stub.__dict__)

    assert _os.path.exists(_marker_path(tmp_path))


def test_preamble_source_has_background_kwarg():
    """Static contract: the fix lives in the template text."""
    import tools.browser_use_cli as cli

    assert (
        'cdp("Target.createTarget", url="about:blank", background=True)'
        in cli._OWN_TAB_PREAMBLE
    )
