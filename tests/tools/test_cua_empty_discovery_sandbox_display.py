"""_empty_discovery_reason must diagnose the SANDBOX's display, not the host gateway process's.

Live finding (Wian's Docker-sandboxed Bot Desktop, 2026-10-09): a headless gateway host legitimately
has no DISPLAY in its own os.environ by design (that is why the desktop runs in a sandbox at all);
checking os.environ there for a sandboxed desktop misdiagnoses a healthy, manually-confirmed-working
sandbox screen as "no DISPLAY is set" whenever window discovery happens to come back empty for any
reason, masking the real cause the operator actually needs to see.
"""
from __future__ import annotations

import pytest

from tools.computer_use import cua_backend as cb


def test_gateway_placement_still_reads_the_host_environ(monkeypatch):
    """Non-sandboxed desktops (the common case): behaviour is unchanged."""
    from tools.bot_desktop import placement
    monkeypatch.setattr(placement, "_setting", lambda: "gateway")
    monkeypatch.delenv("DISPLAY", raising=False)
    assert cb._effective_display() == ""
    monkeypatch.setenv("DISPLAY", ":1")
    assert cb._effective_display() == ":1"


def test_sandboxed_placement_reads_the_sandbox_display_not_the_host(monkeypatch):
    """The bug: a sandboxed desktop must be judged by ITS OWN published DISPLAY, never the host's
    (empty) os.environ."""
    from tools.bot_desktop import placement, runtime as bd_runtime
    monkeypatch.setattr(placement, "_setting", lambda: "terminal")
    monkeypatch.setattr(placement, "_terminal_backend", lambda: "docker")
    monkeypatch.setattr(bd_runtime, "tool_placement", lambda: placement.TERMINAL)
    monkeypatch.setattr(bd_runtime, "published_env", lambda: {"DISPLAY": ":20", "XAUTHORITY": "/x"})
    monkeypatch.delenv("DISPLAY", raising=False)  # the host genuinely has none -- must not matter here

    assert cb._effective_display() == ":20"


@pytest.mark.platforms("linux")
def test_empty_discovery_no_longer_blames_a_present_sandbox_display(monkeypatch):
    """Direct regression test for the misdiagnosis: with the sandbox's DISPLAY confirmed present,
    _empty_discovery_reason must NOT claim 'no DISPLAY is set' (that claim would be false)."""
    from tools.bot_desktop import placement, runtime as bd_runtime
    monkeypatch.setattr(bd_runtime, "tool_placement", lambda: placement.TERMINAL)
    monkeypatch.setattr(bd_runtime, "published_env", lambda: {"DISPLAY": ":20"})
    monkeypatch.setattr(cb, "_linux_session_locked", lambda: False)
    monkeypatch.delenv("DISPLAY", raising=False)

    reason = cb._empty_discovery_reason()
    assert "no DISPLAY is set" not in reason


@pytest.mark.platforms("linux")
def test_empty_discovery_still_blames_a_genuinely_missing_sandbox_display(monkeypatch):
    """Non-regression: when the sandbox's own DISPLAY really is gone, the diagnosis must still fire."""
    from tools.bot_desktop import placement, runtime as bd_runtime
    monkeypatch.setattr(bd_runtime, "tool_placement", lambda: placement.TERMINAL)
    monkeypatch.setattr(bd_runtime, "published_env", lambda: {})
    monkeypatch.setattr(cb, "_linux_session_locked", lambda: False)
    monkeypatch.setenv("DISPLAY", ":1")  # host happens to have one -- must not matter here

    reason = cb._empty_discovery_reason()
    assert "no DISPLAY is set" in reason


def test_effective_display_fails_open_to_host_environ_on_placement_error(monkeypatch):
    """A placement-resolution failure must not crash the diagnostic path -- fall back to the plain
    host check rather than raising."""
    from tools.bot_desktop import runtime as bd_runtime

    def _boom():
        raise RuntimeError("placement probe failed")
    monkeypatch.setattr(bd_runtime, "tool_placement", _boom)
    monkeypatch.setenv("DISPLAY", ":7")

    assert cb._effective_display() == ":7"
