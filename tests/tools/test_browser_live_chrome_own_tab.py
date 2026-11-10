"""Live Chrome's own-tab step fails closed (``browser_use_live_chrome.LIVE_OWN_TAB_PREAMBLE``).

On the user's live Chrome the page the harness attaches to first may be one of their signed-in tabs, so a
call that can't prove it is in a tab its daemon opened must stop before the model's code runs, and must not
record the session as pinned. The preamble runs inside the harness, so these tests run it with stand-ins for
the harness helpers it calls (``cdp``, ``switch_tab``, ``current_tab``).
"""
import pytest

import tools.browser_use_cli as bu_cli
import tools.browser_use_live_chrome as live

_USER_CODE = "ran.append('user code')\n"


class _Harness:
    """The three harness helpers the preamble calls, attached to the user's tab ``user-tab`` at start."""

    def __init__(self, *, create=lambda: {"targetId": "own-tab"}, switch_fails=False, stays_on=None):
        self.attached = "user-tab"
        self.calls = []
        self._create, self._switch_fails, self._stays_on = create, switch_fails, stays_on

    def cdp(self, method, **params):
        self.calls.append(method)
        assert method == "Target.createTarget" and params == {"url": "about:blank"}
        return self._create()

    def switch_tab(self, target):
        self.calls.append("switch_tab")
        if self._switch_fails:
            raise RuntimeError("attach failed")
        self.attached = self._stays_on or target

    def current_tab(self):
        return {"targetId": self.attached}


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Run preamble + user code like the harness does; returns (harness, ran, marker files)."""
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    monkeypatch.setenv("BU_NAME", "_live_default")

    def _run(harness):
        ran = []
        namespace = {"cdp": harness.cdp, "switch_tab": harness.switch_tab, "current_tab": harness.current_tab,
                     "ran": ran}
        error = None
        try:
            exec(compile(live.LIVE_OWN_TAB_PREAMBLE + _USER_CODE, "<browser_exec>", "exec"), namespace)
        except RuntimeError as exc:
            error = str(exc)
        markers = sorted(p.name for p in tmp_path.glob("hermes-bu-owntab-*"))
        return harness, ran, markers, error
    return _run


def _boom():
    raise RuntimeError("Target.createTarget refused")


@pytest.mark.parametrize("harness, cause", [
    (_Harness(create=_boom), "Target.createTarget refused"),
    (_Harness(create=dict), "no id for the new tab"),
    (_Harness(create=lambda: None), "no id for the new tab"),
    (_Harness(switch_fails=True), "attach failed"),
    (_Harness(stays_on="user-tab"), "still attached to another tab"),
], ids=["create-raises", "no-target-id", "no-result", "switch-raises", "switch-did-not-move"])
def test_failure_stops_before_user_code_and_pins_nothing(run, harness, cause):
    harness, ran, markers, error = run(harness)
    assert error and error.startswith("LIVE CHROME STOPPED") and cause in error
    assert ran == []  # the model's code never ran in the user's tab
    assert markers == []  # nothing pinned: the next call tries again


def test_success_switches_to_own_tab_then_runs_and_pins(run):
    harness, ran, markers, error = run(_Harness())
    assert error is None
    assert harness.attached == "own-tab"
    assert ran == ["user code"]
    assert len(markers) == 1 and markers[0].endswith("-_live_default-0")


def test_retry_after_a_failure_tries_again(run):
    run(_Harness(create=_boom))
    harness, ran, _markers, error = run(_Harness())
    assert error is None and harness.calls == ["Target.createTarget", "switch_tab"] and ran == ["user code"]


def test_pinned_daemon_skips_the_step(run):
    run(_Harness())
    harness, ran, _markers, error = run(_Harness(create=_boom))
    assert error is None and harness.calls == [] and ran == ["user code"]


def test_preamble_choice():
    assert live.own_tab_preamble(True, bu_cli._OWN_TAB_PREAMBLE) is live.LIVE_OWN_TAB_PREAMBLE
    assert live.own_tab_preamble(False, bu_cli._OWN_TAB_PREAMBLE) is bu_cli._OWN_TAB_PREAMBLE

