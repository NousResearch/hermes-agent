"""Live Chrome's own-tab step fails closed (``browser_use_live_chrome.LIVE_OWN_TAB_PREAMBLE``).

On the user's live Chrome the page the harness attaches to first may be one of their signed-in tabs, so a
call that can't prove it is in a tab its daemon opened must stop before the model's code runs, and must not
record the session as pinned. The preamble runs inside the harness, so these tests run it with stand-ins for
the harness helpers it calls (``cdp``, ``switch_tab``, ``current_tab``).
"""
import json
import os
import sys
import types

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
        self.create_params = params
        return self._create()

    def switch_tab(self, target):
        self.calls.append("switch_tab")
        if self._switch_fails:
            raise RuntimeError("attach failed")
        self.attached = self._stays_on or target

    def current_tab(self):
        return {"targetId": self.attached}


@pytest.fixture(autouse=True)
def harness_pid_file(tmp_path, monkeypatch):
    """The harness's ``_ipc.pid_path`` pointed at a tmp file (absent until a test writes it), so no test
    reads a real daemon's pid or imports the real harness (which creates its dirs)."""
    pid_file = tmp_path / "harness-runtime" / "bu.pid"
    pid_file.parent.mkdir()
    ipc = types.ModuleType("browser_harness._ipc")
    ipc.pid_path = lambda name: pid_file
    pkg = types.ModuleType("browser_harness")
    pkg._ipc = ipc
    monkeypatch.setitem(sys.modules, "browser_harness", pkg)
    monkeypatch.setitem(sys.modules, "browser_harness._ipc", ipc)
    return pid_file


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



def test_tab_is_created_in_the_background(run):
    harness, *_ = run(_Harness())
    assert harness.create_params == {"url": "about:blank", "background": True}


def test_marker_records_the_tab_id(run, tmp_path):
    run(_Harness())
    (marker,) = tmp_path.glob("hermes-bu-owntab-*")
    assert marker.read_text() == "own-tab"


@pytest.mark.parametrize("pid_text", ['{"pid": 4242, "started": 1.5}', "4242", "4242\n"],
                         ids=["json", "bare", "bare-newline"])
def test_marker_name_uses_only_the_parsed_pid(run, harness_pid_file, pid_text):
    """The pid file is JSON on current harnesses: its raw text must never reach the file name (braces,
    quotes and colons are not valid in Windows file names)."""
    harness_pid_file.write_text(pid_text)
    _, _, markers, error = run(_Harness())
    assert error is None
    assert markers and markers[0].endswith("-_live_default-4242")
    assert not set(markers[0]) & set('{}":')


class TestCloseOwnTab:
    """``LIVE_CLOSE_OWN_TAB_CODE``: stopping a live daemon closes only the tab its preamble recorded."""

    @pytest.fixture
    def close(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
        monkeypatch.setenv("BU_NAME", "_live_default")

        def _close():
            calls = []

            def cdp(method, **params):
                calls.append((method, params))
            exec(compile(live.LIVE_CLOSE_OWN_TAB_CODE, "<close>", "exec"), {"cdp": cdp})
            return calls, sorted(tmp_path.glob("hermes-bu-owntab-*"))
        return _close

    def test_closes_exactly_the_recorded_tab_and_drops_the_marker(self, run, close):
        run(_Harness(create=lambda: {"targetId": "A1B2C3D4"}))
        calls, markers = close()
        assert calls == [("Target.closeTarget", {"targetId": "A1B2C3D4"})]
        assert markers == []

    def test_nothing_recorded_closes_nothing(self, close):
        assert close() == ([], [])

    @pytest.mark.parametrize("content", ["", "own-tab; Target.closeTarget", "../x", "a b"])
    def test_malformed_marker_closes_nothing(self, close, tmp_path, content):
        uid = os.getuid() if hasattr(os, "getuid") else 0
        (tmp_path / f"hermes-bu-owntab-{uid}-_live_default-0").write_text(content)
        calls, markers = close()
        assert calls == [] and markers == []

    def test_close_failure_is_swallowed(self, tmp_path, monkeypatch, run):
        run(_Harness())

        def cdp(method, **params):
            raise RuntimeError("tab already gone")
        exec(compile(live.LIVE_CLOSE_OWN_TAB_CODE, "<close>", "exec"), {"cdp": cdp})
        assert not list(tmp_path.glob("hermes-bu-owntab-*"))


class TestStopClosesTheTab:
    """_reload_daemons (live off, backend swap, process cleanup) closes the live daemon's own tab first."""

    @pytest.fixture
    def runs(self, monkeypatch):
        runs = []
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: ["bh"])
        monkeypatch.setattr(bu_cli, "_base_subprocess_env", dict)
        monkeypatch.setattr(bu_cli, "_run_cli_killing_process_group",
                            lambda cmd, code, env, timeout: runs.append((list(cmd), code, env["BU_NAME"], env)))
        monkeypatch.setattr(bu_cli, "_live_daemon_running", lambda env: True)
        return runs

    def test_live_daemon_tab_closed_before_reload(self, runs):
        bu_cli._reload_daemons([("", "", "", "", "_live_r7k2")])
        assert [(c[-1:], code, name) for c, code, name, _env in runs] == [
            (["bh"][-1:], live.LIVE_CLOSE_OWN_TAB_CODE, "_live_r7k2"), (["--reload"], "", "_live_r7k2")]

    def test_close_run_never_restarts_the_daemon(self, runs):
        """Without BH_REQUIRE_EXISTING_DAEMON the harness CLI runs ensure_daemon() first, which can restart
        an unhealthy daemon (Chrome launch / chrome://inspect / a new Allow popup)."""
        bu_cli._reload_daemons([("", "", "", "", "_live_default")])
        (_c, _code, _n, close_env), (_c2, _code2, _n2, reload_env) = runs
        assert close_env["BH_REQUIRE_EXISTING_DAEMON"] == "1"
        assert "BH_REQUIRE_EXISTING_DAEMON" not in reload_env

    def test_other_daemons_are_not_touched(self, runs):
        bu_cli._reload_daemons([("", "", "", "", "default"), ("", "", "", "", "_local_default")])
        assert [code for _c, code, _n, _e in runs] == ["", ""]  # only --reload

    def test_dead_daemon_gets_no_cli_run_that_could_spawn_one(self, runs, monkeypatch):
        monkeypatch.setattr(bu_cli, "_live_daemon_running", lambda env: False)
        bu_cli._reload_daemons([("", "", "", "", "_live_default")])
        assert [code for _c, code, _n, _e in runs] == [""]

    def test_close_error_does_not_block_the_reload(self, monkeypatch):
        seen = []

        def run_cli(cmd, code, env, timeout):
            seen.append(code)
            if code:
                raise OSError("boom")
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: ["bh"])
        monkeypatch.setattr(bu_cli, "_base_subprocess_env", dict)
        monkeypatch.setattr(bu_cli, "_run_cli_killing_process_group", run_cli)
        monkeypatch.setattr(bu_cli, "_live_daemon_running", lambda env: True)
        bu_cli._reload_daemons([("", "", "", "", "_live_default")])
        assert seen == [live.LIVE_CLOSE_OWN_TAB_CODE, ""]


class TestLiveDaemonRunning:
    def test_no_pid_file(self, monkeypatch):
        monkeypatch.setattr(bu_cli, "_published_daemon_pid", lambda env: None)
        assert bu_cli._live_daemon_running({}) is False

    def test_recycled_pid_is_not_the_daemon(self, monkeypatch):
        monkeypatch.setattr(bu_cli, "_published_daemon_pid", lambda env: os.getpid())  # this pytest process
        assert bu_cli._live_daemon_running({}) is False

    def test_no_psutil(self, monkeypatch):
        monkeypatch.setattr(bu_cli, "_published_daemon_pid", lambda env: os.getpid())
        monkeypatch.setitem(sys.modules, "psutil", None)
        assert bu_cli._live_daemon_running({}) is False
