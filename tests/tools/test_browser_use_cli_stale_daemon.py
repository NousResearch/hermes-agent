"""Regression tests for #101576: browser_exec must not silently hang when the
browser-use harness daemon is stranded on a stale CDP endpoint.

Failure mode being locked out: the real-profile (or any local-attach) browser
is relaunched on a NEW ephemeral port; Hermes resolves the fresh endpoint and
exports it via BU_CDP_URL — but an already-running ``browser_harness.daemon``
still holds its ORIGINAL attach endpoint (resolved at first attach, never
revalidated). The daemon's socket stays in LISTEN, every browser_exec connects
to it and blocks, and the call burns the full tool timeout with no output and
no log line.

The Hermes-side fix: before launching the CLI, check the daemon's recorded
endpoint; when it no longer answers, recycle the daemon (terminate + clear
its runtime pid/sock files) so the call respawns a fresh one — exactly the
issue's verified workaround, automated.

These tests exercise the real production path: ``browser_exec()`` with a fake
CLI, a synthetic harness runtime dir, and monkeypatched probes. No real
daemon, browser, or network is touched.
"""

import json
import stat

import pytest

import tools.browser_use_cli as bu_cli


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("BU_NAME", raising=False)
    monkeypatch.delenv("BU_AUTOSPAWN", raising=False)
    monkeypatch.delenv("BROWSER_USE_API_KEY", raising=False)
    # Real-profile consent off: these tests drive the local-attach path via
    # the CDP override, without touching browser_connect.
    monkeypatch.setattr(bu_cli, "_real_profile_consented", lambda: False)
    yield


def _fake_cli(tmp_path, body):
    """Write an executable fake browser-use CLI and return its path."""
    script = tmp_path / "browser-use"
    script.write_text("#!/bin/sh\n" + body)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


def _harness_home(monkeypatch, tmp_path):
    """Redirect the harness runtime/tmp dirs into an isolated tmp_path."""
    home = tmp_path / "bh-home"
    for sub in ("runtime", "tmp"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(bu_cli, "_HARNESS_HOME", home, raising=False)
    return home


def _plant_daemon_state(home, name="default", *, pid=None, sock=True, endpoint=None, log_extra=""):
    """Create the on-disk state a live harness daemon leaves behind."""
    runtime = home / "runtime"
    if pid is not None:
        (runtime / f"bu-{name}.pid").write_text(str(pid))
    if sock:
        (runtime / f"bu-{name}.sock").write_text("")
    if endpoint is not None:
        lines = [f"connecting to {endpoint}"]
        if log_extra:
            lines.append(log_extra)
        (home / "tmp" / f"bu-{name}.log").write_text("\n".join(lines) + "\n")


def _configure_local_attach(monkeypatch, tmp_path, cdp_url="http://127.0.0.1:36367"):
    """backend/env resolution → a local http CDP endpoint (BU_CDP_URL)."""
    import tools.browser_tool as bt

    monkeypatch.setattr(bu_cli, "_find_cli", lambda: [_fake_cli(tmp_path, "cat\necho ok\n")])
    monkeypatch.setattr(bt, "_get_cdp_override", lambda: cdp_url)
    monkeypatch.setattr(bt, "_get_cloud_provider", lambda: None)
    monkeypatch.setattr(bu_cli, "_read_browser_cfg", lambda: {})


def _track_terminations(monkeypatch):
    """Record _terminate_host_pid calls instead of signaling anything."""
    import tools.process_registry as pr

    killed = []

    def fake_terminate(pid, expected_start=None):
        killed.append(pid)

    monkeypatch.setattr(pr.ProcessRegistry, "_terminate_host_pid", fake_terminate)
    return killed


def _live_harness_daemon(monkeypatch, pid=4242):
    """Make the pid resolve as a live browser-harness daemon process."""
    import psutil

    class _Daemon:
        def name(self):
            return "python"

        def cmdline(self):
            return ["python", "-m", "browser_harness.daemon"]

    monkeypatch.setattr(psutil, "Process", lambda p: _Daemon() if p == pid else (_ for _ in ()).throw(psutil.NoSuchProcess(p)))


class TestStrandedDaemonRecycle:
    """The daemon recycle hook inside browser_exec (#101576)."""

    def test_stranded_daemon_recycled_before_cli_launch(self, tmp_path, monkeypatch):
        """L1+L2: dead recorded endpoint + live daemon pid → daemon terminated,
        runtime pid/sock cleared, CLI still runs and succeeds. Covers both
        variants: the daemon latched onto the OLD port (36159) while Hermes
        resolved a new one, and the same-port relaunch race where Hermes
        resolved the reused port but the new browser has not opened it yet —
        either way a dead recorded endpoint means the daemon can never work
        again, so it is recycled."""
        home = _harness_home(monkeypatch, tmp_path)
        _plant_daemon_state(
            home, pid=4242, endpoint="ws://127.0.0.1:36159/devtools/browser/abc"
        )
        killed = _track_terminations(monkeypatch)
        _live_harness_daemon(monkeypatch, pid=4242)
        monkeypatch.setattr(bu_cli, "_cdp_endpoint_ready", lambda url: False)
        _configure_local_attach(monkeypatch, tmp_path)

        result = json.loads(bu_cli.browser_exec("print('payload')"))
        assert result["success"] is True
        assert killed == [4242], "stranded daemon must be terminated"
        assert not (home / "runtime" / "bu-default.pid").exists()
        assert not (home / "runtime" / "bu-default.sock").exists()

        # Same-port relaunch race: daemon recorded the port this call also
        # resolved, but the endpoint is dead — still recycled (a daemon
        # attached to a dead browser can never recover).
        home2 = _harness_home(monkeypatch, tmp_path)
        _plant_daemon_state(
            home2, pid=4343, endpoint="http://127.0.0.1:36367"
        )
        killed2 = _track_terminations(monkeypatch)
        _live_harness_daemon(monkeypatch, pid=4343)
        monkeypatch.setattr(bu_cli, "_cdp_endpoint_ready", lambda url: False)
        _configure_local_attach(monkeypatch, tmp_path, "http://127.0.0.1:36367")

        result2 = json.loads(bu_cli.browser_exec("print('payload')"))
        assert result2["success"] is True
        assert killed2 == [4343]
        assert not (home2 / "runtime" / "bu-default.pid").exists()

    def test_healthy_daemon_untouched(self, tmp_path, monkeypatch):
        """L7: recorded endpoint answers and matches the resolved one → no
        signal, no unlink."""
        home = _harness_home(monkeypatch, tmp_path)
        _plant_daemon_state(
            home, pid=4242, endpoint="http://127.0.0.1:36367"
        )
        killed = _track_terminations(monkeypatch)

        def boom(pid, expected_start=None):
            raise AssertionError("healthy daemon must not be terminated")

        monkeypatch.setattr(bu_cli, "_cdp_endpoint_ready", lambda url: True)
        import tools.process_registry as pr

        monkeypatch.setattr(pr.ProcessRegistry, "_terminate_host_pid", boom)
        _configure_local_attach(monkeypatch, tmp_path)

        result = json.loads(bu_cli.browser_exec("print('payload')"))
        assert result["success"] is True
        assert killed == []
        assert (home / "runtime" / "bu-default.pid").exists()
        assert (home / "runtime" / "bu-default.sock").exists()

    def test_alive_but_mismatched_endpoint_recycles(self, tmp_path, monkeypatch):
        """Issue fix 1: the daemon latched onto a DIFFERENT live loopback port
        than the one this call resolved — it would drive the wrong browser
        even though its endpoint answers. Recycle it."""
        home = _harness_home(monkeypatch, tmp_path)
        _plant_daemon_state(
            home, pid=4242, endpoint="ws://127.0.0.1:36159/devtools/browser/abc"
        )
        killed = _track_terminations(monkeypatch)
        _live_harness_daemon(monkeypatch, pid=4242)
        monkeypatch.setattr(bu_cli, "_cdp_endpoint_ready", lambda url: True)
        _configure_local_attach(monkeypatch, tmp_path, "http://127.0.0.1:36367")

        result = json.loads(bu_cli.browser_exec("print('x')"))
        assert result["success"] is True
        assert killed == [4242]
        assert not (home / "runtime" / "bu-default.pid").exists()

    def test_no_resolved_endpoint_alive_daemon_untouched(self, tmp_path, monkeypatch):
        """No BU_CDP_* resolved (plain local Chrome): the harness manages its
        own attach, so a live recorded endpoint means a healthy daemon — even
        though there is nothing to match against."""
        home = _harness_home(monkeypatch, tmp_path)
        _plant_daemon_state(
            home, pid=4242, endpoint="http://127.0.0.1:9223"
        )
        killed = _track_terminations(monkeypatch)

        def boom(pid, expected_start=None):
            raise AssertionError("self-managed healthy daemon must not be terminated")

        import tools.browser_tool as bt
        import tools.process_registry as pr

        monkeypatch.setattr(bt, "_get_cdp_override", lambda: "")
        monkeypatch.setattr(bt, "_get_cloud_provider", lambda: None)
        monkeypatch.setattr(bu_cli, "_read_browser_cfg", lambda: {})
        monkeypatch.setattr(pr.ProcessRegistry, "_terminate_host_pid", boom)
        monkeypatch.setattr(bu_cli, "_cdp_endpoint_ready", lambda url: True)
        monkeypatch.setattr(
            bu_cli, "_find_cli", lambda: [_fake_cli(tmp_path, "cat\necho ok\n")]
        )

        result = json.loads(bu_cli.browser_exec("print('x')"))
        assert result["success"] is True
        assert killed == []
        assert (home / "runtime" / "bu-default.pid").exists()

    def test_missing_pid_file_is_noop(self, tmp_path, monkeypatch):
        """L6: no pid file → fresh-daemon path unchanged."""
        home = _harness_home(monkeypatch, tmp_path)
        _plant_daemon_state(
            home, pid=None, endpoint="ws://127.0.0.1:36159/devtools/browser/abc"
        )
        killed = _track_terminations(monkeypatch)
        probed = []
        monkeypatch.setattr(
            bu_cli, "_cdp_endpoint_ready", lambda url: probed.append(url) or False
        )
        _configure_local_attach(monkeypatch, tmp_path)

        result = json.loads(bu_cli.browser_exec("print('x')"))
        assert result["success"] is True
        assert killed == []
        assert probed == [], "no pid file → the endpoint probe must not even run"

    def test_no_connecting_line_is_noop(self, tmp_path, monkeypatch):
        """L6: daemon log without a recorded endpoint → nothing to validate."""
        home = _harness_home(monkeypatch, tmp_path)
        _plant_daemon_state(home, pid=4242, endpoint=None, log_extra="listening on x")
        killed = _track_terminations(monkeypatch)
        probed = []
        monkeypatch.setattr(
            bu_cli, "_cdp_endpoint_ready", lambda url: probed.append(url) or False
        )
        _configure_local_attach(monkeypatch, tmp_path)

        result = json.loads(bu_cli.browser_exec("print('x')"))
        assert result["success"] is True
        assert killed == []
        assert probed == []

    def test_named_session_recycles_its_own_daemon(self, tmp_path, monkeypatch):
        """L3: BU_NAME namespaces the daemon state — session=r7k2 must recycle
        bu-r7k2.* and leave bu-default.* alone."""
        home = _harness_home(monkeypatch, tmp_path)
        _plant_daemon_state(home, "r7k2", pid=5151, endpoint="ws://127.0.0.1:36159/x")
        _plant_daemon_state(home, "default", pid=4242, endpoint="ws://127.0.0.1:36159/x")
        killed = _track_terminations(monkeypatch)
        _live_harness_daemon(monkeypatch, pid=5151)
        monkeypatch.setattr(bu_cli, "_cdp_endpoint_ready", lambda url: False)
        _configure_local_attach(monkeypatch, tmp_path)

        result = json.loads(bu_cli.browser_exec("print('x')", session="r7k2"))
        assert result["success"] is True
        assert killed == [5151]
        assert not (home / "runtime" / "bu-r7k2.pid").exists()
        assert not (home / "runtime" / "bu-r7k2.sock").exists()
        assert (home / "runtime" / "bu-default.pid").exists()
        assert (home / "runtime" / "bu-default.sock").exists()

    def test_recycled_pid_stranger_is_not_signaled(self, tmp_path, monkeypatch):
        """L4: pid file pointing at a live NON-harness process → refuse to
        signal AND leave the state files alone (fail-closed, reaper posture)."""
        import psutil

        home = _harness_home(monkeypatch, tmp_path)
        _plant_daemon_state(
            home, pid=4242, endpoint="ws://127.0.0.1:36159/devtools/browser/abc"
        )
        killed = _track_terminations(monkeypatch)
        monkeypatch.setattr(bu_cli, "_cdp_endpoint_ready", lambda url: False)

        class _Stranger:
            def __init__(self, pid):
                self._pid = pid

            def name(self):
                return "someapp"

            def cmdline(self):
                return ["/usr/bin/someapp", "--serve"]

        monkeypatch.setattr(psutil, "Process", lambda pid: _Stranger(pid))
        _configure_local_attach(monkeypatch, tmp_path)

        result = json.loads(bu_cli.browser_exec("print('x')"))
        assert result["success"] is True
        assert killed == [], "a stranger process must never be signaled"
        assert (home / "runtime" / "bu-default.pid").exists()

    def test_dead_pid_clears_state_without_signaling(self, tmp_path, monkeypatch):
        """L4 edge: recorded pid already gone → nothing to signal, but the
        stale runtime files are cleared so the next call respawns clean."""
        import psutil

        home = _harness_home(monkeypatch, tmp_path)
        _plant_daemon_state(
            home, pid=4242, endpoint="ws://127.0.0.1:36159/devtools/browser/abc"
        )
        killed = _track_terminations(monkeypatch)
        monkeypatch.setattr(bu_cli, "_cdp_endpoint_ready", lambda url: False)
        monkeypatch.setattr(
            psutil,
            "Process",
            lambda pid: (_ for _ in ()).throw(psutil.NoSuchProcess(pid)),
        )
        _configure_local_attach(monkeypatch, tmp_path)

        result = json.loads(bu_cli.browser_exec("print('x')"))
        assert result["success"] is True
        assert killed == []
        assert not (home / "runtime" / "bu-default.pid").exists()
        assert not (home / "runtime" / "bu-default.sock").exists()

    def test_remote_endpoint_skips_probe(self, tmp_path, monkeypatch):
        """L5: cloud/remote recorded endpoints are long-lived provider URLs,
        not ephemeral local ports — recycle logic must not apply."""
        home = _harness_home(monkeypatch, tmp_path)
        _plant_daemon_state(home, pid=4242, endpoint="wss://browser.example/cdp/abc")
        killed = _track_terminations(monkeypatch)
        probed = []
        monkeypatch.setattr(
            bu_cli, "_cdp_endpoint_ready", lambda url: probed.append(url) or False
        )
        _configure_local_attach(monkeypatch, tmp_path)

        result = json.loads(bu_cli.browser_exec("print('x')"))
        assert result["success"] is True
        assert killed == []
        assert probed == []
        assert (home / "runtime" / "bu-default.pid").exists()

    def test_unreadable_log_is_noop(self, tmp_path, monkeypatch):
        """L6: a daemon log that cannot be read → no crash, no recycle."""
        home = _harness_home(monkeypatch, tmp_path)
        _plant_daemon_state(home, pid=4242, endpoint=None)
        log = home / "tmp" / "bu-default.log"
        log.write_text("listening on something\n")
        log.chmod(0o000)
        killed = _track_terminations(monkeypatch)
        _configure_local_attach(monkeypatch, tmp_path)

        try:
            result = json.loads(bu_cli.browser_exec("print('x')"))
        finally:
            log.chmod(0o644)
        assert result["success"] is True
        assert killed == []

    def test_last_connecting_line_wins(self, tmp_path, monkeypatch):
        """A daemon that re-attached records several `connecting to` lines —
        the LAST one is its current endpoint."""
        home = _harness_home(monkeypatch, tmp_path)
        _plant_daemon_state(
            home,
            pid=4242,
            endpoint="ws://127.0.0.1:11111/devtools/browser/old",
            log_extra="connecting to ws://127.0.0.1:22222/devtools/browser/new",
        )
        killed = _track_terminations(monkeypatch)
        probed = []
        monkeypatch.setattr(
            bu_cli, "_cdp_endpoint_ready", lambda url: probed.append(url) or True
        )
        _configure_local_attach(monkeypatch, tmp_path)

        result = json.loads(bu_cli.browser_exec("print('x')"))
        assert result["success"] is True
        assert probed == ["ws://127.0.0.1:22222/devtools/browser/new"]
        assert killed == []


class TestHarnessNameResolution:
    """BU_NAME → harness daemon file naming."""

    def test_default_when_no_session(self):
        assert bu_cli._harness_name({}) == "default"

    def test_bu_name_wins(self):
        assert bu_cli._harness_name({"BU_NAME": "r7k2"}) == "r7k2"
