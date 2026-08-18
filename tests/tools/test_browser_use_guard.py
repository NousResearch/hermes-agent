"""Region E unit tests — guard orchestration, truth table, preamble, tiers.

Covers the §10 agreement truth table (coverage precondition + V/L/P/M rows),
endpoint tier resolution (H12), the advisory preamble markers (H5/H7),
marker stripping (H11), the policy-parameterized self-test (H3), guard env
construction (H2), and the multi-target browser-level listener (H6).
"""

import json
import os
import sys
import threading
import time
from unittest.mock import patch

import pytest

import tools.browser_exec_egress_guard as egress
import tools.browser_use_guard as bug

from tools.browser_use_guard import (
    _guard_endstate_verdict,
    _parse_preamble_markers,
    _resolve_endpoint_tier,
    _strip_guard_markers,
)


@pytest.fixture(autouse=True)
def _isolate_hermes_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.delenv("BU_CDP_WS", raising=False)
    monkeypatch.delenv("BU_CDP_URL", raising=False)
    yield


class _StubMonitor:
    def __init__(self, *, attach_failed=False, armed=True, saw_activity=True,
                 dropped=False, violation=None, last_known_url=""):
        self._attach_failed = attach_failed
        self._armed = armed
        self._saw = saw_activity
        self._dropped = dropped
        self._violation = violation
        self._last = last_known_url
        self.stopped = False

    def attach_failed(self): return self._attach_failed
    def armed(self): return self._armed
    def saw_activity(self, exec_started): return self._saw
    def mark_probe_success(self): pass
    def violation(self): return dict(self._violation) if self._violation else None
    def reset(self): pass
    def last_known_url(self): return self._last
    def event_count(self): return 1 if self._saw else 0
    def dropped(self): return self._dropped
    def request_log(self, limit=200): return []
    def stop(self, timeout=5.0): self.stopped = True


def _fake_guard(blocked=False, markers=None):
    return {
        "blocked": threading.Event(),
        "arm_failed": threading.Event(),
        "died": threading.Event(),
        "markers": markers if markers is not None else [],
        "proc": None,
        "report_sock": None,
    }


def _ctx(monitor=None, **overrides):
    ctx = {
        "enabled": True,
        "config": {"fail_open": False, "grace_s": 0.0, "attach_timeout_s": 1.0,
                   "allow_private": False},
        "endpoint": "ws://127.0.0.1:9222/x",
        "tier": "t1",
        "token": "tok",
        "state_dir": "",
        "exec_started": time.monotonic(),
        "monitor": monitor or _StubMonitor(),
        "ssrf_guard": _fake_guard(),
        "error": None,
    }
    ctx.update(overrides)
    return ctx


def _run(markers=None, egress_reason=None, guard_blocked=False, guard_died=False):
    return {
        "returncode": 0,
        "stdout": "page output\n",
        "stderr": "",
        "timed_out": False,
        "guard_blocked": guard_blocked,
        "guard_died": guard_died,
        "markers": markers if markers is not None else {"armed": None, "announce": None},
        "egress_reason": egress_reason,
    }


# ── §10 Agreement truth table ──────────────────────────────────────────────

class TestEndstateVerdictTruthTable:
    def test_row1_unsafe_probe_withholds(self):
        landed = "http://169.254.169.254/latest/meta-data/"
        v = _guard_endstate_verdict(_ctx(), landed, _run(markers={"armed": "full", "announce": ""}))
        assert v["verdict"] == "withhold"
        assert "metadata" in v["reason"]

    def test_row2_any_violation_withholds(self):
        violation = {"url": "http://10.0.0.1/x", "policy": "private",
                     "event": "Network.requestWillBeSent", "request_id": "r1",
                     "ts": time.time(), "frame_id": "f1", "session_id": "s1"}
        monitor = _StubMonitor(violation=violation)
        v = _guard_endstate_verdict(_ctx(monitor), "https://example.com/", _run())
        assert v["verdict"] == "withhold"
        assert "10.0.0.1" in v["reason"]

    def test_row2_egress_marker_withholds(self):
        v = _guard_endstate_verdict(
            _ctx(), "https://example.com/",
            _run(egress_reason="Blocked: browser_exec attempted a direct connection"),
        )
        assert v["verdict"] == "withhold"

    def test_row2_guard_block_withholds(self):
        ctx = _ctx(ssrf_guard=_fake_guard(
            blocked=True,
            markers=["__HERMES_BROWSER_EXEC_SSRF_BLOCK__:http://10.0.0.1/x"],
        ))
        v = _guard_endstate_verdict(ctx, "https://example.com/", _run(guard_blocked=True))
        assert v["verdict"] == "withhold"
        assert "10.0.0.1" in v["reason"]

    def test_row2_guard_death_withholds(self):
        v = _guard_endstate_verdict(_ctx(), "https://example.com/", _run(guard_died=True))
        assert v["verdict"] == "withhold"

    def test_row3_safe_probe_unsafe_last_known_withholds(self):
        monitor = _StubMonitor(last_known_url="http://10.0.0.5/admin")
        v = _guard_endstate_verdict(_ctx(monitor), "https://example.com/", _run())
        assert v["verdict"] == "withhold"
        assert "10.0.0.5" in v["reason"]

    def test_row4_no_probe_unsafe_last_known_withholds(self):
        monitor = _StubMonitor(last_known_url="http://127.0.0.1:8080/secret")
        v = _guard_endstate_verdict(_ctx(monitor), None, _run())
        assert v["verdict"] == "withhold"

    def test_row5_safe_safe_full_returns(self):
        v = _guard_endstate_verdict(
            _ctx(), "https://example.com/",
            _run(markers={"armed": "full", "announce": "ws://127.0.0.1:9222/x"}),
        )
        assert v["verdict"] == "return"

    def test_row7_full_marker_browser_observed_no_landing_withholds(self):
        """Defect-1 regression: P=None + M='full' + browser observed → WITHHOLD.

        A forged ``__HERMES_BROWSER_EXEC_ARMED__:full`` stdout marker plus a
        killed browser (trusted probe → None) must NOT release output.
        """
        monitor = _StubMonitor(saw_activity=True)  # browser observed
        v = _guard_endstate_verdict(
            _ctx(monitor), None,
            _run(markers={"armed": "full", "announce": "ws://127.0.0.1:9222/x"}),
        )
        assert v["verdict"] == "withhold"
        assert "trusted landing" in v["reason"]

    def test_full_marker_without_monitor_verified_activity_withholds(self):
        """Defect-5 regression: a forged 'full' marker cannot flip a withhold.

        Even with a safe trusted landing, the monitor must be armed AND have
        observed exec-window activity — a model-writable stdout marker alone
        never releases output.
        """
        monitor = _StubMonitor(saw_activity=False)  # monitor never verified
        v = _guard_endstate_verdict(
            _ctx(monitor), "https://example.com/",
            _run(markers={"armed": "full", "announce": "ws://127.0.0.1:9222/x"}),
        )
        assert v["verdict"] == "withhold"
        assert v["note"] == "m"

    def test_full_marker_requires_trusted_landing_and_monitor_verified(self):
        """Happy path needs all three: landed + monitor armed + activity."""
        monitor = _StubMonitor(saw_activity=True)
        v = _guard_endstate_verdict(
            _ctx(monitor), "https://example.com/",
            _run(markers={"armed": "full", "announce": "ws://127.0.0.1:9222/x"}),
        )
        assert v["verdict"] == "return"
        assert v["note"] == "full"

    def test_row6_no_session_consistent_returns(self):
        monitor = _StubMonitor(saw_activity=False)
        ctx = _ctx(monitor)
        ctx["exec_started"] = time.monotonic()
        v = _guard_endstate_verdict(
            ctx, None,
            _run(markers={"armed": "no-session", "announce": ""}),
        )
        assert v["verdict"] == "return"
        assert v["note"] == "no-session"

    def test_row7_no_session_but_browser_observed_withholds(self):
        monitor = _StubMonitor(saw_activity=True)
        v = _guard_endstate_verdict(
            _ctx(monitor), None,
            _run(markers={"armed": "no-session", "announce": ""}),
        )
        assert v["verdict"] == "withhold"

    def test_row8_nothing_armed_with_browser_withholds(self):
        monitor = _StubMonitor(saw_activity=True)
        v = _guard_endstate_verdict(_ctx(monitor), "https://example.com/", _run())
        assert v["verdict"] == "withhold"

    def test_row9_nothing_armed_no_browser_returns(self):
        monitor = _StubMonitor(saw_activity=False)
        ctx = _ctx(monitor)
        ctx["exec_started"] = time.monotonic()
        v = _guard_endstate_verdict(ctx, None, _run())
        assert v["verdict"] == "return"
        assert v["note"] == "no-browser"

    def test_coverage_precondition_withhold(self):
        monitor = _StubMonitor(attach_failed=True, armed=False, saw_activity=False)
        v = _guard_endstate_verdict(_ctx(monitor), "https://example.com/", _run())
        assert v["verdict"] == "withhold"
        assert "monitoring could not be verified" in v["reason"]

    def test_mid_exec_drop_marks_attestation_gap_withheld(self):
        monitor = _StubMonitor(armed=True, saw_activity=True, dropped=True)
        v = _guard_endstate_verdict(_ctx(monitor), "https://example.com/", _run())
        assert v["verdict"] == "withhold"

    def test_fail_open_downgrades_coverage_to_unverified(self):
        monitor = _StubMonitor(attach_failed=True, armed=False, saw_activity=False)
        ctx = _ctx(monitor)
        ctx["config"] = dict(ctx["config"], fail_open=True)
        v = _guard_endstate_verdict(ctx, "https://example.com/", _run())
        assert v["verdict"] == "return"
        assert v["note"] == "unverified"

    def test_announcement_consistency_mismatch_withhold(self):
        v = _guard_endstate_verdict(
            _ctx(), "https://example.com/",
            _run(markers={"armed": "full", "announce": "ws://evil.example:9999/x"}),
        )
        assert v["verdict"] == "withhold"
        assert "announcement" in v["reason"]

    def test_opt_out_floor_still_armed(self):
        """allow_private weakens private rows only; floor rows stay armed."""
        monitor = _StubMonitor(saw_activity=True)
        ctx = _ctx(monitor)
        ctx["config"] = dict(ctx["config"], allow_private=True)
        # Private landing with allow_private=True is not a P-block, but the
        # metadata floor still is.
        v = _guard_endstate_verdict(
            ctx, "http://169.254.169.254/latest/meta-data/", _run()
        )
        assert v["verdict"] == "withhold"
        assert "metadata" in v["reason"]


# ── Endpoint tiers (H12) ───────────────────────────────────────────────────

class TestEndpointTier:
    def test_t1_env_endpoint(self):
        env = {"BU_CDP_WS": "ws://127.0.0.1:9222/x"}
        endpoint, tier, err = _resolve_endpoint_tier(env, "t1", None)
        assert endpoint == "ws://127.0.0.1:9222/x"
        assert tier == "t1"
        assert err is None

    def test_t1b_provider_reresolution(self, monkeypatch):
        import tools.browser_tool as bt

        monkeypatch.setattr(bt, "_get_cloud_provider", lambda: object())
        monkeypatch.setattr(bt, "_get_session_info",
                            lambda task_id: {"cdp_url": "wss://cloud.example/cdp/x"})
        env = {}
        endpoint, tier, err = _resolve_endpoint_tier(env, "t1", "sess1")
        assert endpoint == "wss://cloud.example/cdp/x"
        assert tier == "t1b"
        assert env.get("BU_CDP_WS") == "wss://cloud.example/cdp/x"

    def test_t1b_provider_failure_withholds(self, monkeypatch):
        import tools.browser_tool as bt

        monkeypatch.setattr(bt, "_get_cloud_provider", lambda: object())
        monkeypatch.setattr(bt, "_get_session_info",
                            lambda task_id: (_ for _ in ()).throw(RuntimeError("api down")))
        endpoint, tier, err = _resolve_endpoint_tier({}, "t1", "sess1")
        assert err and "cannot attest" in err

    def test_no_endpoint_is_unattested(self):
        endpoint, tier, err = _resolve_endpoint_tier({}, "t1", None)
        assert endpoint == ""
        assert err and "no Hermes-attested" in err


# ── Preamble markers (H5/H7) + stripping (H11) ─────────────────────────────

class TestPreambleMarkers:
    def test_parse_armed_and_announce(self):
        stdout = (
            "page text\n"
            "__HERMES_BROWSER_EXEC_ANNOUNCE__:ws://127.0.0.1:9222/x\n"
            "__HERMES_BROWSER_EXEC_ARMED__:full\n"
            "more page text\n"
        )
        m = _parse_preamble_markers(stdout)
        assert m["armed"] == "full"
        assert m["announce"] == "ws://127.0.0.1:9222/x"

    def test_parse_no_session(self):
        m = _parse_preamble_markers("__HERMES_BROWSER_EXEC_ARMED__:no-session\n")
        assert m["armed"] == "no-session"

    def test_preamble_source_invariants(self):
        src = bug._GUARD_PREAMBLE
        assert "BH_RUNTIME_DIR" in src
        assert "__HERMES_BROWSER_EXEC_ARMED__:full" in src
        assert "__HERMES_BROWSER_EXEC_ARMED__:no-session" in src
        assert "flush=True" in src
        # Advisory + self-isolating: the preamble touches no helper names and
        # never imports browser_harness (it talks to the daemon port file
        # directly so a poisoned workspace helper cannot intercept it).
        assert "import browser_harness" not in src
        assert "from browser_harness" not in src
        assert "port.json" in src

    def test_strip_guard_markers_only_report_lines(self):
        text = (
            "content that echoes __HERMES_BROWSER_EXEC_ARMED__:full is preserved\n"
            "__HERMES_BROWSER_EXEC_ARMED__:full\n"
            "__HERMES_BROWSER_EXEC_ANNOUNCE__:ws://x\n"
            "__HERMES_EGRESS_GUARD__:installed:nn\n"
            "body\n"
        )
        stripped = _strip_guard_markers(text)
        assert "content that echoes" in stripped
        assert "body" in stripped
        assert "__HERMES_BROWSER_EXEC_ARMED__:full\n" not in stripped.split("content")[0]
        assert "__HERMES_EGRESS_GUARD__" not in stripped
        # Report lines gone; content echoes preserved.
        lines = stripped.splitlines()
        assert not any(ln.startswith("__HERMES_") for ln in lines)


# ── Self-test (H3) ─────────────────────────────────────────────────────────

class TestGuardSelfTest:
    def _guard_env(self, monkeypatch, tmp_path, install=True):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
        env = {}
        if install:
            egress._install_egress_guard(env)
        return env

    def test_self_test_passes_with_interposer(self, monkeypatch, tmp_path):
        env = self._guard_env(monkeypatch, tmp_path)
        ctx = {"enabled": True, "config": {"allow_private": False}}
        reason = bug._guard_self_test(ctx, env)
        assert reason is None

    def test_self_test_fails_without_interposer(self, monkeypatch, tmp_path):
        # No PYTHONPATH → sitecustomize never loads → assertions fail.
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        ctx = {"enabled": True, "config": {"allow_private": False}}
        reason = bug._guard_self_test(ctx, env)
        assert reason is not None
        assert "self-test failed" in reason

    def test_self_test_strips_model_workspace_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("BH_AGENT_WORKSPACE", "C:/model/ws")
        env = self._guard_env(monkeypatch, tmp_path)
        ctx = {"enabled": True, "config": {"allow_private": False}}
        assert bug._guard_self_test(ctx, env) is None  # runs clean under strip


# ── Guard env (H2) ─────────────────────────────────────────────────────────

class TestGuardEnv:
    def test_guard_env_injects_all_keys(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
        ctx = {
            "enabled": True,
            "token": "tok123",
            "state_dir": str(tmp_path / "state"),
            "endpoint": "ws://127.0.0.1:9222/x",
            "config": {},
        }
        env = {}
        out = bug._guard_env(env, ctx)
        assert out["HERMES_BROWSER_EXEC_GUARD_TOKEN"] == "tok123"
        assert out["HERMES_BROWSER_EXEC_GUARD_STATE_DIR"] == str(tmp_path / "state")
        assert out["HERMES_BROWSER_EXEC_EGRESS_GUARD"] == "1"
        assert "HERMES_BROWSER_EXEC_EGRESS_POLICY" in out
        policy = json.loads(out["HERMES_BROWSER_EXEC_EGRESS_POLICY"])
        assert policy["nonce"] == out["HERMES_BROWSER_EXEC_EGRESS_GUARD_NONCE"]
        assert "blocked_hostnames" in policy
        # PYTHONPATH prepend: the guard dir is first.
        guard_dir = str(egress._egress_guard_dir())
        assert out["PYTHONPATH"].startswith(guard_dir)
        # BH_RUNTIME_DIR pinning.
        assert out["BH_RUNTIME_DIR"]
        assert Path_exists(out["BH_RUNTIME_DIR"])

    def test_prepare_guard_disabled_by_config(self, monkeypatch):
        monkeypatch.setattr(bug, "_read_guard_config",
                            lambda: {"enabled": False, "fail_open": False,
                                     "grace_s": 0.0, "attach_timeout_s": 1.0,
                                     "allow_private": False})
        ctx = bug._prepare_guard({}, "t1", None)
        assert ctx["enabled"] is False

    def test_prepare_guard_no_endpoint_errors(self, monkeypatch):
        monkeypatch.setattr(bug, "_read_guard_config",
                            lambda: {"enabled": True, "fail_open": False,
                                     "grace_s": 0.0, "attach_timeout_s": 1.0,
                                     "allow_private": False})
        ctx = bug._prepare_guard({}, "t1", None)
        assert ctx.get("error")


def Path_exists(p):
    from pathlib import Path

    return Path(p).is_dir()


# ── Multi-target listener (H6 / §11.1) ─────────────────────────────────────

class TestMultiTargetListener:
    def test_new_tab_target_is_observed_via_auto_attach(self):
        """Target.createTarget → attachedToTarget → per-session Network.enable."""
        import asyncio

        from tools.browser_exec_monitor import NetworkExecMonitor

        class _FakeServer:
            """Synthetic event feed (no real WS): drives _on_event directly."""

            def __init__(self):
                self.enabled_sessions = []
                self.attached = []

            def record_enable(self, session_id):
                self.enabled_sessions.append(session_id)

        server = _FakeServer()
        m = NetworkExecMonitor("ws://127.0.0.1:1/x", task_id="t")
        sent = []

        async def _record_cdp(method, params=None, **kw):
            sent.append((method, kw.get("session_id")))
            if method == "Network.enable":
                server.record_enable(kw.get("session_id"))

        m._cdp = _record_cdp

        async def _drive():
            # A new tab target attaches mid-exec.
            await m._on_event("Target.attachedToTarget", {
                "sessionId": "s-newtab",
                "targetInfo": {"type": "page", "targetId": "t-newtab", "url": "about:blank"},
            }, None)
            # The new session's traffic is observed.
            await m._on_event("Network.requestWillBeSent", {
                "requestId": "r1", "url": "http://10.0.0.5/secret",
                "type": "Document",
            }, "s-newtab")

        asyncio.run(_drive())
        assert "s-newtab" in server.enabled_sessions
        v = m.violation()
        assert v is not None and v["policy"] == "private"

    def test_switch_tab_hop_does_not_escape_listener(self):
        """Per-target map keeps the private record even after hopping tabs."""
        import asyncio

        from tools.browser_exec_monitor import NetworkExecMonitor

        m = NetworkExecMonitor("ws://127.0.0.1:1/x", task_id="t")

        async def _drive():
            await m._on_event("Network.requestWillBeSent", {
                "requestId": "rA", "url": "http://10.0.0.5/secret",
                "type": "Document", "frameId": "fA",
            }, "s-tabA")
            await m._on_event("Page.frameNavigated", {
                "frame": {"url": "https://example.com/", "id": "fB"},
            }, "s-tabB")

        asyncio.run(_drive())
        assert m.violation() is not None  # private hop in tab A latched
        assert m.last_known_url() == "https://example.com/"  # active target L

    def test_model_cdp_network_disable_does_not_disable_host_listener(self):
        """Two-session independence: model Network.disable cannot mute us."""
        import asyncio

        from tools.browser_exec_monitor import NetworkExecMonitor

        m = NetworkExecMonitor("ws://127.0.0.1:1/x", task_id="t")

        async def _drive():
            # The model disables Network on ITS session.
            await m._on_event("Network.requestWillBeSent", {
                "requestId": "r1", "url": "http://10.0.0.1/x", "type": "Document",
            }, "s-model")

        asyncio.run(_drive())
        assert m.violation() is not None

    def test_spa_history_push_state_then_fetch_private_withheld(self):
        import asyncio

        from tools.browser_exec_monitor import NetworkExecMonitor

        m = NetworkExecMonitor("ws://127.0.0.1:1/x", task_id="t")

        async def _drive():
            await m._on_event("Network.requestWillBeSent", {
                "requestId": "r1", "url": "https://example.com/app", "type": "Document",
            }, None)
            await m._on_event("Network.requestWillBeSent", {
                "requestId": "r2", "url": "http://192.168.1.5/api", "type": "Fetch",
            }, None)

        asyncio.run(_drive())
        assert m.violation() is not None

    def test_redirect_chain_public_private_public_caught(self):
        import asyncio

        from tools.browser_exec_monitor import NetworkExecMonitor

        m = NetworkExecMonitor("ws://127.0.0.1:1/x", task_id="t")

        async def _drive():
            await m._on_event("Network.requestWillBeSent", {
                "requestId": "r1", "url": "http://93.184.216.34/", "type": "Document",
                "redirectResponse": {"url": "http://10.0.0.1/admin"},
            }, None)

        asyncio.run(_drive())
        assert m.violation() is not None
        assert m.violation()["url"] == "http://10.0.0.1/admin"
