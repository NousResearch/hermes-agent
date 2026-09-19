"""A PARKED MCP server the user switched off must not be re-authenticated by the timed self-probe.

The run task works off a STARTUP SNAPSHOT of its ``mcp_servers`` entry, so a server that was
enabled when it connected and was then set to ``enabled: false`` keeps waking every
``_PARKED_RETRY_INTERVAL`` (300s) to rebuild its transport — re-driving OAuth and emitting a
WARNING pair per tick for the life of the process: ~2,100 warning pairs from 8 disabled servers in
two days, enough to push ``errors.log`` to its line cap. The parked probe therefore has to consult
``enabled`` in config.yaml as it is NOW, not as it was at boot.

Retrying is NOT dropped for enabled servers: an enabled server whose credentials changed still
self-probes back to life (and so does one whose entry is a config we cannot read).
"""

import asyncio
import logging

import pytest

from tools.mcp_tool import MCPServerTask


def _group(*excs, msg="unhandled errors in a TaskGroup") -> BaseExceptionGroup:
    return BaseExceptionGroup(msg, list(excs))


def _auth_error(httpx):
    request = httpx.Request("POST", "https://mcp.example.test/mcp")
    return httpx.HTTPStatusError("401", request=request, response=httpx.Response(401, request=request))


class _ProbeTask(MCPServerTask):
    """stdio server whose transport fails to authenticate until ``state["authenticated"]`` flips."""

    def __init__(self, name, httpx, state):
        super().__init__(name)
        self._httpx, self._state = httpx, state

    def _is_http(self):
        return False

    def _deregister_tools(self):
        self._state["parked"] = True
        self._registered_tool_names = []

    async def _run_stdio(self, config):
        self._state["transport_calls"] += 1
        if not self._state["authenticated"]:
            raise _group(_auth_error(self._httpx))
        self.session = object()
        await self._wait_for_lifecycle_event()


def _run_parked(monkeypatch, config_on_disk, *, probe_ticks=10):
    """Drive one server to the parked state, let ``probe_ticks`` self-probe windows elapse, then
    re-authenticate it and give it a chance to revive. Returns the recorded ``state``."""
    httpx = pytest.importorskip("httpx")
    from tools import mcp_tool
    from tools import mcp_tool_config as _config

    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 0.05)
    monkeypatch.setattr(mcp_tool, "_MAX_INITIAL_CONNECT_RETRIES", 0)
    monkeypatch.setattr(_config, "_load_mcp_config", lambda: dict(config_on_disk))

    _real_sleep = asyncio.sleep
    state = {"transport_calls": 0, "parked": False, "authenticated": False,
             "calls_before_relogin": 0, "revived": None}

    async def _scenario():
        task = _ProbeTask("figma", httpx, state)
        task._config = {"command": "x"}  # the boot snapshot: no `enabled` key -> enabled
        run_task = asyncio.ensure_future(task.run({"command": "x"}))
        for _ in range(500):  # let it park on the auth failure
            await _real_sleep(0)
            if state["parked"]:
                break
        assert state["parked"], "auth failure never parked"
        for _ in range(probe_ticks * 4):  # several _PARKED_RETRY_INTERVAL windows elapse
            await _real_sleep(0.01)
        state["calls_before_relogin"] = state["transport_calls"]
        # The user re-authenticates (`hermes mcp login`). Nothing sets _reconnect_event: a revival
        # can only come from the timed self-probe.
        state["authenticated"] = True
        for _ in range(600):
            await _real_sleep(0.005)
            if task.session is not None:
                break
        state["revived"] = task.session is not None
        task._shutdown_event.set()
        task._reconnect_event.set()
        try:
            await asyncio.wait_for(run_task, timeout=15)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            run_task.cancel()

    asyncio.run(_scenario())
    return state


def _auth_warnings(caplog):
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and "failed initial authentication" in r.getMessage()]


@pytest.mark.no_isolate
def test_disabled_server_is_not_reprobed_and_stays_silent(monkeypatch, caplog):
    """``enabled: false`` on disk: the parked probe never rebuilds the transport again — not even
    once credentials are valid again — and the per-tick ``failed initial authentication`` WARNING
    stops after the attempt that parked it."""
    with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
        state = _run_parked(monkeypatch, {"figma": {"command": "x", "enabled": False}})

    warnings = _auth_warnings(caplog)
    assert len(warnings) == 1, warnings
    assert state["transport_calls"] == 1, (
        f"a server disabled in config was re-authenticated {state['transport_calls']} times "
        "by the parked self-probe")
    assert state["revived"] is False, "a disabled server came back up"


@pytest.mark.no_isolate
def test_disabled_flag_is_read_from_config_file(monkeypatch, tmp_path):
    """The flag comes from ``config.yaml`` itself (a fresh, mtime-cached read), so a disable made
    after boot is visible to the run task; an entry that is absent or not disabled is not."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "mcp_servers:\n  figma:\n    command: x\n    enabled: false\n"
        "  asana:\n    url: https://x/mcp\n    enabled: true\n")
    from tools.mcp_tool_server_run import _disabled_in_live_config

    assert _disabled_in_live_config("figma") is True
    assert _disabled_in_live_config("asana") is False
    assert _disabled_in_live_config("never-configured") is False


@pytest.mark.no_isolate
def test_enabled_server_still_self_probes_after_a_credentials_change(monkeypatch, caplog):
    """PROTECTION: the disable-check must not become a retry-killer. An ENABLED server whose
    credentials changed still revives on the timed self-probe alone."""
    with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
        state = _run_parked(monkeypatch, {"figma": {"command": "x", "enabled": True}})

    assert state["calls_before_relogin"] >= 2, "enabled server was never re-probed"
    assert state["revived"] is True, "enabled server never revived after re-authentication"


@pytest.mark.no_isolate
def test_server_absent_from_config_keeps_retrying(monkeypatch, caplog):
    """PROTECTION: an absent/unreadable config entry (``_load_mcp_config`` also returns ``{}`` on
    error and in safe mode) must not be read as "disabled" — the retry guarantee wins."""
    with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
        state = _run_parked(monkeypatch, {})

    assert state["calls_before_relogin"] >= 2, "an unlisted server lost its self-probe"
    assert state["revived"] is True, "unlisted server never revived after re-authentication"
