"""An OAuth server parked on an authentication failure logs "parking until credentials change".
The timed self-probe must honour that: with the same token file on disk it can only replay the
dead refresh token, every ``_PARKED_RETRY_INTERVAL``. A new token file (``hermes mcp login`` from
another process) revives it, and a token-endpoint outage (5xx) is not an authentication verdict:
that park keeps probing."""

import asyncio
from types import SimpleNamespace

import pytest

from tests.waiting import async_wait_until

_TOKENS = '{"access_token": "%s", "token_type": "Bearer", "refresh_token": "%s"}'


def _run_parked(monkeypatch, tmp_path, scenario, transport=None):
    """Run OAuth server ``srv`` whose every attempt is refused, and *scenario(state, tokens)*
    beside it; ``state`` counts transport attempts and self-probe wakes (a parked server's timer)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool, mcp_tool_config
    from tools.mcp_oauth import HermesTokenStorage, OAuthNonInteractiveError

    cfg = {"url": "https://mcp.example.com/mcp", "auth": "oauth"}
    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 0.02)
    monkeypatch.setattr(mcp_tool_config, "_load_mcp_config", lambda: {"srv": cfg})
    tokens = HermesTokenStorage("srv")._tokens_path()
    tokens.parent.mkdir(parents=True, exist_ok=True)
    tokens.write_text(_TOKENS % ("A1", "R1"))
    state = {"calls": 0, "wakes": 0}

    class _Task(mcp_tool.MCPServerTask):
        def _is_http(self):
            return True

        async def _prepare_run(self, config):
            self._config, self._auth_type = config, "oauth"
            return True

        async def _backoff_sleep(self, budget):
            return None

        async def _wait_for_reconnect_or_shutdown(self, timeout=None):
            outcome = await super()._wait_for_reconnect_or_shutdown(timeout)
            state["wakes"] += outcome == "self-probe"
            return outcome

        async def _run_http(self, config):
            state["calls"] += 1
            if transport:
                await transport(state)
            raise OAuthNonInteractiveError("MCP OAuth requires browser authorization")

    async def _main():
        task = state["task"] = _Task("srv")
        run_task = asyncio.ensure_future(task.run(dict(cfg)))
        try:
            await scenario(state, tokens)
        finally:
            task._shutdown_event.set()
            task._reconnect_event.set()
            await asyncio.wait_for(run_task, timeout=15)

    asyncio.run(_main())


@pytest.mark.no_isolate
@pytest.mark.parametrize("refresh_inconclusive", [False, True])
def test_auth_parked_probe_waits_for_new_credentials(monkeypatch, tmp_path, refresh_inconclusive):
    from tools.mcp_oauth_manager import get_manager

    # The provider's verdict on its last refresh, where the manager keeps the server's provider.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = SimpleNamespace(_hermes_refresh_inconclusive=refresh_inconclusive)
    monkeypatch.setitem(get_manager()._entries, get_manager()._key("srv"), SimpleNamespace(provider=provider))

    async def _scenario(state, tokens):
        await async_wait_until(lambda: state["wakes"] >= 5, what="server never parked and probed")
        if refresh_inconclusive:  # the token endpoint was down: only the probe can find out it is back
            assert state["calls"] > 1, "park after a token-endpoint outage stopped probing"
            return
        assert state["calls"] == 1, "auth-parked self-probe retried with unchanged credentials"
        tokens.write_text(_TOKENS % ("A2", "R2"))  # same size: only the mtime differs
        await async_wait_until(lambda: state["calls"] > 1, what="probe ignored the new token file")

    _run_parked(monkeypatch, tmp_path, _scenario)


@pytest.mark.no_isolate
def test_login_landing_during_the_failing_attempt_wakes_the_probe(monkeypatch, tmp_path):
    """The gate compares against the credentials the failed attempt started with. Taken after
    the failure, the snapshot would already hold the new token file and never see it change."""
    release = asyncio.Event()

    async def _held_first_attempt(state):
        if state["calls"] == 1:
            await release.wait()

    async def _scenario(state, tokens):
        await async_wait_until(lambda: state["calls"] == 1, what="first attempt never started")
        tokens.write_text(_TOKENS % ("A2", "R2"))
        release.set()
        await async_wait_until(lambda: state["calls"] >= 2, what="login during the attempt was never picked up")

    _run_parked(monkeypatch, tmp_path, _scenario, transport=_held_first_attempt)


@pytest.mark.no_isolate
def test_gate_does_not_outlive_the_auth_park(monkeypatch, tmp_path):
    """A requested reconnect that then fails on the network parks again. That park is not about
    credentials: its probe runs on schedule, or the server never comes back by itself."""
    from tools import mcp_tool

    async def _unreachable_after_the_first(state):
        if state["calls"] > 1:
            raise ConnectionError("connection refused")

    async def _scenario(state, tokens):
        await async_wait_until(lambda: state["wakes"] >= 2, what="server never parked on authentication")
        state["task"]._reconnect_event.set()
        ladder = 2 + mcp_tool._MAX_INITIAL_CONNECT_RETRIES + mcp_tool._MAX_RECONNECT_RETRIES
        await async_wait_until(lambda: state["calls"] > ladder, what="network park inherited the credential gate")

    _run_parked(monkeypatch, tmp_path, _scenario, transport=_unreachable_after_the_first)


@pytest.mark.parametrize("status, inconclusive", [(400, False), (429, True), (503, True)])
def test_only_a_4xx_refresh_refusal_is_an_auth_verdict(status, inconclusive):
    from tools.mcp_oauth_provider import HermesProviderMixin

    class _Provider(HermesProviderMixin):
        async def _hermes_handle_refresh_response(self, response):
            return False

        async def _hermes_release_refresh_fence(self):
            return None

    provider = object.__new__(_Provider)
    asyncio.run(provider._handle_refresh_response(SimpleNamespace(status_code=status)))
    assert provider._hermes_refresh_inconclusive is inconclusive
