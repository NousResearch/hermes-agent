"""Hosted-dashboard bridge for MCP OAuth browser callbacks."""

import asyncio
import logging
import threading

import pytest


def _flow(flow_id: str = "flow-retry", server_name: str = "asana"):
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    return DashboardOAuthFlow(
        flow_id=flow_id,
        server_name=server_name,
        profile=None,
        hermes_home="/tmp/hermes-test",
        redirect_uri=f"https://agent.example/mcp/oauth/callback/{flow_id}",
    )


def test_dashboard_flow_exposes_authorization_url_and_accepts_callback():
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-1",
        server_name="reports",
        profile=None,
        hermes_home="/tmp/hermes-test",
        redirect_uri="https://agent.example/mcp/oauth/callback/flow-1",
    )

    asyncio.run(flow.publish_authorization_url("https://idp.example/authorize?state=s1"))
    assert flow.snapshot() == {
        "flow_id": "flow-1",
        "server_name": "reports",
        "status": "authorization_required",
        "authorization_url": "https://idp.example/authorize?state=s1",
        "error": None,
    }

    flow.deliver_callback(code="code-1", state="s1", error=None)
    assert asyncio.run(flow.wait_for_callback()) == ("code-1", "s1", None)


def test_dashboard_flow_preserves_rfc9207_iss():
    """RFC 9207 ``iss`` survives the callback bridge: mcp 2.x rejects an authorization response
    that omits it when the authorization server advertised support (Cloudflare, Resend)."""
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-iss",
        server_name="cloudflare",
        profile=None,
        hermes_home="/tmp/hermes-test",
        redirect_uri="https://agent.example/mcp/oauth/callback/flow-iss",
    )
    asyncio.run(flow.publish_authorization_url("https://idp.example/authorize?state=s1"))

    flow.deliver_callback(code="code-1", state="s1", error=None, iss="https://mcp.cloudflare.com")
    assert asyncio.run(flow.wait_for_callback()) == ("code-1", "s1", "https://mcp.cloudflare.com")


def test_dashboard_flow_accepts_only_one_concurrent_callback():
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-race",
        server_name="reports",
        profile=None,
        hermes_home="/tmp/hermes-test",
        redirect_uri="https://agent.example/mcp/oauth/callback/flow-race",
    )
    asyncio.run(flow.publish_authorization_url("https://idp.example/authorize?state=state"))

    start = threading.Barrier(3)
    outcomes: list[str] = []

    def deliver(code: str) -> None:
        start.wait()
        try:
            flow.deliver_callback(code=code, state="state", error=None)
            outcomes.append("accepted")
        except ValueError:
            outcomes.append("rejected")

    workers = [threading.Thread(target=deliver, args=(code,)) for code in ("one", "two")]
    for worker in workers:
        worker.start()
    start.wait()
    for worker in workers:
        worker.join()

    assert sorted(outcomes) == ["accepted", "rejected"]


@pytest.mark.usefixtures("require_mcp_2_sdk")
def test_mcp_oauth_helpers_use_dashboard_flow_without_loopback_port():
    # _build_client_metadata validates through the SDK's OAuthClientMetadata model.
    pytest.importorskip("mcp.shared.auth", reason="MCP SDK not installed")
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow, dashboard_oauth_flow
    from tools.mcp_oauth import (
        HermesTokenStorage,
        _build_client_metadata,
        _configure_callback_port,
        _make_callback_waiter,
        _make_redirect_handler,
    )

    flow = DashboardOAuthFlow(
        flow_id="flow-4",
        server_name="reports",
        profile=None,
        hermes_home="/tmp/hermes-test",
        redirect_uri="https://agent.example/mcp/oauth/callback/flow-4",
    )
    cfg = {}
    with dashboard_oauth_flow(flow):
        assert _configure_callback_port(cfg, HermesTokenStorage("reports")) == 0
        metadata = _build_client_metadata(cfg)
        assert str(metadata.redirect_uris[0]) == flow.redirect_uri

        asyncio.run(
            _make_redirect_handler(0)(
                "https://idp.example/authorize?state=state-4"
            )
        )
        flow.deliver_callback(code="code-4", state="state-4", error=None)
        # mcp 2.0's callback_handler contract returns an
        # AuthorizationCodeResult, not the legacy (code, state) tuple.
        result = asyncio.run(_make_callback_waiter(0)())
        assert (result.code, result.state) == ("code-4", "state-4")

    assert flow.authorization_url == "https://idp.example/authorize?state=state-4"


def test_first_mark_error_reason_reaches_callback_waiter_and_is_never_clobbered():
    """A failure marked before any browser redirect (worker crash, authorization-URL timeout,
    user cancel) must reach the SDK's callback waiter — not the generic no-code line — and the
    worker's follow-on ``mark_error`` (the waiter's own exception) must not overwrite the cause
    the dashboard/Desktop polls. A delivered callback likewise survives a late ``mark_error``."""
    flow = _flow("flow-err")
    flow.mark_error("OAuth cancelled by user")
    with pytest.raises(RuntimeError, match="OAuth cancelled by user"):
        asyncio.run(flow.wait_for_callback())
    flow.mark_error("OAuth authorization failed: OAuth cancelled by user")
    assert flow.snapshot()["error"] == "OAuth cancelled by user"

    delivered = _flow("flow-late")
    asyncio.run(delivered.publish_authorization_url("https://idp.example/authorize?state=s9"))
    delivered.deliver_callback(code="code-9", state="s9", error=None)
    delivered.mark_error("worker crashed after the browser redirected")
    assert asyncio.run(delivered.wait_for_callback())[:2] == ("code-9", "s9")


def test_empty_exception_text_stays_diagnosable():
    """``str()`` of a bare ``TimeoutError()``/``RuntimeError()`` is ""; the workers record the type
    name and the flow never exposes a blank cause to the waiter or the poller."""
    from tools.mcp_dashboard_oauth import exception_message

    assert exception_message(RuntimeError()) == "RuntimeError"
    assert exception_message(RuntimeError("boom")) == "boom"

    flow = _flow("flow-empty")
    flow.mark_error("")
    assert flow.snapshot()["error"]
    with pytest.raises(RuntimeError, match="empty error message"):
        asyncio.run(flow.wait_for_callback())


def test_failed_reauth_rollback_preserves_newer_oauth_state(tmp_path, monkeypatch):
    from tools.mcp_oauth import HermesTokenStorage

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    storage = HermesTokenStorage("reports")
    storage._tokens_path().parent.mkdir(parents=True)
    storage._tokens_path().write_text("OLD", encoding="utf-8")
    backup = storage.snapshot()
    storage.remove()

    storage._tokens_path().write_text("FRESH", encoding="utf-8")
    storage.restore(backup, only_if_absent=True)

    assert storage._tokens_path().read_text(encoding="utf-8") == "FRESH"


def test_preregistered_pinned_redirect_port_keeps_loopback_listener_under_dashboard_flow(tmp_path, monkeypatch):
    """A no-DCR entry (``client_id`` + ``redirect_port``, e.g. the shipped Asana manifest) registered
    ``http://localhost:<port>/callback`` with the vendor, which matches redirect URLs exactly. The
    dashboard/Desktop Authorize button must therefore keep that loopback URI and listener — the
    dashboard flow only publishes the authorization URL — instead of forcing its own callback URL."""
    import socket
    import urllib.request

    from tools.mcp_dashboard_oauth import DashboardOAuthFlow, dashboard_oauth_flow
    from tools.mcp_oauth import (
        HermesTokenStorage,
        _build_client_metadata,
        _configure_callback_port,
        _make_callback_waiter,
        _make_redirect_handler,
        force_interactive_oauth,
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    flow = DashboardOAuthFlow(flow_id="flow-5", server_name="asana", profile=None, hermes_home=str(tmp_path),
                              redirect_uri="https://agent.example/mcp/oauth/callback/flow-5")
    cfg = {"client_id": "pre-registered", "client_secret": "s", "redirect_host": "localhost", "redirect_port": port}
    with dashboard_oauth_flow(flow), force_interactive_oauth():
        assert _configure_callback_port(cfg, HermesTokenStorage("asana")) == port
        assert str(_build_client_metadata(cfg).redirect_uris[0]) == f"http://localhost:{port}/callback"
        asyncio.run(_make_redirect_handler(port)("https://idp.example/authorize?state=state-5"))
        assert flow.authorization_url == "https://idp.example/authorize?state=state-5"  # dashboard shows the URL

        async def _authorize():
            waiter = asyncio.ensure_future(_make_callback_waiter(port, timeout=10)())
            await asyncio.sleep(0.3)  # listener bound
            await asyncio.to_thread(
                lambda: urllib.request.urlopen(f"http://127.0.0.1:{port}/callback?code=code-5&state=state-5", timeout=5).read())
            return await waiter

        result = asyncio.run(_authorize())
    assert (result.code, result.state) == ("code-5", "state-5")


@pytest.mark.no_isolate
@pytest.mark.parametrize("ended_by", ["approved", "error"])
def test_server_task_does_not_park_on_an_ended_dashboard_flow(
    ended_by, monkeypatch, tmp_path, caplog
):
    """End-to-end shape of the report: an MCP server task whose inherited dashboard flow already
    ended used to raise out of the redirect handler on every attempt of the initial-connect ladder
    and park ("failed initial connection after 3 attempts").

    With the flow re-minted, the ladder fails (if at all) only for the ordinary reason — nobody
    completed the consent screen — and the handle is live and completable again.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_dashboard_oauth import dashboard_oauth_flow
    from tools.mcp_oauth import _make_redirect_handler
    from tools.mcp_tool import MCPServerTask

    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 0.05)
    _real_sleep = asyncio.sleep

    async def _fast_sleep(_delay, *a, **kw):
        await _real_sleep(0)

    monkeypatch.setattr(mcp_tool.asyncio, "sleep", _fast_sleep)

    flow = _flow("flow-stale", "asana")
    asyncio.run(flow.publish_authorization_url("https://idp.example/authorize?state=spent"))
    if ended_by == "approved":
        flow.mark_approved()
    else:
        flow.mark_error("worker gave up on the earlier attempt")
    flow.mark_worker_done()

    handler = _make_redirect_handler(0)

    class _Task(MCPServerTask):
        def _is_http(self) -> bool:
            return False

        def _deregister_tools(self) -> None:
            self._registered_tool_names = []

        async def _run_stdio(self, config):
            # The SDK's auth flow sits inside the transport and calls the redirect handler.
            with dashboard_oauth_flow(flow):
                await handler("https://idp.example/authorize?state=fresh")
            raise TimeoutError("OAuth callback timed out — the consent screen was never completed")

    async def _scenario():
        task = _Task("asana")
        failure = None
        try:
            await task.start({"command": "x"})
        except Exception as exc:  # noqa: BLE001 — the connect failure under test
            failure = exc
        parked = task._was_parked
        await task.shutdown()
        return failure, parked, task._task

    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        failure, parked, run_task = asyncio.run(_scenario())

    assert failure is not None, "the transport never failed, so the ladder was not exercised"
    assert not (isinstance(failure, RuntimeError) and "already ended" in str(failure)), (
        "an ended dashboard flow handle reached the connect ladder"
    )
    assert parked, "the ladder should end in a park, not a silent exit"
    assert run_task.done(), "run task must be reaped by shutdown()"
    assert any("parking" in record.getMessage() for record in caplog.records)

    # The retry re-minted the handle: the browser can still complete the new attempt.
    assert flow.snapshot()["status"] == "authorization_required"
    assert flow.expected_state == "fresh"
    flow.deliver_callback(code="late-code", state="fresh", error=None)
    assert asyncio.run(flow.wait_for_callback())[0] == "late-code"
