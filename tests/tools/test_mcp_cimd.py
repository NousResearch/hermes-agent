"""Tests for CIMD (Client ID Metadata Document) support in MCP OAuth.

Under CIMD the ``client_id`` is the HTTPS URL of a document Hermes publishes,
which the authorization server fetches to learn our redirect URIs. The spec
requires an exact string match between the redirect URI in the authorization
request and one listed in that document
(draft-ietf-oauth-client-id-metadata-document section 4.2), so most of what
follows guards the two halves staying consistent: the published document, and
the conditions under which Hermes is allowed to present it.

Port mechanics run against a private range rather than the real one. The
production range is bound for real by ``_pick_cimd_port``, and test files run
as concurrent subprocesses, so sharing it across files would make whichever
test lost the race fail on a port another file legitimately held.
"""

import asyncio
import json
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "mcp.client.auth.oauth2",
    reason="MCP SDK 1.26.0+ required for OAuth support",
)

from tools.mcp_oauth import (  # noqa: E402 — after the SDK availability gate
    HermesTokenStorage,
    _CIMD_CLIENT_METADATA_URL,
    _CIMD_PORTS,
    _CIMD_REDIRECT_HOSTS,
    _build_client_metadata,
    _configure_callback_port,
    _is_valid_cimd_url,
    _maybe_use_cimd,
)

_DOCUMENT_PATH = (
    Path(__file__).resolve().parents[2]
    / "website" / "static" / "oauth" / "client-metadata.json"
)


def _document() -> dict:
    return json.loads(_DOCUMENT_PATH.read_text())


def _set_interactive_stdin(monkeypatch, *, is_tty: bool = True) -> None:
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = is_tty
    monkeypatch.setattr("tools.mcp_oauth.sys.stdin", mock_stdin)


@pytest.fixture(autouse=True)
def clean_port_state():
    """Give each test the port state of a freshly started process.

    Pinned-port assignments accumulate for the life of the process, and both
    ``_pick_cimd_port`` and ``_reserve_callback_port`` hold their socket until
    ``_wait_for_callback`` adopts it — which these tests stop short of.
    """
    import tools.mcp_oauth as mod

    mod._assigned_cimd_ports.clear()
    yield
    mod._assigned_cimd_ports.clear()
    for port in list(mod._reserved_sockets):
        sock = mod._reserved_sockets.pop(port, None)
        if sock is not None:
            sock.close()


@pytest.fixture
def private_ports(monkeypatch):
    """Swap the pinned range for ports no other test file competes for."""
    import tools.mcp_oauth as mod

    ports = (28890, 28891, 28892)
    monkeypatch.setattr(mod, "_CIMD_PORTS", ports)
    return ports


# ---------------------------------------------------------------------------
# The published document and the code must agree
# ---------------------------------------------------------------------------


def test_document_client_id_is_the_url_hermes_sends():
    """A CIMD document is only valid when its client_id is its own URL."""
    assert _document()["client_id"] == _CIMD_CLIENT_METADATA_URL


def test_document_declares_every_callback_hermes_can_build(tmp_path, monkeypatch):
    """Every loopback URI a CIMD flow could produce must be registered.

    Exact string matching means one missing entry is a hard auth failure on
    whichever port the OS happens to hand out that day. Built through the real
    ``_build_client_metadata`` so pydantic's URL serialization — not an
    f-string that merely resembles it — is what gets compared.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    declared = set(_document()["redirect_uris"])

    for host in _CIMD_REDIRECT_HOSTS:
        for port in _CIMD_PORTS:
            cfg = {"redirect_host": host, "_resolved_port": port}
            uri = str(_build_client_metadata(cfg).redirect_uris[0])
            assert uri in declared, f"{uri} is not registered in the document"


def test_document_url_passes_the_sdk_validator():
    """The SDK's constructor rejects a URL that fails this check."""
    from mcp.client.auth.utils import is_valid_client_metadata_url

    assert is_valid_client_metadata_url(_CIMD_CLIENT_METADATA_URL)


def test_document_advertises_a_public_native_client():
    """Loopback redirects need application_type=native (SEP-837), and CIMD
    carries no secret, so the client must be public."""
    doc = _document()
    assert doc["application_type"] == "native"
    assert doc["token_endpoint_auth_method"] == "none"
    assert "authorization_code" in doc["grant_types"]
    assert "refresh_token" in doc["grant_types"]


def test_document_carries_no_shared_secret():
    """Draft section 4.1 forbids secret material in the document."""
    doc = _document()
    assert "client_secret" not in doc
    assert "client_secret_expires_at" not in doc


def test_default_document_url_is_a_valid_client_identifier():
    """Section 3 constrains the URL beyond the SDK's https + path check."""
    assert _is_valid_cimd_url(_CIMD_CLIENT_METADATA_URL)


@pytest.mark.parametrize("url", [
    pytest.param("http://example.com/cimd.json", id="not-https"),
    pytest.param("https://example.com/", id="root-path"),
    pytest.param("https://example.com/cimd.json#frag", id="fragment"),
    pytest.param("https://user:pw@example.com/cimd.json", id="userinfo"),
    pytest.param("https://example.com/./cimd.json", id="dot-segment"),
    pytest.param("https://example.com/../cimd.json", id="double-dot-segment"),
])
def test_client_identifier_url_requirements_are_enforced(url):
    """Rejecting locally beats an opaque invalid-client page mid-flow."""
    assert not _is_valid_cimd_url(url)


def test_generated_redirect_uri_is_registered_in_the_document(tmp_path, monkeypatch):
    """End to end on the real range: the URI the SDK will actually send is
    one the authorization server accepts.

    The only test that runs the whole chain on production constants, so it
    binds a real pinned port. Another test file mid-flight can legitimately
    be holding all of them; the invariant itself is covered port-by-port,
    without binding, by the document tests above.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg: dict = {}

    _configure_callback_port(cfg, HermesTokenStorage("srv"))
    if "_cimd_url" not in cfg:
        pytest.skip("every pinned CIMD port is held by another process")
    metadata = _build_client_metadata(cfg)

    assert cfg["_cimd_url"] == _CIMD_CLIENT_METADATA_URL
    assert cfg["_resolved_port"] in _CIMD_PORTS
    assert str(metadata.redirect_uris[0]) in set(_document()["redirect_uris"])


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def test_eligible_flow_gets_a_pinned_port(tmp_path, monkeypatch, private_ports):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    result = _maybe_use_cimd({}, HermesTokenStorage("srv"))

    assert result is not None
    url, port = result
    assert url == _CIMD_CLIENT_METADATA_URL
    assert port in private_ports


def test_pinned_port_is_held_until_the_callback_adopts_it(
    tmp_path, monkeypatch, private_ports
):
    """A fixed port is as stealable as an ephemeral one in the minutes
    between selection and the browser redirect (#22161), so the socket stays
    bound rather than being probed and released."""
    import tools.mcp_oauth as mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    result = _maybe_use_cimd({}, HermesTokenStorage("srv"))
    assert result is not None
    port = result[1]

    assert port in mod._reserved_sockets
    thief = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(OSError):
        thief.bind(("127.0.0.1", port))
    thief.close()


def test_pinned_socket_survives_the_ephemeral_eviction_cap(
    tmp_path, monkeypatch, private_ports
):
    """The reservation FIFO cap must never close a parked pinned socket.

    Ephemeral reservations churn through ``_reserve_callback_port`` on every
    reconnect loop, and past the cap the oldest gets evicted. A pinned CIMD
    socket parked in the same dict would be the oldest under heavy
    concurrency — closing it converts the pinned flow back into a stealable
    window, the exact race the pin exists to prevent (#22161).
    """
    import tools.mcp_oauth as mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    result = _maybe_use_cimd({}, HermesTokenStorage("srv"))
    assert result is not None
    pinned = result[1]

    # Churn well past the cap; the pinned socket must stay parked and bound.
    ephemeral = [mod._reserve_callback_port() for _ in range(mod._MAX_RESERVED_SOCKETS + 4)]
    try:
        assert pinned in mod._reserved_sockets
        thief = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(OSError):
            thief.bind(("127.0.0.1", pinned))
        thief.close()
    finally:
        for port in ephemeral:
            sock = mod._reserved_sockets.pop(port, None)
            if sock is not None:
                sock.close()


def test_concurrent_servers_get_different_pinned_ports(
    tmp_path, monkeypatch, private_ports
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    ports = {
        _maybe_use_cimd({}, HermesTokenStorage(f"srv-{i}"))[1]
        for i in range(len(private_ports))
    }

    assert ports == set(private_ports)


def test_occupied_port_moves_to_the_next_in_the_range(
    tmp_path, monkeypatch, private_ports
):
    """Another profile mid-login holds a port; we take a different one."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    squatter = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        squatter.bind(("127.0.0.1", private_ports[0]))
    except OSError:
        squatter.close()
        pytest.skip(f"could not occupy port {private_ports[0]}")

    try:
        result = _maybe_use_cimd({}, HermesTokenStorage("srv"))
    finally:
        squatter.close()

    assert result is not None
    assert result[1] in private_ports[1:]


def test_self_hosted_document_url_overrides_the_default(
    tmp_path, monkeypatch, private_ports
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = {"client_metadata_url": "https://example.com/my-cimd.json"}

    result = _maybe_use_cimd(cfg, HermesTokenStorage("srv"))

    assert result is not None
    assert result[0] == "https://example.com/my-cimd.json"


@pytest.mark.parametrize("cfg", [
    pytest.param({"cimd": False}, id="explicitly-disabled"),
    pytest.param({"client_id": "preregistered"}, id="preregistered-client-id"),
    pytest.param({"client_secret": "shh"}, id="confidential-client"),
    pytest.param({"client_name": "Claude Code"}, id="pinned-client-name"),
    pytest.param(
        {"token_endpoint_auth_method": "client_secret_post"}, id="secret-auth-method"
    ),
    pytest.param({"redirect_port": 49399}, id="pinned-redirect-port"),
    pytest.param(
        {"redirect_uri": "https://proxy.example/callback"}, id="proxied-redirect-uri"
    ),
    pytest.param({"redirect_host": "example.test"}, id="non-loopback-redirect-host"),
    pytest.param(
        {"client_metadata_url": "http://insecure.example/cimd.json"}, id="http-document"
    ),
    pytest.param(
        {"client_metadata_url": "https://example.com/"}, id="root-path-document"
    ),
])
def test_config_that_conflicts_with_the_document_falls_back_to_dcr(
    cfg, tmp_path, monkeypatch, private_ports
):
    """Each of these asks for an identity or a callback the document can't
    present."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    assert _maybe_use_cimd(dict(cfg), HermesTokenStorage("srv")) is None


def test_dashboard_flow_falls_back_to_dcr(tmp_path, monkeypatch, private_ports):
    """The dashboard redirects to its own public URL, which no static
    document can declare — it is per-deployment."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow, dashboard_oauth_flow

    flow = DashboardOAuthFlow(
        flow_id="flow-1",
        server_name="srv",
        profile=None,
        hermes_home=str(tmp_path),
        redirect_uri="https://agent.example/api/mcp/oauth/callback/srv",
    )

    cfg: dict = {}
    with dashboard_oauth_flow(flow):
        _configure_callback_port(cfg, HermesTokenStorage("srv"))

    assert "_cimd_url" not in cfg
    assert cfg["redirect_uri"] == flow.redirect_uri


def test_existing_registration_falls_back_to_dcr(tmp_path, monkeypatch, private_ports):
    """A stored client_id is bound to the redirect URI it registered with;
    switching to CIMD now would invalidate it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    storage = HermesTokenStorage("srv")
    storage._client_info_path().parent.mkdir(parents=True, exist_ok=True)
    storage._client_info_path().write_text('{"client_id": "dcr-issued"}')

    assert _maybe_use_cimd({}, storage) is None


def test_exhausted_port_range_falls_back_to_dcr(tmp_path, monkeypatch, private_ports):
    """With every pinned port held elsewhere, the flow reverts to an
    ephemeral port and no CIMD client_id."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    squatters = []
    try:
        for port in private_ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", port))
            squatters.append(sock)
    except OSError:
        for sock in squatters:
            sock.close()
        pytest.skip("could not occupy the full pinned CIMD port range")

    try:
        cfg: dict = {}
        port = _configure_callback_port(cfg, HermesTokenStorage("srv"))
    finally:
        for sock in squatters:
            sock.close()

    assert "_cimd_url" not in cfg
    assert port not in private_ports
    assert cfg["_resolved_port"] == port


def test_more_servers_than_pinned_ports_all_get_cimd(
    tmp_path, monkeypatch, private_ports
):
    """Providers are built per configured server, well before any browser
    flow runs, so the size of the port range must not cap how many servers
    can use CIMD."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    results = [
        _maybe_use_cimd({}, HermesTokenStorage(f"srv-{i}"))
        for i in range(len(private_ports) + 3)
    ]

    assert all(r is not None for r in results)
    assert {r[0] for r in results} == {_CIMD_CLIENT_METADATA_URL}


# ---------------------------------------------------------------------------
# Only pin a port for servers that might actually want a document
# ---------------------------------------------------------------------------


def _cache_server_metadata(storage, *, supports_cimd):
    from mcp.shared.auth import OAuthMetadata

    storage.save_oauth_metadata(OAuthMetadata.model_validate({
        "issuer": "https://idp.example.com",
        "authorization_endpoint": "https://idp.example.com/authorize",
        "token_endpoint": "https://idp.example.com/token",
        "response_types_supported": ["code"],
        "client_id_metadata_document_supported": supports_cimd,
    }))


@pytest.mark.parametrize("supports_cimd, expect_pinned", [
    pytest.param(True, True, id="server-advertises-cimd"),
    pytest.param(False, False, id="server-does-not"),
])
def test_cached_metadata_decides_whether_to_pin(
    supports_cimd, expect_pinned, tmp_path, monkeypatch, private_ports
):
    """The SDK only learns whether a server does CIMD during its 401 branch,
    long after Hermes fixes the redirect URI. Metadata cached by an earlier
    connection closes that gap, so a known DCR-only server keeps the reserved
    ephemeral port it has always used instead of a guessable fixed one."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    storage = HermesTokenStorage("srv")
    _cache_server_metadata(storage, supports_cimd=supports_cimd)

    cfg: dict = {}
    port = _configure_callback_port(cfg, storage)

    assert (cfg.get("_cimd_url") is not None) is expect_pinned
    assert (port in private_ports) is expect_pinned


def test_unknown_server_still_gets_a_document(tmp_path, monkeypatch, private_ports):
    """No cached metadata means a first-ever connect, where guessing CIMD is
    the only way to ever use it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    cfg: dict = {}
    _configure_callback_port(cfg, HermesTokenStorage("srv"))

    assert cfg["_cimd_url"] == _CIMD_CLIENT_METADATA_URL


def test_cached_pinned_port_is_not_handed_to_a_sibling_server(
    tmp_path, monkeypatch, private_ports
):
    """An earlier CIMD login leaves its pinned port in the registration on
    disk. Restoring it must also claim it, or the next server picks the same
    one and the two flows fight over one listener."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    settled = HermesTokenStorage("settled")
    settled._client_info_path().parent.mkdir(parents=True, exist_ok=True)
    settled._client_info_path().write_text(json.dumps({
        "client_id": _CIMD_CLIENT_METADATA_URL,
        "redirect_uris": [f"http://127.0.0.1:{private_ports[0]}/callback"],
    }))

    restored = _configure_callback_port({}, settled)
    fresh = _maybe_use_cimd({}, HermesTokenStorage("fresh"))

    assert restored == private_ports[0]
    assert fresh is not None
    assert fresh[1] != private_ports[0]


# ---------------------------------------------------------------------------
# Provider wiring
# ---------------------------------------------------------------------------


def test_build_oauth_auth_forwards_the_document_url(
    tmp_path, monkeypatch, private_ports
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth import build_oauth_auth

    provider = build_oauth_auth("srv", "https://mcp.example.com/mcp", {})

    assert provider.context.client_metadata_url == _CIMD_CLIENT_METADATA_URL


def test_build_oauth_auth_omits_the_url_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth import build_oauth_auth

    provider = build_oauth_auth("srv", "https://mcp.example.com/mcp", {"cimd": False})

    assert provider.context.client_metadata_url is None


def test_dcr_flow_passes_no_cimd_keyword_at_all():
    """An SDK predating CIMD rejects the keyword outright, so a DCR flow must
    not carry it — otherwise one unsupported argument breaks every login."""
    from tools.mcp_oauth import cimd_provider_kwargs

    assert cimd_provider_kwargs({}) == {}
    assert cimd_provider_kwargs({"_cimd_url": "https://x.example/c.json"}) == {
        "client_metadata_url": "https://x.example/c.json"
    }


@pytest.mark.parametrize("advertised, expect_cimd", [
    pytest.param(True, True, id="server-supports-cimd"),
    pytest.param(False, False, id="server-does-not"),
    pytest.param(None, False, id="server-silent"),
])
def test_sdk_chooses_cimd_only_when_the_server_advertises_it(
    advertised, expect_cimd, tmp_path, monkeypatch, private_ports
):
    """Closes the loop on the handoff: feed what Hermes configured into the
    SDK's own branch condition rather than asserting on our side of it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from mcp.client.auth.utils import should_use_client_metadata_url
    from tools.mcp_oauth import build_oauth_auth

    provider = build_oauth_auth("srv", "https://mcp.example.com/mcp", {})
    server_metadata = SimpleNamespace(client_id_metadata_document_supported=advertised)

    chosen = should_use_client_metadata_url(
        server_metadata, provider.context.client_metadata_url
    )

    assert chosen is expect_cimd


def test_manager_forwards_the_document_url(tmp_path, monkeypatch, private_ports):
    """The manager is the path live MCP connections actually take."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import MCPOAuthManager, reset_manager_for_tests

    reset_manager_for_tests()
    provider = MCPOAuthManager().get_or_build_provider(
        "srv", "https://mcp.example.com/mcp", {}
    )

    assert provider.context.client_metadata_url == _CIMD_CLIENT_METADATA_URL


# ---------------------------------------------------------------------------
# Rejection fallback
# ---------------------------------------------------------------------------


def _fake_response(status, url, body):
    """A minimal stand-in for the httpx.Response the SDK feeds our bridge."""
    resp = MagicMock()
    resp.status_code = status
    resp.request = SimpleNamespace(url=url)

    async def _aread():
        return body

    resp.aread = _aread
    return resp


def _provider_rejected_at_token_endpoint(tmp_path, monkeypatch, client_id):
    from tools.mcp_oauth_manager import MCPOAuthManager, reset_manager_for_tests

    reset_manager_for_tests()
    _set_interactive_stdin(monkeypatch)
    token_endpoint = "https://idp.example.com/oauth/token"

    provider = MCPOAuthManager().get_or_build_provider(
        "srv", "https://mcp.example.com", {}
    )
    provider.context.oauth_metadata = SimpleNamespace(token_endpoint=token_endpoint)
    provider.context.client_info = SimpleNamespace(client_id=client_id)
    provider._initialized = True

    asyncio.run(provider._maybe_flag_poisoned_client(
        _fake_response(400, token_endpoint, b'{"error":"invalid_client"}')
    ))
    return provider


def test_rejected_document_stops_being_presented(tmp_path, monkeypatch, private_ports):
    """A server that fetched our document and refused it would loop if we
    kept sending the same client_id, so the retry drops to DCR."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    provider = _provider_rejected_at_token_endpoint(
        tmp_path, monkeypatch, _CIMD_CLIENT_METADATA_URL
    )

    assert provider.context.client_metadata_url is None
    assert provider.context.client_info is None
    assert provider._initialized is False


def test_rejected_document_stays_rejected_after_a_restart(
    tmp_path, monkeypatch, private_ports
):
    """The in-memory drop dies with the process; a fresh one would walk back
    into the same refusal without a marker on disk."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    _provider_rejected_at_token_endpoint(
        tmp_path, monkeypatch, _CIMD_CLIENT_METADATA_URL
    )
    storage = HermesTokenStorage("srv")

    assert storage.cimd_rejected()
    assert _maybe_use_cimd({}, storage) is None


def test_reauthorizing_clears_the_rejection(tmp_path, monkeypatch, private_ports):
    """`hermes mcp login` wipes stored state, so a fixed document is retried."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    storage = HermesTokenStorage("srv")
    storage.mark_cimd_rejected()

    storage.remove()

    assert not storage.cimd_rejected()
    assert _maybe_use_cimd({}, storage) is not None


def test_rejected_dcr_client_leaves_cimd_available(
    tmp_path, monkeypatch, private_ports
):
    """A dead DCR registration says nothing about our document."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    provider = _provider_rejected_at_token_endpoint(
        tmp_path, monkeypatch, "dcr-issued-id"
    )

    assert provider.context.client_metadata_url == _CIMD_CLIENT_METADATA_URL
    assert not HermesTokenStorage("srv").cimd_rejected()


# ---------------------------------------------------------------------------
# Diagnosing a refusal the protocol gives us no signal for
# ---------------------------------------------------------------------------


def _timed_out_waiter_message(monkeypatch, cimd_url):
    """Run a callback waiter to its timeout and return the error text."""
    import tools.mcp_oauth as mod

    # No paste thread and no fail-fast: this exercises the timeout itself.
    monkeypatch.setattr(mod, "_is_interactive", lambda: False)
    monkeypatch.setattr(mod, "_raise_if_non_interactive", lambda lead: None)

    async def instant_sleep(_seconds):
        pass

    waiter = mod._make_callback_waiter(mod._reserve_callback_port(), cimd_url)
    monkeypatch.setattr(mod.asyncio, "sleep", instant_sleep)
    with pytest.raises(mod.OAuthNonInteractiveError) as excinfo:
        asyncio.run(waiter())
    return str(excinfo.value)


def test_timeout_on_a_cimd_flow_names_the_document_and_the_escape_hatch(monkeypatch):
    """A server that can't validate the document aborts at the authorization
    endpoint (draft section 5.1), so no redirect ever arrives and the only
    symptom Hermes sees is the callback timing out."""
    message = _timed_out_waiter_message(monkeypatch, _CIMD_CLIENT_METADATA_URL)

    assert _CIMD_CLIENT_METADATA_URL in message
    assert "cimd: false" in message


def test_timeout_without_cimd_stays_quiet_about_it(monkeypatch):
    assert "cimd" not in _timed_out_waiter_message(monkeypatch, None).lower()
