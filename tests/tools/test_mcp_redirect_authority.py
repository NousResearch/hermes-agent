"""Cross-origin redirect credential authority for the MCP HTTP client.

The security boundary under test is the redirect chain, not a single hop. These
regressions drive the REAL ``httpx.AsyncClient`` redirect machinery
(``follow_redirects=True`` plus ``httpx.MockTransport``) rather than synthetic
response doubles, because a double cannot reproduce the condition that made the
original guard inert: with ``follow_redirects=True``, httpx never populates
``response.next_request``, so a response hook silently never fires on a redirect.
Only the real client exercises ``_send_handling_redirects``,
``_build_redirect_request`` and the actual hook ordering.
"""

import asyncio
from unittest.mock import patch

import httpx
import pytest

from tools.mcp_tool import _make_redirect_header_stripper

ORIGIN = "https://origin.example.test"
OTHER = "https://other.example.test"
SENSITIVE = {
    "Authorization": "Bearer super-secret",
    "X-Tenant": "tenant-secret",
    "X-User-Id": "alice",
}


class _Harness:
    """Builds a real AsyncClient wired exactly like the production client.

    The guard is registered as the only (or last) REQUEST hook, which is the
    final send boundary in httpx's redirect loop.
    """

    def __init__(
        self,
        routes,
        seen,
        *,
        strict=False,
        configured=frozenset(),
        identity_header_name=None,
        re_inject_hook=None,
    ):
        self.routes = routes
        self.seen = seen
        self.guard = _make_redirect_header_stripper(
            httpx.URL(ORIGIN),
            strict=strict,
            configured_header_names=configured,
            identity_header_name=identity_header_name,
        )
        # ``re_inject_hook`` simulates another request hook trying to re-add a
        # sensitive header. It is registered BEFORE the guard so the guard is
        # the last request hook and therefore the authoritative boundary.
        self._hooks = [re_inject_hook, self.guard] if re_inject_hook else [self.guard]

    async def __aenter__(self):
        harness = self

        async def handler(request):
            harness.seen.append((str(request.url), dict(request.headers)))
            status, location = harness.routes[str(request.url)]
            if location is None:
                return httpx.Response(status, request=request)
            return httpx.Response(
                status, headers={"location": location}, request=request
            )

        self.client = httpx.AsyncClient(
            follow_redirects=True,
            headers=dict(SENSITIVE, Accept="application/json"),
            transport=httpx.MockTransport(handler),
            event_hooks={"request": self._hooks},
        )
        return self.client

    async def __aexit__(self, *exc):
        await self.client.aclose()
        return False


def _strict_harness(routes, seen):
    return _Harness(
        routes,
        seen,
        strict=True,
        configured=frozenset({"x-tenant"}),
        identity_header_name="X-User-Id",
    )


def _headers_for(seen, url):
    for seen_url, headers in seen:
        if seen_url == url:
            return {k.lower(): v for k, v in headers.items()}
    raise AssertionError(f"no request was sent to {url}; saw {[u for u, _ in seen]}")


def _assert_stripped(headers):
    assert "authorization" not in headers, headers
    assert "x-tenant" not in headers, headers
    assert "x-user-id" not in headers, headers


def test_cross_origin_then_return_does_not_regain_authority():
    """A -> B -> A must not restore credentials on the return hop.

    This is the core P1: an original-origin-relative policy would see the final
    hop match the original origin and re-admit every secret.
    """
    seen = []
    routes = {
        f"{ORIGIN}/mcp": (302, f"{OTHER}/mcp"),
        f"{OTHER}/mcp": (302, f"{ORIGIN}/final"),
        f"{ORIGIN}/final": (200, None),
    }

    async def run():
        async with _strict_harness(routes, seen) as client:
            return await client.get(f"{ORIGIN}/mcp")

    response = asyncio.run(run())
    assert response.status_code == 200
    assert [url for url, _ in seen] == [
        f"{ORIGIN}/mcp",
        f"{OTHER}/mcp",
        f"{ORIGIN}/final",
    ]
    # Present on the first hop...
    first = _headers_for(seen, f"{ORIGIN}/mcp")
    assert first["authorization"] == SENSITIVE["Authorization"]
    assert first["x-tenant"] == SENSITIVE["X-Tenant"]
    assert first["x-user-id"] == SENSITIVE["X-User-Id"]
    # ...gone at the foreign origin...
    _assert_stripped(_headers_for(seen, f"{OTHER}/mcp"))
    # ...and STILL gone when the chain returns to the original origin.
    _assert_stripped(_headers_for(seen, f"{ORIGIN}/final"))


def test_same_origin_redirect_retains_client_authorization():
    """Same-origin OAuth control-plane redirects must keep working.

    A same-origin hop is not an origin change, so it must not taint the chain.
    """
    seen = []
    routes = {
        f"{ORIGIN}/token": (302, f"{ORIGIN}/token/callback"),
        f"{ORIGIN}/token/callback": (200, None),
    }

    async def run():
        async with _strict_harness(routes, seen) as client:
            return await client.get(f"{ORIGIN}/token")

    response = asyncio.run(run())
    assert response.status_code == 200
    for url in (f"{ORIGIN}/token", f"{ORIGIN}/token/callback"):
        headers = _headers_for(seen, url)
        assert headers["authorization"] == SENSITIVE["Authorization"], url
        assert headers["x-tenant"] == SENSITIVE["X-Tenant"], url


def test_reinjecting_request_hook_cannot_bypass_taint():
    """An earlier request hook re-adding a secret must still be stripped.

    The guard is registered as the LAST request hook, so it runs after every
    other request hook on each hop and is the authoritative send boundary.
    """
    seen = []
    routes = {
        f"{ORIGIN}/mcp": (302, f"{OTHER}/mcp"),
        f"{OTHER}/mcp": (200, None),
    }

    async def re_inject(request):
        request.headers["X-Tenant"] = SENSITIVE["X-Tenant"]
        request.headers["X-User-Id"] = SENSITIVE["X-User-Id"]
        request.headers["Authorization"] = SENSITIVE["Authorization"]

    async def run():
        async with _Harness(
            routes,
            seen,
            strict=True,
            configured=frozenset({"x-tenant"}),
            identity_header_name="X-User-Id",
            re_inject_hook=re_inject,
        ) as client:
            return await client.get(f"{ORIGIN}/mcp")

    response = asyncio.run(run())
    assert response.status_code == 200
    # The hostile hook ran on both hops...
    assert len(seen) == 2
    # ...and the guard still removed every secret on the tainted chain.
    _assert_stripped(_headers_for(seen, f"{OTHER}/mcp"))


def test_concurrent_chains_do_not_share_taint_state():
    """Taint is per-chain, so one tainted chain cannot strip a same-origin one."""
    seen = []
    routes = {
        f"{ORIGIN}/same": (302, f"{ORIGIN}/same/final"),
        f"{ORIGIN}/same/final": (200, None),
        f"{ORIGIN}/cross": (302, f"{OTHER}/cross"),
        f"{OTHER}/cross": (200, None),
    }

    async def run():
        async with _strict_harness(routes, seen) as client:
            await asyncio.gather(
                client.get(f"{ORIGIN}/same"),
                client.get(f"{ORIGIN}/cross"),
            )

    asyncio.run(run())
    # The same-origin chain kept its credentials...
    for url in (f"{ORIGIN}/same", f"{ORIGIN}/same/final"):
        headers = _headers_for(seen, url)
        assert headers["authorization"] == SENSITIVE["Authorization"], url
        assert headers["x-tenant"] == SENSITIVE["X-Tenant"], url
    assert headers["x-user-id"] == SENSITIVE["X-User-Id"], url
    # ...while the cross-origin chain lost them.
    _assert_stripped(_headers_for(seen, f"{OTHER}/cross"))


def test_default_mode_strips_only_authorization():
    """Non-strict mode keeps its documented default: only Authorization goes."""
    seen = []
    routes = {
        f"{ORIGIN}/mcp": (302, f"{OTHER}/mcp"),
        f"{OTHER}/mcp": (200, None),
    }

    async def run():
        async with _Harness(routes, seen) as client:
            return await client.get(f"{ORIGIN}/mcp")

    response = asyncio.run(run())
    assert response.status_code == 200
    headers = _headers_for(seen, f"{OTHER}/mcp")
    assert "authorization" not in headers, headers
    # Unrelated client-generated and non-strict package headers survive.
    assert headers["x-tenant"] == SENSITIVE["X-Tenant"], headers
    assert headers["x-user-id"] == SENSITIVE["X-User-Id"], headers
    assert headers["accept"] == "application/json", headers


def test_identity_header_stripped_even_when_not_strict():
    """Identity/tenant authority is stripped after taint regardless of strict."""
    seen = []
    routes = {
        f"{ORIGIN}/mcp": (302, f"{OTHER}/mcp"),
        f"{OTHER}/mcp": (200, None),
    }

    async def run():
        async with _Harness(
            routes,
            seen,
            strict=False,
            identity_header_name="X-User-Id",
        ) as client:
            return await client.get(f"{ORIGIN}/mcp")

    response = asyncio.run(run())
    assert response.status_code == 200
    headers = _headers_for(seen, f"{OTHER}/mcp")
    assert "authorization" not in headers, headers
    assert "x-user-id" not in headers, headers
    # A non-strict package header is still not a strict-set member.
    assert headers["x-tenant"] == SENSITIVE["X-Tenant"], headers


def test_http_to_https_upgrade_taints_chain():
    """An http -> https upgrade of the same host is an authority change.

    httpx's own ``_redirect_headers`` deliberately exempts this upgrade from
    stripping ``Authorization``. That exemption is looser than the policy
    required here, so the guard must still taint.
    """
    seen = []
    routes = {
        "http://origin.example.test/mcp": (301, "https://origin.example.test/mcp"),
        "https://origin.example.test/mcp": (200, None),
    }

    async def run():
        async with _strict_harness(routes, seen) as client:
            return await client.get("http://origin.example.test/mcp")

    response = asyncio.run(run())
    assert response.status_code == 200
    assert len(seen) == 2, seen
    _assert_stripped(_headers_for(seen, "https://origin.example.test/mcp"))


def test_max_redirects_does_not_leak_secret():
    """A long tainted chain never re-admits the secret, even when truncated."""
    seen = []
    routes = {f"{OTHER}/hop{i}": (302, f"{OTHER}/hop{i + 1}") for i in range(6)}
    routes[f"{OTHER}/hop6"] = (200, None)
    routes[f"{ORIGIN}/mcp"] = (302, f"{OTHER}/hop0")

    async def run():
        async with _strict_harness(routes, seen) as client:
            return await client.get(f"{ORIGIN}/mcp")

    response = asyncio.run(run())
    assert response.status_code == 200
    assert len(seen) == 8, seen
    for url, _ in seen[1:]:
        _assert_stripped(_headers_for(seen, url))


@pytest.mark.parametrize("strict", [True, False])
def test_guard_is_a_noop_for_a_direct_single_hop_request(strict):
    """No redirect means no origin change, so nothing is stripped."""
    seen = []
    routes = {f"{ORIGIN}/mcp": (200, None)}

    async def run():
        async with _Harness(routes, seen, strict=strict) as client:
            return await client.get(f"{ORIGIN}/mcp")

    response = asyncio.run(run())
    assert response.status_code == 200
    headers = _headers_for(seen, f"{ORIGIN}/mcp")
    assert headers["authorization"] == SENSITIVE["Authorization"]
    assert headers["x-tenant"] == SENSITIVE["X-Tenant"]
    assert headers["x-user-id"] == SENSITIVE["X-User-Id"]


# ---------------------------------------------------------------------------
# The preflight probe: a SECOND egress path, reached through run()
# ---------------------------------------------------------------------------
#
# run() calls _preflight_content_type() BEFORE _run_http(), and that probe
# builds its OWN httpx.AsyncClient(follow_redirects=True). Every test above
# exercises only the guarded client wired inside _run_http(), which is why a
# cross-origin redirect during the probe leaked the package header on the
# reviewed revision. These drive the real run() entry point with a
# MockTransport wired into the probe's client, so the genuine
# _send_handling_redirects / _build_redirect_request machinery executes.

PKG_ORIGIN = "https://pkg.example.test"
FOREIGN = "https://attacker.example.test"
PKG_SECRET = "package-secret"


def _two_hop_probe_handler(seen, *, final_origin):
    """Endpoint A 302s cross-origin to B; B 302s back to ``final_origin``."""

    def handler(request):
        url = str(request.url)
        seen.append((url, {k.lower(): v for k, v in request.headers.items()}))
        if url == f"{PKG_ORIGIN}/mcp":
            return httpx.Response(
                302, headers={"location": f"{FOREIGN}/mcp"}, request=request
            )
        if url == f"{FOREIGN}/mcp":
            return httpx.Response(
                302, headers={"location": f"{final_origin}/final"}, request=request
            )
        return httpx.Response(
            200, headers={"content-type": "application/json"}, request=request
        )

    return handler


def _headers_at(seen, url):
    for seen_url, headers in seen:
        if seen_url == url:
            return headers
    raise AssertionError(f"no probe request reached {url}; saw {[u for u, _ in seen]}")


def _run_probe_through_run(config, handler, captured_clients):
    """Call MCPServerTask.run() with the probe's client bound to *handler*.

    ``_run_http`` is stubbed so the assertion stays on the probe, but
    everything up to and including the probe invocation is production code:
    ``run()`` validates the URL, decides the probe applies, and forwards
    ``strict_redirect_headers`` / configured header names / the identity
    header name.
    """
    from tools.mcp_tool import MCPServerTask

    original_init = httpx.AsyncClient.__init__

    def patched_init(self_client, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        kwargs["follow_redirects"] = True
        captured_clients.append(dict(kwargs))
        original_init(self_client, **kwargs)

    async def stub_run_http(self_srv, _config):
        self_srv._ready.set()
        self_srv._shutdown_event.set()
        await self_srv._shutdown_event.wait()
        return "stopped"

    async def _drive():
        task = MCPServerTask("preflight_srv")
        with patch.object(MCPServerTask, "_run_http", stub_run_http), patch.object(
            httpx.AsyncClient, "__init__", patched_init
        ):
            await task.run(config)
        return task

    return asyncio.run(_drive())


def test_preflight_probe_does_not_forward_package_header_cross_origin():
    """Review #92906 regression: a 302 during preflight leaked X-Tenant.

    run() probes before _run_http() ever builds its guarded client. The probe
    followed redirects on an unguarded client, so a strict portable package's
    configured header reached whatever origin the 302 named.
    """
    seen = []
    captured = []
    task = _run_probe_through_run(
        {
            "url": f"{PKG_ORIGIN}/mcp",
            "headers": {"X-Tenant": PKG_SECRET},
            "strict_redirect_headers": True,
        },
        _two_hop_probe_handler(seen, final_origin=FOREIGN),
        captured,
    )

    # The probe really ran on a redirect-following client (not skipped).
    assert len(captured) == 1, captured
    assert captured[0]["follow_redirects"] is True
    # Credential present on the first hop...
    assert _headers_at(seen, f"{PKG_ORIGIN}/mcp")["x-tenant"] == PKG_SECRET
    # ...gone at the foreign origin.
    assert "x-tenant" not in _headers_at(seen, f"{FOREIGN}/mcp")
    assert task._error is None


def test_preflight_probe_strips_authorization_and_identity_header():
    """The probe boundary covers Authorization and the identity header too."""
    seen = []
    task = _run_probe_through_run(
        {
            "url": f"{PKG_ORIGIN}/mcp",
            "headers": {"Authorization": f"Bearer {PKG_SECRET}"},
            "identity_header": {
                "name": "X-User-Id",
                "value_from": "static",
                "value": PKG_SECRET,
            },
        },
        _two_hop_probe_handler(seen, final_origin=FOREIGN),
        [],
    )

    first = _headers_at(seen, f"{PKG_ORIGIN}/mcp")
    assert first["authorization"] == f"Bearer {PKG_SECRET}", first
    # The identity header is config-resolved credential authority, so it is
    # part of the stripped set even though it is not strict-mode config.
    redirected = _headers_at(seen, f"{FOREIGN}/mcp")
    assert "authorization" not in redirected, redirected
    assert "x-user-id" not in redirected, redirected
    assert task._error is None


def test_preflight_probe_return_hop_cannot_regain_authority():
    """A -> B -> A inside the probe stays tainted (monotonic per chain)."""
    seen = []
    _run_probe_through_run(
        {
            "url": f"{PKG_ORIGIN}/mcp",
            "headers": {"X-Tenant": PKG_SECRET},
            "strict_redirect_headers": True,
        },
        _two_hop_probe_handler(seen, final_origin=PKG_ORIGIN),
        [],
    )

    assert _headers_at(seen, f"{PKG_ORIGIN}/mcp")["x-tenant"] == PKG_SECRET
    assert "x-tenant" not in _headers_at(seen, f"{FOREIGN}/mcp")
    # Returning to the ORIGINAL origin must not restore the credential.
    assert "x-tenant" not in _headers_at(seen, f"{PKG_ORIGIN}/final")


def test_preflight_probe_same_origin_redirect_keeps_headers():
    """A same-origin hop in the probe is not an authority change."""
    seen = []

    def handler(request):
        url = str(request.url)
        seen.append((url, {k.lower(): v for k, v in request.headers.items()}))
        if url == f"{PKG_ORIGIN}/mcp":
            return httpx.Response(
                302, headers={"location": f"{PKG_ORIGIN}/mcp/"}, request=request
            )
        return httpx.Response(
            200, headers={"content-type": "application/json"}, request=request
        )

    _run_probe_through_run(
        {
            "url": f"{PKG_ORIGIN}/mcp",
            "headers": {"X-Tenant": PKG_SECRET},
            "strict_redirect_headers": True,
        },
        handler,
        [],
    )

    assert _headers_at(seen, f"{PKG_ORIGIN}/mcp")["x-tenant"] == PKG_SECRET
    # Same-origin redirect: no authority change, so the probe keeps the header
    # and still reaches the MCP-shaped final hop.
    assert _headers_at(seen, f"{PKG_ORIGIN}/mcp/")["x-tenant"] == PKG_SECRET
