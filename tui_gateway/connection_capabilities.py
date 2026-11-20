"""Applying a client's declared capabilities — the ONE seam both handshake paths share.

``client.capabilities`` (the sidecar's own probe, sent once per connection by every client that
predates the fold) and the authority's ``client.hello`` (the folded handshake) must apply the same
declaration to the same transport: the transport records who answers server→client requests, whether
this peer is the local Desktop, and the answer reports what this backend may send. Two copies of that
rule is how a peer ends up advertised on one path and stalled on the other, so the body lives here and
both callers import it.
"""

from __future__ import annotations


def apply_client_capabilities(transport, params: dict) -> dict:
    """Record *params* on *transport* and return the ``ClientCapabilitiesResult`` body.

    Each key updates only what it names: the Desktop sends ``desktop_local`` on its own, so a backend
    older than that field refuses just that call and keeps the ``server_requests`` it accepted. A
    transport without ``auth_identity`` (stdio, harnesses) never takes a desktop identity.
    """
    from hermes_cli.process_identity import local_desktop_client
    from tui_gateway import server_requests
    from tui_gateway.contracts import registry as contracts

    if transport is not None and "server_requests" in params:
        server_requests.advertise(transport, bool(params.get("server_requests")))
    if transport is not None and "desktop_local" in params and hasattr(transport, "auth_identity"):
        transport.desktop_local = local_desktop_client(transport.auth_identity, params.get("desktop_local"))
        if transport.desktop_local:
            from tui_gateway import server
            server.begin_local_desktop_boot()
    return {"server_requests": sorted(contracts.SERVER_REQUESTS), "declines_not_shown": True,
            "canonical_clarify_batch": True}
