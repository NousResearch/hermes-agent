"""Peer route refresh and status tracking."""

from gateway.hosted_room_peer import GatewayRoomCatalog, HostedMemberDispatch, room_grant_needs_dispatch_refresh
from tui_gateway.hosted_room_peer_http import digest_reauthorization_error


def _hook(obj: Any, name: str):
    """Optional callable attribute of a duck-typed peer client, or None."""
    value = getattr(obj, name, None)
    return value if callable(value) else None


class _RouteStatusPeerClient:
    """Classify scoped-auth failures without exposing route credentials."""

    def __init__(
        self, client, *, on_ready, on_reauthorization, on_unavailable, on_refreshed) -> None:
        self._client, self._on_ready, self._on_refreshed = client, on_ready, on_refreshed
        self._on_reauthorization, self._on_unavailable = on_reauthorization, on_unavailable

    def _refresh_grant(self, kwargs: dict) -> dict:
        """Rotate an expiring grant before dispatch; return the kwargs to send. Refresh
        failures escalate to reauthorization only when the peer says so or the grant is
        past its hard expiry; otherwise the original grant is tried as-is. A refreshed
        catalog whose digests drift from the dispatch is a policy change: refused."""
        grant = kwargs["grant"]
        if not room_grant_needs_dispatch_refresh(grant):
            return kwargs
        checked = HostedMemberDispatch.from_mapping(kwargs["dispatch"])
        refresh = _hook(self._client, "refresh_grant")
        if refresh is None:
            return kwargs
        try:
            refreshed = refresh(
                grant=grant, capability_digest=checked.capability_digest,
                execution_policy_digest=checked.execution_policy_digest)
        except Exception as exc:
            if getattr(exc, "needs_reauthorization", False) or (
                room_grant_needs_dispatch_refresh(grant, leeway_seconds=0)):
                self._on_reauthorization()
                raise
            return kwargs
        replacement = str(refreshed.get("grant") or "")
        if not replacement:
            raise RuntimeError("peer returned no refreshed room grant")
        refreshed_catalog = None
        if refreshed.get("catalog") is not None:
            refreshed_catalog = GatewayRoomCatalog.from_mapping(refreshed.get("catalog"))
            drift = digest_reauthorization_error(
                refreshed_catalog, capability_digest=checked.capability_digest,
                execution_policy_digest=checked.execution_policy_digest)
            if drift is not None:
                self._on_reauthorization()
                raise drift
        self._on_refreshed(replacement, refreshed_catalog)
        return {**kwargs, "grant": replacement}

    def __getattr__(self, name):
        value = getattr(self._client, name)
        if not callable(value):
            return value

        def tracked(*args, **kwargs):
            if name in {"dispatch", "recover_dispatch"} and "grant" in kwargs:
                kwargs = self._refresh_grant(kwargs)
            try:
                result = value(*args, **kwargs)
            except Exception as exc:
                if getattr(exc, "needs_reauthorization", False):
                    self._on_reauthorization()
                elif getattr(exc, "not_admitted", False):
                    self._on_unavailable()
                raise
            if name != "prepare":
                self._on_ready()
            return result
        return tracked
