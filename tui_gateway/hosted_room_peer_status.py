"""Grant-bound status and renewal callbacks for RoomLink operations."""

from __future__ import annotations

import hashlib
import logging
from contextlib import contextmanager
from copy import copy

from gateway.hosted_room_peer import GatewayRoomCatalog, HostedMemberDispatch
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError

logger = logging.getLogger("tui_gateway.hosted_room_service")


@contextmanager
def _admission_preflight(method):
    """Report this call's phase without claiming the identity was never admitted.

    Only the fenced driver can prove its generation is fresh. The actual
    admission call and its callbacks must remain outside this block.
    """
    try:
        yield
    except Exception as exc:
        if method in {"dispatch", "recover_dispatch"}:
            dispatch_call = method == "dispatch"
            try:
                failure = copy(exc)
                if failure is exc:
                    raise TypeError("exception cannot carry private phase evidence")
                failure.not_admitted = False
                failure.dispatch_not_attempted = dispatch_call
                if not dispatch_call:
                    failure.ambiguous = True
            except Exception:
                failure = PeerRunsHTTPError(
                    "peer admission preflight failed",
                    not_admitted=False,
                    ambiguous=not dispatch_call or bool(getattr(exc, "ambiguous", False)),
                    retryable=bool(getattr(exc, "retryable", False)),
                    status_code=getattr(exc, "status_code", None),
                    error_code=getattr(exc, "error_code", None),
                )
                failure.dispatch_not_attempted = dispatch_call
            raise failure from exc
        raise


class _RouteStatusPeerClient:
    """Classify scoped-auth failures without exposing route credentials."""

    def __init__(
        self,
        client,
        *,
        on_ready,
        on_reauthorization,
        on_unavailable,
        on_refreshed,
        grant=None,
        capability_digest="",
        execution_policy_digest="",
        before_admission=None,
    ) -> None:
        self._client = client
        self._on_ready = on_ready
        self._on_reauthorization = on_reauthorization
        self._on_unavailable = on_unavailable
        self._on_refreshed = on_refreshed
        self._initial_grant = grant
        self._current_grant = grant
        self._capability_digest = str(capability_digest or "")
        self._execution_policy_digest = str(execution_policy_digest or "")
        self._before_admission = before_admission

    def _notify(self, callback, grant):
        if self._initial_grant is None:
            return callback()
        return callback(
            expected_grant_sha256=hashlib.sha256(grant.encode()).hexdigest()
        )

    def __getattr__(self, name):
        value = getattr(self._client, name)
        if not callable(value):
            return value

        def tracked(*args, **kwargs):
            with _admission_preflight(name):
                observed_grant = kwargs.get("grant") or self._current_grant
                if (
                    self._initial_grant is not None
                    and kwargs.get("grant") == self._initial_grant
                ):
                    observed_grant = self._current_grant
                    kwargs = {**kwargs, "grant": observed_grant}
                if (
                    name
                    in {
                        "acknowledge_artifacts",
                        "discard_artifacts",
                        "dispatch",
                        "read_artifact",
                        "recover_dispatch",
                        "stage_attachments",
                    }
                    and "grant" in kwargs
                ):
                    from gateway.hosted_room_peer import (
                        room_grant_needs_dispatch_refresh,
                    )

                    grant = kwargs["grant"]
                    if self._before_admission is not None:
                        self._before_admission(grant)
                    if room_grant_needs_dispatch_refresh(grant):
                        checked = (
                            HostedMemberDispatch.from_mapping(kwargs["dispatch"])
                            if "dispatch" in kwargs
                            else None
                        )
                        capability_digest = (
                            checked.capability_digest
                            if checked is not None
                            else self._capability_digest
                        )
                        execution_policy_digest = (
                            checked.execution_policy_digest
                            if checked is not None
                            else self._execution_policy_digest
                        )
                        refresh = getattr(self._client, "refresh_grant", None)
                        if callable(refresh):
                            try:
                                refreshed = refresh(
                                    grant=grant,
                                    capability_digest=capability_digest,
                                    execution_policy_digest=execution_policy_digest,
                                )
                            except Exception as exc:
                                if bool(getattr(exc, "needs_reauthorization", False)):
                                    self._notify(self._on_reauthorization, observed_grant)
                                    raise
                                if room_grant_needs_dispatch_refresh(
                                    grant, leeway_seconds=0
                                ):
                                    self._notify(self._on_reauthorization, observed_grant)
                                    raise
                            else:
                                replacement = str(refreshed.get("grant") or "")
                                if not replacement:
                                    raise RuntimeError(
                                        "peer returned no refreshed room grant"
                                    )
                                try:
                                    refreshed_catalog = None
                                    if refreshed.get("catalog") is not None:
                                        refreshed_catalog = GatewayRoomCatalog.from_mapping(
                                            refreshed.get("catalog")
                                        )
                                        if (
                                            refreshed_catalog.execution_policy.policy_digest
                                            != execution_policy_digest
                                        ):
                                            raise PeerRunsHTTPError(
                                                "peer room execution policy needs reauthorization",
                                                status_code=403,
                                                error_code="room_execution_policy_changed",
                                                not_admitted=True,
                                            )
                                        if (
                                            refreshed_catalog.catalog_digest
                                            != capability_digest
                                        ):
                                            raise PeerRunsHTTPError(
                                                "peer room capabilities need reauthorization",
                                                status_code=403,
                                                error_code="room_capability_catalog_changed",
                                                not_admitted=True,
                                            )
                                    if self._initial_grant is None:
                                        self._on_refreshed(replacement, refreshed_catalog)
                                    else:
                                        self._on_refreshed(
                                            replacement,
                                            refreshed_catalog,
                                            expected_grant_sha256=hashlib.sha256(
                                                grant.encode()
                                            ).hexdigest(),
                                        )
                                        self._current_grant = replacement
                                except Exception:
                                    revoke = getattr(
                                        self._client, "revoke_grant_exact", None
                                    )
                                    try:
                                        self._notify(
                                            self._on_reauthorization, observed_grant
                                        )
                                    except Exception:
                                        logger.warning(
                                            "Could not persist peer reauthorization status"
                                        )
                                    try:
                                        if callable(revoke):
                                            revoke(grant=replacement)
                                        else:
                                            logger.warning(
                                                "Peer cannot retire an unpublished grant exactly"
                                            )
                                    except Exception:
                                        logger.warning(
                                            "Exact unpublished-grant cleanup could not be confirmed"
                                        )
                                    raise
                                kwargs = {**kwargs, "grant": replacement}
                                observed_grant = replacement
                    if self._before_admission is not None:
                        self._before_admission(kwargs["grant"])
            try:
                result = value(*args, **kwargs)
            except Exception as exc:
                if bool(getattr(exc, "needs_reauthorization", False)):
                    self._notify(self._on_reauthorization, observed_grant)
                    raise
                elif bool(getattr(exc, "not_admitted", False)):
                    self._notify(self._on_unavailable, observed_grant)
                    raise
                else:
                    raise
            if name != "prepare":
                self._notify(self._on_ready, observed_grant)
            return result

        return tracked
