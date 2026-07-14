"""Exact-call authorization capability for ``ucm_structured_process`` (Phase 1).

This module is the sole mint factory for a sealed, one-shot capability used by
the future UCM structured-process tool. Phase 1 ships the capability and its
tests only — no production dispatch site may mint yet (enforced by the
mint-site census test).

Security properties (authoritative spec Section 16.3/16.4):
- private sealed type + module-private construction seal
- immutable enabled-tool snapshot at mint time
- fixed expected tool name ``ucm_structured_process``
- atomic one-shot ``consume`` under a lock
- owner-side ``invalidate`` (works even if never consumed)
- exact provenance (no duck typing / subclass authorization)
- copy/deepcopy cannot create a second independently authorizing object
- pickle/JSON reconstruction rejected
- redacted ``repr`` / ``str``
- no TTL, no provider-id auth, no global/ContextVar/session storage
"""

from __future__ import annotations

import threading
from typing import Any, Iterable, Optional

__all__ = [
    "EXPECTED_TOOL_NAME",
    "is_ucm_auth_capability",
    "mint_ucm_auth_context",
]

EXPECTED_TOOL_NAME = "ucm_structured_process"

# Non-exported construction seal. Deliberate external construction without this
# object must fail.
_MINT_SEAL = object()

_REDACTED_REPR = "<UcmAuthCapability redacted>"


class _UcmAuthCapability:
    """Sealed one-shot authorization capability.

    Not part of the public API. Construct only via ``mint_ucm_auth_context``.
    """

    __slots__ = (
        "_active",
        "_consumed",
        "_enabled",
        "_tool_call_id",
        "_lock",
    )

    def __init__(
        self,
        enabled_tools: Iterable[str],
        tool_call_id: Optional[str] = None,
        *,
        _seal: Any = None,
    ) -> None:
        if _seal is not _MINT_SEAL:
            raise TypeError(
                "UcmAuthCapability cannot be constructed directly; "
                "use mint_ucm_auth_context()"
            )
        self._enabled: frozenset[str] = frozenset(enabled_tools)
        self._tool_call_id: Optional[str] = tool_call_id
        self._active: bool = True
        self._consumed: bool = False
        self._lock = threading.Lock()

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        raise TypeError("UcmAuthCapability cannot be subclassed for authorization")

    def consume(self, tool_name: str) -> bool:
        """Atomically attempt one-shot authorization for ``tool_name``.

        Returns True only when all of the following hold under the lock:
        - capability is still active
        - not already consumed
        - ``tool_name`` equals ``EXPECTED_TOOL_NAME``
        - ``EXPECTED_TOOL_NAME`` is in the mint-time enabled-tool snapshot
        """
        with self._lock:
            if not self._active:
                return False
            if self._consumed:
                return False
            if tool_name != EXPECTED_TOOL_NAME:
                return False
            if EXPECTED_TOOL_NAME not in self._enabled:
                return False
            self._consumed = True
            return True

    def invalidate(self) -> None:
        """Owner-side unconditional invalidation (idempotent)."""
        with self._lock:
            self._active = False

    # --- anti-copy / anti-forgery -------------------------------------------

    def __copy__(self) -> "_UcmAuthCapability":
        # Share exact one-shot state: copy is not an independent authorization.
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> "_UcmAuthCapability":
        return self

    def __getstate__(self) -> None:
        raise TypeError("UcmAuthCapability cannot be pickled")

    def __setstate__(self, state: Any) -> None:  # noqa: ANN401
        raise TypeError("UcmAuthCapability cannot be unpickled")

    def __reduce__(self) -> None:
        raise TypeError("UcmAuthCapability cannot be pickled")

    def __reduce_ex__(self, protocol: int) -> None:
        raise TypeError("UcmAuthCapability cannot be pickled")

    def __repr__(self) -> str:
        return _REDACTED_REPR

    def __str__(self) -> str:
        return _REDACTED_REPR


def mint_ucm_auth_context(
    enabled_tools: Iterable[str] | None,
    tool_call_id: Optional[str] = None,
) -> _UcmAuthCapability:
    """Mint a sealed one-shot capability for ``ucm_structured_process``.

    Phase 1: public factory exists for unit tests and future audited mint sites.
    Production call-site census must remain empty until a later authorized phase
    wires M1/M2/M3. The factory itself has no location gate (case 120); the
    mint-site census test is the control against unauthorized call sites.

    ``tool_call_id`` is correlation metadata only and is never used for
    authorization decisions.
    """
    if enabled_tools is None:
        snapshot: Iterable[str] = ()
    else:
        snapshot = enabled_tools
    return _UcmAuthCapability(snapshot, tool_call_id=tool_call_id, _seal=_MINT_SEAL)


def is_ucm_auth_capability(value: Any) -> bool:  # noqa: ANN401
    """Exact provenance check — True only for the factory's direct product."""
    return type(value) is _UcmAuthCapability
