"""Canonical Bot Chat transport and durable legacy delivery receipts.

Only gateway/session_bot.py admits execution. Old claimed/terminal records remain
inspectable; the legacy UI claim consumer is retired. No authority means refusal.
Receipt storage derives from FalconOrtiz's live-owner mailbox (#101564).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from contextlib import ExitStack, contextmanager
from hermes_dir_acl import make_user_dir, unusable_dir_remedy

from utils import atomic_json_write, fsync_directory
from pathlib import Path
from typing import Any, Callable

from hermes_cli.active_sessions import _FileLock

log = logging.getLogger(__name__)

DELIVERY_DIR_NAME = "bot_live_delivery"
_OWNER_KEYS = ("profile_home", "session_id", "lease_id", "live_session_id")
_TERMINAL = frozenset({"settled", "failed", "cancelled", "ambiguous"})


def _bot_chat_tip(home: Path) -> str | None:
    """Current Bot Chat tip id, or None when the profile has no store or no Bot Chat yet."""
    from persistence_assembly import open_persistence_reader, profile_scope, store_location

    location = store_location(profile_scope(home=home))
    if location is None or not location.is_file():
        return None
    with open_persistence_reader(home) as store:
        row = store.sessions.get_by_title('Bot Chat')
        return store.lineage.tip(row.session_id) if row else None


def find_canonical_live_owner(profile_home: Path | str) -> dict[str, Any] | None:
    """Discover the profile authority and its exact Bot Chat without acquiring a lease.

    ``session_id`` is the current Bot Chat tip, or ``''`` when the profile has none yet: the
    authority's deliver door creates it on first delivery (main's ``--create-if-missing``),
    so a missing chat is not a refusal. None only when the profile has no store at all."""
    from hermes_cli.gateway_runtime import discover_gateway_endpoint
    from persistence_assembly import profile_scope, store_location

    home = Path(profile_home).resolve()
    discovery = discover_gateway_endpoint(home, timeout=5)
    if discovery.state != 'ready' or discovery.endpoint is None:
        raise ValueError('profile authority is not ready')
    location = store_location(profile_scope(home=home))
    if location is None or not location.is_file():
        return None
    tip = _bot_chat_tip(home)
    return dict(profile_home=str(home), session_id=tip or '', canonical=True,
                lease_id=discovery.endpoint.instance_id, live_session_id=tip or '')


class PreDialError(ValueError):
    """The authority call died before its RPC reached the socket: nothing was admitted, so
    this is a definite no-send the sender may safely retry — never the ambiguous
    post-admission state that orders 'do not resend' (#139303). A ``ValueError`` like the
    "authority is not ready" refusals it wraps, so existing ``except ValueError`` callers hold."""


def authority_delivery(home, params):
    """Call only this home's already-running authority; never start a fallback."""
    return _authority_rpc(home, 'bot_relay.deliver', params)


def authority_settle(home, params):
    """The same authority's ``bot_relay.settle``: the delivery id settled as never sent, or its
    existing receipt (``gateway/session_bot.py:_settle``)."""
    return _authority_rpc(home, 'bot_relay.settle', params)


def _authority_rpc(home, method, params):
    import asyncio
    from hermes_cli.gateway_runtime import discover_gateway_endpoint
    from hermes_cli.gateway_client import GATEWAY_WS_PROTOCOL, GatewayClient, _session_ticket, gateway_ws_target
    from websockets.asyncio.client import connect

    home = Path(home).resolve()
    rpc_started = False

    async def request():
        nonlocal rpc_started
        discovery = await asyncio.to_thread(discover_gateway_endpoint, home, timeout=5)
        if discovery.state != 'ready' or discovery.endpoint is None:
            raise ValueError('profile authority is not ready')
        endpoint = discovery.endpoint
        ticket = await asyncio.to_thread(_session_ticket, home, endpoint)
        url, protocols = gateway_ws_target(endpoint, ticket)
        # Loopback authority dial: never through HTTP(S)_PROXY (websockets>=14 honours it by default).
        async with connect(url, subprotocols=protocols, open_timeout=10, proxy=None) as ws:
            if ws.subprotocol != GATEWAY_WS_PROTOCOL:
                raise ValueError('authority protocol mismatch')
            async with GatewayClient(ws) as client:
                rpc_started = True  # the RPC is going out: past here a failure may have been admitted
                return await client.rpc(method, **params)

    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(request())
        # A gateway-hosted tool call runs on a thread with a live loop, where asyncio.run() raises
        # before anything is dialed (#139303).
        from model_tools import _run_async
        return _run_async(request())
    except Exception as exc:
        if not rpc_started:
            raise PreDialError(str(exc)) from exc
        raise


def _owner(home: Path | str, owner: dict[str, Any]) -> dict[str, str]:
    pinned = {key: owner.get(key) for key in _OWNER_KEYS}
    # An empty session_id is the discovered "no Bot Chat yet" state; the door creates it.
    if not all(isinstance(value, str) and (value or key in ("session_id", "live_session_id"))
               for key, value in pinned.items()):
        raise ValueError("owner requires profile_home, session_id, lease_id and live_session_id")
    if pinned["profile_home"] != str(Path(home).resolve()):
        raise ValueError("owner belongs to a different profile home")
    return pinned


def _delivery_id(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32,64}", value) is None:
        raise ValueError("delivery id must be 32 to 64 lowercase hex characters")
    return value


def _root(home: Path | str) -> Path:
    return Path(home).resolve() / "runtime" / DELIVERY_DIR_NAME


_warned_unhardened: set[Path] = set()
# Mailbox dirs already reported unusable by this process: WARNING once per dir, DEBUG after,
# because the receipt readers re-enter ``_locked`` while Bot Chat is waiting.
_warned_unusable: set[Path] = set()


class MailboxUnusable(PermissionError):
    """This token cannot list, create or lock inside the mailbox dir (#139485, #138872).

    On Windows an elevated install/update creates ``runtime/bot_live_delivery`` owned by
    Administrators with no entry for the user; the same user's non-elevated gateway then
    cannot even open ``.lock``. Bot Chat deliveries pause; nothing else depends on the dir.
    A ``PermissionError`` subclass so every existing caller keeps its handling.
    """


def _dir_owner(path: Path) -> str:
    """Best-effort owner name for the log line; never raises (an ACL that hides it is the reported case)."""
    if os.name == "nt":
        try:
            import pywintypes
            import win32security
        except ImportError as exc:
            return f"unknown ({exc})"
        try:
            descriptor = win32security.GetFileSecurity(str(path), win32security.OWNER_SECURITY_INFORMATION)
            name, domain, _ = win32security.LookupAccountSid(None, descriptor.GetSecurityDescriptorOwner())
        except pywintypes.error as exc:
            return f"unknown ({exc})"
        return f"{domain}\\{name}" if domain else name
    import pwd  # windows-footgun: ok — POSIX branch only

    try:
        uid = path.stat().st_uid
    except OSError as exc:
        return f"unknown ({exc})"
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return f"uid {uid}"


def _unusable(root: Path, exc: PermissionError) -> MailboxUnusable:
    remedy = unusable_dir_remedy(root)
    level = logging.DEBUG if root in _warned_unusable else logging.WARNING
    _warned_unusable.add(root)
    log.log(level, "bot_live_delivery: mailbox directory %s (owner %s) is not usable by this process (%s); "
            "Bot Chat deliveries are paused until it is repaired, other clients are unaffected. To repair, %s",
            root, _dir_owner(root), exc, remedy)
    return MailboxUnusable(exc.errno, f"bot mailbox directory is not usable: {exc}", str(root))


def _prepare(root: Path) -> Path:
    """Create/harden the mailbox dir and its ``.lock`` file; returns the lock path."""
    created = not root.is_dir()
    make_user_dir(root.parent, parents=True, exist_ok=True)
    make_user_dir(root, mode=0o700, exist_ok=True)
    try:
        root.chmod(0o700)
    except OSError as exc:
        # On POSIX chmod fails only when another uid owns the dir: refuse a mailbox someone else controls.
        if os.name != "nt" and root.stat().st_uid != os.getuid():  # windows-footgun: ok — POSIX only
            raise
        # A foreign Windows ACL can deny FILE_WRITE_ATTRIBUTES even when this process can
        # still use the existing mailbox. Hardening must not abort recovery. The owner's
        # receipt reads and writes re-enter this lock, so WARNING once per dir and DEBUG after.
        level = logging.DEBUG if root in _warned_unhardened else logging.WARNING
        _warned_unhardened.add(root)
        log.log(level, "bot_live_delivery: could not harden mailbox directory %s (%s)", root, exc)
    if created:
        # Only a fresh mailbox dir needs its parents durably linked; the owner's
        # receipt reads and writes re-enter this lock, and two directory fsyncs
        # per call would be disk churn for nothing.
        fsync_directory(root.parent)
        fsync_directory(root.parent.parent)
    lock = root / ".lock"
    fd = os.open(lock, os.O_CREAT | os.O_WRONLY, 0o600)
    os.close(fd)
    return lock


@contextmanager
def _locked(home: Path | str):
    """Hold the mailbox dir lock. A dir this token cannot use raises ``MailboxUnusable``
    (logged once per dir with its owner and the remedy); errors inside the body pass through."""
    root = _root(home)
    with ExitStack() as stack:
        try:
            stack.enter_context(_FileLock(_prepare(root)))
        except PermissionError as exc:
            raise _unusable(root, exc) from exc
        _warned_unusable.discard(root)
        yield root


def _read(path: Path) -> dict[str, Any] | None:
    """Exact-id read: absent → None; unreadable or not a JSON object → raises (callers fail closed)."""
    try:
        record = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return None
    if not isinstance(record, dict):
        raise ValueError(f"ticket {path.name} is not a JSON object ({type(record).__name__})")
    return record


def _ticket_shape_error(path: Path, record: dict[str, Any]) -> str | None:
    """Why a parsed ticket is unusable by the scans, or None when it is well-formed.

    A ticket that parses as JSON but lost a field (truncated rewrite, foreign
    writer, hand edit) would otherwise raise KeyError/TypeError out of a
    directory scan and wedge delivery for the whole profile.
    """
    owner = record.get("owner")
    created_at, sequence = record.get("created_at"), record.get("sequence", record.get("created_at"))
    if record.get("delivery_id") != path.stem or record.get("id") != path.stem:
        return "id does not match filename"
    status = record.get("status")
    if not isinstance(status, str) or status not in ({"queued", "claimed"} | _TERMINAL):
        return f"unknown status {status!r}"
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (created_at, sequence)):
        return "created_at/sequence are not integers"
    if not isinstance(owner, dict) or not all(isinstance(owner.get(k), str) and owner[k] for k in _OWNER_KEYS):
        return "owner pin is incomplete"
    return None


def _write(path: Path, record: dict[str, Any]) -> None:
    atomic_json_write(path, record, indent=None, sort_keys=True, fsync_dir=True, mode=0o600)


def deliver_to_live_owner(
    profile_home: Path | str, owner: dict[str, Any], message: str,
    *, delivery_id: str | None = None, author: dict[str, Any] | None = None,
    notification_category: str = "result",
) -> dict[str, Any]:
    """Return durable admission immediately, without waiting for the owner.

    Retry with the same id AND pinned owner/message to inspect the existing
    state. Reusing an id with a different payload is an error, never an overwrite.
    """
    return _door(authority_delivery, profile_home, owner, message, delivery_id=delivery_id, author=author,
                 notification_category=notification_category)


def settle_held_delivery(profile_home: Path | str, owner: dict[str, Any], message: str, *, delivery_id: str,
                         author: dict[str, Any] | None = None) -> dict[str, Any]:
    """Finalize a refusal marked ``admitted: false`` under the same identity: the owner's terminal
    not-sent receipt (``refused``), or the receipt of an attempt that admitted the id meanwhile."""
    return _door(authority_settle, profile_home, owner, message, delivery_id=delivery_id, author=author)


def _door(call: Callable[[Path, dict], dict], profile_home, owner, message, *, delivery_id=None, author=None,
          notification_category="result"):
    pinned = _owner(profile_home, owner)
    if not isinstance(message, str):
        raise ValueError("message must be a string")
    if notification_category not in ('result', 'diagnostic'):
        raise ValueError("invalid notification category")
    home = Path(profile_home).resolve()
    category = {"notification_category": "diagnostic"} if notification_category == "diagnostic" else {}
    # The authority refuses a category-changing replay (admission_conflict), so the admitted
    # category is the producer's; echo it so callers can check the receipt's identity.
    return {**call(home, dict(id=_delivery_id(delivery_id if delivery_id is not None else uuid.uuid4().hex),
        profile=home.name if home.parent.name == "profiles" else "default",
        message=message, **({"session_id": pinned["session_id"]} if pinned["session_id"] else {}),
        **({"author": dict(author)} if author else {}), **category)), **category}


def owner_holds_delivery(profile_home: Path | str, record: dict) -> bool:
    """Whether the profile authority still holds ``record`` (main's bounded DM wait asks this).

    An admission lives in the authority's durable FIFO, so it is held while the authority is
    reachable; an unreachable one raises (outcome unknown, never cancelled). Only an unadmitted
    legacy record whose profile has no store is provably ownerless.
    """
    owner = find_canonical_live_owner(profile_home)  # raises while the authority is unreachable
    return owner is not None or bool(record.get("admission_id"))


def cancel_queued_delivery(
    profile_home: Path | str, delivery_id: str, *, error: str, reason: str,
) -> dict[str, Any] | None:
    """Cancel an ownerless queued receipt without racing a consumer's claim.

    A claim or terminal result that won the mailbox lock is returned unchanged.
    """
    key = _delivery_id(delivery_id)
    with _locked(profile_home) as root:
        path = root / f"{key}.json"
        record = _read(path)
        # An authority admission lives in its FIFO: only the authority may settle it.
        if record is None or record["status"] != "queued" or record.get("admission_id"):
            return record
        record.update(status="cancelled", reply="", error=error, reason=reason,
                      completed_at=time.time_ns())
        _write(path, record)
        return record


def read_delivery_result(profile_home: Path | str, delivery_id: str) -> dict[str, Any] | None:
    """Read admission/claim/terminal state without waiting or deleting its receipt."""
    key = _delivery_id(delivery_id)
    if not _root(profile_home).is_dir():
        return None
    # atomic_replace can fall back to an in-place rewrite on Windows. Use the writer's
    # short lock so local pollers never open a partially rewritten or sharing-locked file.
    with _locked(profile_home) as root:
        record = _read(root / f'{key}.json')
    if record is not None and record.get('admission_id') and record.get('status') == 'canonical':
        # Older owners used this transitional spelling. Publication/recovery updates it;
        # a reader never dials the owner or replays an admission just to inspect its receipt.
        return {**record, 'status': 'queued'}
    return record


_PENDING = ("queued", "claimed")
_POLL_SECONDS = 0.5


def _poll_receipt(profile_home: Path | str, delivery_id: str) -> dict[str, Any] | None:
    """A waiter's read: the receipt file the owner republishes when the admission settles (and on
    restart recovery). Never an authority RPC: a wait outlives owner restarts, and replaying the
    envelope (``read_delivery_result``) is for producers re-confirming their admission."""
    record = _read(_root(profile_home) / f"{_delivery_id(delivery_id)}.json")
    if record is not None and record.get("status") == "canonical":
        # Admitted with its outcome not yet published (an older owner's last write): pending.
        return {**record, "status": "queued"}
    return record


def await_delivery(
    profile_home: Path | str, delivery_id: str, timeout: float | None,
    *, should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any] | None:
    """Poll a receipt until the owner settles it, ``timeout`` lapses, or ``should_stop`` says so.

    Every transport that hands a turn to a live Bot Chat owner (local ``message_agent``, the
    Desktop relay, ``hermes peer dm`` and ``hermes peer run``) waits on the same receipt; keeping
    the loop here is what stops the lanes drifting (one lane returned a receipt sentence instead
    of the reply, two never waited at all). Returns the last record read — still pending when the
    budget lapsed, None when the receipt was never readable. Polls the local receipt file only, so
    an owner restart mid-wait is just a longer pending stretch.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        record = _poll_receipt(profile_home, delivery_id)
        if record is None or record["status"] not in _PENDING:
            return record
        if should_stop is not None and should_stop():
            return record
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            return record
        time.sleep(_POLL_SECONDS if remaining is None else min(_POLL_SECONDS, remaining))


async def await_delivery_async(
    profile_home: Path | str, delivery_id: str, timeout: float | None,
    *, should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any] | None:
    """``await_delivery`` for an event loop: never blocks a worker thread for the whole budget."""
    import asyncio

    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        record = await asyncio.to_thread(_poll_receipt, profile_home, delivery_id)
        if record is None or record["status"] not in _PENDING:
            return record
        if should_stop is not None and should_stop():
            return record
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            return record
        await asyncio.sleep(_POLL_SECONDS if remaining is None else min(_POLL_SECONDS, remaining))
