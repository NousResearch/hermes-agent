"""Bot Mode agent-to-agent DM tool — ``message_agent``.

Lets a Bot Mode agent message a teammate (a profile on this install, an agent on
a registered peer gateway, or one on another Desktop-connected machine): the
target is validated against the live roster, the attribution prefix is applied
server-side, and the reply arrives later via the background-process completion
notification (fire-and-forget). Containment: the schema is injected ONLY into a
bot's canonical "Bot Chat" session on a Bot-Mode-managed install (same gate as
``tools/bot_mode_probe.py``; never in the registry or any toolset), and dispatch
re-checks that gate so a forged call returns a structured error. Transports:
local → ``hermes -p <name> chat --in ~ -c "Bot Chat" --create-if-missing -Q
--query-file <tmp>``; peer → ``hermes peer dm <peer>[/<name>] < <tmp>``; both via
``terminal_tool(background=True, notify_on_complete=True)``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Top-level imports stay stdlib-only: this module also runs directly as the background
# delivery runner (``python bot_mode_dm.py --run-delivery …``); Hermes helpers import lazily.

logger = logging.getLogger(__name__)

MESSAGE_AGENT_TOOL_NAME = "message_agent"

# Message body cap — generous for real work, small enough that a runaway paste can't
# turn one DM into a context bomb on the recipient.
MESSAGE_MAX_CHARS = 16000
# The delivery process's completion notification IS the reply: size it like a message, plus the
# runner's header and failure prose, instead of the 2000-char tail a build log gets.
REPLY_COMPLETION_CHARS = MESSAGE_MAX_CHARS + 2000
# A runner owns and removes each DM file; this bounds residual plaintext lifetime if
# the machine dies between spawn ack and the runner's finally.
_DM_DIR_NAME = "hermes-dm"
_DM_STALE_SECONDS = 24 * 60 * 60
_LIVE_WAIT_SECONDS = 300
# A live owner may run a turn past one wait window, but each waiting runner holds ~46 MB and the
# sender's notify slot: past this the runner reports pending (do not resend) and exits.
_LIVE_WAIT_MAX_SECONDS = 30 * 60

# '<peer>/<agent>' — peer names are lowercase (``hermes peer`` normalizes them).
_PEER_TARGET_RE = re.compile(r"^([a-z0-9][a-z0-9_-]{0,63})/([a-zA-Z0-9][a-zA-Z0-9_-]{0,63})$")
# Same shape as ``tools.bot_relay._HANDLE_RE`` (kept local: see import note above).
_LOCAL_TARGET_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
# Fan-out bound: one combined runner holds one sub-runner per target.
FAN_OUT_MAX = 8


def _default_home() -> str:
    from hermes_constants import get_process_hermes_home
    return str(get_process_hermes_home())


def message_agent_tool_schema() -> dict:
    """OpenAI-format schema for ``message_agent`` (injected, not registered)."""
    return {
        "type": "function",
        "function": {
            "name": MESSAGE_AGENT_TOOL_NAME,
            "description": (
                "Send a message to ANOTHER agent (teammate) on this install, or to an "
                "agent on a registered peer gateway. This is FIRE-AND-FORGET and "
                "asynchronous, like texting: it validates the target against the live "
                "roster, delivers your message into that agent's own Bot Chat with your "
                "attribution automatically prefixed, and returns immediately with a "
                "dispatch acknowledgement — status queued plus a delivery_id (the hand-off to a background "
                "delivery process, not a delivery receipt). It does NOT return their reply and you must "
                "not wait or poll for one — send it, finish your turn, and that process's "
                "completion notification wakes you with the outcome: their reply, or the "
                "delivery failure — unless the ack returns reply_delivery=\"poll\", in which case "
                "follow its process(action=\"wait\") instruction before ending the turn. COMPOSE the message yourself: write what YOU want to say to "
                "that agent (lead with the point; include the concrete ask or result). "
                "Never paste the user's words verbatim — paraphrase the actionable "
                "substance, and keep private 1:1 chat content private. Message one "
                "clearly relevant teammate when it genuinely helps the user's goal; "
                "don't fan out to several agents unless the user explicitly asked. "
                "Use the teammate roster in your system prompt (names + roles) to pick "
                "the right recipient; targets: a teammate name (e.g. 'researcher'), "
                "'<peer>/<agent>' for an agent on a registered peer gateway "
                "(e.g. 'spark/researcher', or just '<peer>' for the peer's main agent), "
                "or an agent on another connected machine from your roster (use "
                "'<handle>@<connection>' if the same handle exists on several)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": (
                            "Who to message: a teammate profile name from your roster "
                            "('researcher', 'hermes' for the default agent), or "
                            "'<peer>' / '<peer>/<agent>' for a registered peer gateway. "
                            f"Up to {FAN_OUT_MAX} comma-separated names fan out: one combined reply."
                        ),
                    },
                    "message": {
                        "type": "string",
                        "description": (
                            "The message YOU composed for that agent (max "
                            f"{MESSAGE_MAX_CHARS} chars). Do not include the "
                            "'Message from …' prefix — it is added automatically."
                        ),
                    },
                    "expects_reply": {
                        "type": "boolean",
                        "description": "false: notify only — no reply comes back and nothing wakes you.",
                    },
                    "cancel": {
                        "type": "string",
                        "description": "A delivery_id from your own ack: withdraw that queued request or stop the turn it started.",
                    },
                },
                "required": ["target"],
            },
        },
    }


def message_agent_authorized(agent: Any) -> bool:
    """The ``message_agent`` gate: a protocol-enabled agent whose session is a managed
    Bot-Mode canonical Bot Chat. Session-stable, so it is prompt-cache safe to re-evaluate
    on every tool-snapshot rebuild. Never raises."""
    try:
        if not getattr(agent, "_bot_mode_protocol", True):
            return False
        from tools.bot_mode_probe import BOT_CHAT_TITLE, is_bot_mode_managed

        # Managed-install check, NOT section non-emptiness: a SOUL.md carrying the
        # legacy protocol text gets an empty section but must still get the tool.
        return _session_title(agent) == BOT_CHAT_TITLE and is_bot_mode_managed(_agent_home(agent))
    except Exception:  # pragma: no cover — must never break a turn
        logger.debug("message_agent_authorized failed", exc_info=True)
        return False


def ensure_message_agent_tool(agent: Any) -> bool:
    """Inject the ``message_agent`` schema into a Bot Chat agent's tool list (once per turn).
    Idempotent and deterministic for the session's life (the gate is stable from the
    first turn), so the tool list is byte-identical across turns — prompt-cache safe. Never raises."""
    try:
        if not getattr(agent, "_bot_mode_protocol", True):
            return False
        tools = getattr(agent, "tools", None)
        present = bool(tools) and any(
            isinstance(t, dict) and t.get("function", {}).get("name") == MESSAGE_AGENT_TOOL_NAME
            for t in tools
        )
        if not present:
            if not message_agent_authorized(agent):
                return False
            if agent.tools is None:
                agent.tools = []
            agent.tools.append(message_agent_tool_schema())
        # Success means BOTH halves hold: a tool-surface rebuild (compaction, MCP refresh)
        # can keep the schema while valid_tool_names is republished without it, and an
        # advertised-but-nondispatchable tool sends the model hunting for shellouts (#96105).
        valid = getattr(agent, "valid_tool_names", None)
        if isinstance(valid, set):
            valid.add(MESSAGE_AGENT_TOOL_NAME)
        return True
    except Exception:  # pragma: no cover — must never break a turn
        logger.debug("ensure_message_agent_tool failed", exc_info=True)
        return False


def _resolve_local_name(target: str, roster: list[str], root: Path | None = None) -> Optional[str]:
    """Map a target to a local profile FOLDER id: 'hermes' → 'default'; an exact folder id
    (case-insensitive); else — when ``root`` is given — a friendly name or its Desktop @-slug
    (profile.yaml ``display_name`` / Bot Mode title: 'Scribe', '@scribe', 'Dr. Foo' → 'foo').
    Ambiguous friendly names resolve to None so a DM never lands on the wrong bot (#100671)."""
    want = target.strip().lower()
    if not want:
        return None
    if want == "hermes":
        return "default" if "default" in roster else None
    exact = next((name for name in roster if name.lower() == want), None)
    if exact is not None or root is None:
        return exact
    from tools.bot_mode_probe import alias_forms, local_alias_map

    aliases = local_alias_map(root)
    hits = set().union(*(aliases.get(form, set()) for form in alias_forms(want) | {want}))
    return next(iter(hits)) if len(hits) == 1 else None


def _err(message: str, *, roster: list[str] | None = None, peers: list[str] | None = None) -> str:
    from tools.bot_failure_reasons import classify_agent_error

    payload: dict[str, Any] = {"error": message, "reason": classify_agent_error(message)}
    if roster is not None:
        payload["teammates"] = roster
    if peers is not None:
        payload["peers"] = peers
    return json.dumps(payload)


@dataclass(frozen=True)
class Sender:
    """The calling bot and its live roster, resolved once per ``message_agent`` call."""

    home: str
    root: Path
    me: str
    roster_homes: dict
    peers: list
    teammates: list
    task_id: Optional[str]
    agent: Any

    def roster_err(self, msg: str) -> str:
        return _err(msg, roster=self.teammates, peers=self.peers)


def _wants_reply(value: Any) -> bool:
    """``expects_reply`` as models send it: a bool, or its string spelling."""
    return str(value).strip().lower() not in ("false", "0", "no")


def message_agent_tool(target: str = "", message: str = "", task_id: Optional[str] = None, agent: Any = None,
                       expects_reply: Any = True, cancel: str = "") -> str:
    """Deliver ``message`` to ``target``'s Bot Chat (several comma-separated targets fan out),
    or withdraw the caller's own request ``cancel``. Returns a JSON ack/error.
    ``agent`` is the calling AIAgent — used for the Bot Chat gate and sender identity."""
    home = _agent_home(agent)
    try:
        from tools.bot_mode_probe import (
            BOT_CHAT_TITLE, NO_REPLY_MARKER, _display_name, _handle, _hermes_root, _peers,
            _profile_name as _self_profile_name, _roster, is_bot_mode_managed,
        )

        if _session_title(agent) != BOT_CHAT_TITLE:
            return _err("message_agent is only available in a Bot Mode 'Bot Chat' session. "
                        "This session is not one; do not retry.")
        if not is_bot_mode_managed(home):
            return _err("This install is not Bot-Mode-managed (no bot roster); "
                        "message_agent is unavailable. Do not retry.")
    except Exception as exc:  # pragma: no cover — defensive
        return _err(f"Bot Mode gate check failed: {exc}")
    from hermes_cli.observability.shared_metrics_signals import record_feature_used
    record_feature_used("bot_mode", hermes_home=home)

    root, me = _hermes_root(Path(home)), _self_profile_name(Path(home))
    roster_homes = dict(_roster(root))
    peers = _peers(root)
    sender = Sender(home=home, root=root, me=me, roster_homes=roster_homes, peers=peers,
                    teammates=[_handle(n) for n in roster_homes if n != me], task_id=task_id, agent=agent)
    if str(cancel or "").strip():
        from tools.bot_mode_dm_coordination import cancel_own_delivery

        return cancel_own_delivery(str(cancel).strip(), str(target or ""), sender)

    body = str(message or "").strip()
    if not body:
        return _err("message is required — compose what you want to say to that agent.")
    if len(body) > MESSAGE_MAX_CHARS:
        return _err(f"message too long ({len(body)} chars > {MESSAGE_MAX_CHARS}). "
                    "Send the essentials; share large content as a file path instead.")

    from tools.bot_mode_dm_coordination import fan_out, split_targets

    targets = split_targets(target, sender)
    if not targets:
        return sender.roster_err("target is required.")
    if len(targets) > FAN_OUT_MAX:
        return _err(f"Too many targets ({len(targets)} > {FAN_OUT_MAX}); message the ones the task needs.")
    reply = _wants_reply(expects_reply)
    # Sender signature: the friendly name when the bot has one (#89720); the @handle stays the routing alias.
    content = (f"Message from 🤖 {_display_name(me, roster_homes.get(me, Path(home)))} (@{_handle(me)}): "
               + ("" if reply else f"{NO_REPLY_MARKER} ") + body)
    if len(targets) == 1:
        return _dispatch(targets[0], content, sender, wake="notify" if reply else "none")
    return fan_out(targets, content, sender, expects_reply=reply)


def _dispatch(raw_target: str, content: str, sender: Sender, *, wake: Any) -> str:
    """Resolve one target and hand ``content`` to its transport. ``wake`` is how the reply comes
    back: ``"notify"`` (a tracked runner whose completion wakes the sender), ``"none"`` (notify-only:
    nothing waits), or a list collecting the runner commands one fan-out runner joins."""
    from tools.bot_mode_probe import _handle, _profile_name as _self_profile_name
    from tools.bot_relay import _hermes_cli

    root, me, roster_homes, peers = sender.root, sender.me, sender.roster_homes, sender.peers
    roster = list(roster_homes)
    delivery = dict(task_id=sender.task_id, agent=sender.agent, wake=wake)
    # Attribution for the recipient's memory hooks; the text prefix stays the human-facing signature.
    author = {"id": f"bot:{me}", "name": _handle(me), "is_bot": True}

    # Peer target: '<peer>/<agent>' or a bare registered peer name.
    peer_match = _PEER_TARGET_RE.match(raw_target)
    if peer_match or raw_target.lower() in peers:
        peer_name, peer_profile = peer_match.groups() if peer_match else (raw_target.lower(), None)
        if peer_name not in peers:
            return sender.roster_err(f"No registered peer named '{peer_name}'.")
        dm_target = f"{peer_name}/{peer_profile}" if peer_profile else peer_name
        # A peer dm crosses installs: qualify the id with this host so the peer's own '<me>' stays distinct.
        from agent.turn_author import bot_author_id, local_origin
        peer_author = {**author, "id": bot_author_id(me, local_origin())}
        # Pin the registry-owning profile: `hermes peer` resolves bot_peers via the profile-scoped
        # load_config(), while the roster above reads the machine-root config — the CLI must run
        # in that same profile or a secondary-profile bot sees an empty registry.
        # The delivery runs in a background service context whose PATH lacks the gateway's
        # venv bin dir, so a bare "hermes" resolves to a system install and dies on import
        # under the wrong interpreter (#108628). _hermes_cli pins the entrypoint beside
        # this interpreter; _delivery_lock/_local_delivery_home match argv[0] by basename,
        # so the absolute path stays compatible.
        return _start_delivery([_hermes_cli(), "-p", _self_profile_name(root), "peer", "dm", dm_target], content,
                               f"@{peer_profile or peer_name} on peer '{peer_name}'", stdin_file=True,
                               author=peer_author, **delivery)

    # A connection-qualified target ('hermes@mini') names a relay row outright; it is the form the relay itself
    # hands out for a colliding row, and stamps on replies. Resolved locally first, a local bot whose friendly
    # name slugs to 'hermes-mini' captured it. An '@' name no connection answers to still resolves locally.
    if "@" in raw_target.strip().lstrip("@"):
        relayed = _try_relay_delivery(root, raw_target, content, me, **delivery)
        if relayed is not None:
            return relayed
    # Local teammate — folder id, or a friendly name / Desktop @-slug ('Scribe', 'Dr. Foo').
    resolved = _resolve_local_name(raw_target, roster, root)
    is_local_shape = bool(_LOCAL_TARGET_RE.match(raw_target))
    if resolved is None and not is_local_shape and "@" not in raw_target:
        return sender.roster_err(f"Invalid target: {raw_target!r}.")
    if resolved is None or resolved == me:
        # Unknown locally, or same-name target on ANOTHER connection (this gateway's 'default'
        # messaging the cloud 'default'): every Desktop-connected gateway is reachable via the
        # relay roster, so try that before reporting a resolution failure / self-message.
        relayed = _try_relay_delivery(Path(sender.home), raw_target, content, me, **delivery)
        if relayed is not None:
            return relayed
        if resolved == me:
            return _err("You can't message yourself. Pick a teammate from the roster.")
        return sender.roster_err(f"No teammate named '{raw_target}' on this install, on a connected "
                                 "machine, or on a registered peer. Pick a name from the roster "
                                 "(roles are listed in your system prompt).")
    if (getattr(sender.agent, "_turn_author", None) or {}).get("id") == f"bot:{resolved}":
        # One reply path (#95074): this turn answers @resolved's delivery, whose runner carries the
        # final response back. A second DM races it and lands in their chat as a separate turn.
        return _err(f"@{_handle(resolved)} messaged you this turn and your final response already goes back "
                    "to them: put your reply in it instead of messaging them. Do not retry.")
    # Local teammates are admitted through the profile authority; the argv only names the
    # profile for the runner's lock/home lookups and is never executed.
    return _start_delivery([_hermes_cli(), "-p", resolved], content, f"@{_handle(resolved)}",
                           stdin_file=False, profile_home=roster_homes[resolved], author=author, **delivery)


def _try_relay_delivery(root: Path, raw_target: str, content: str, me: str, *,
                        task_id: Optional[str], agent: Any, wake: Any = "notify") -> Optional[str]:
    """Cross-connection delivery via the Desktop relay; None when the target doesn't
    resolve against the relay roster. The envelope is queued on disk for the Desktop
    to drain; a background waiter is spawned immediately so the relayed reply wakes
    the sender through the standard completion-notification path."""
    try:
        from tools.bot_mode_probe import _handle, local_taken_forms
        from tools.bot_relay import (
            EnvelopeRefusedError, _target_aliases, enqueue_envelope, read_remote_roster, remote_target_forms,
            resolve_remote_target, waiter_command,
        )

        roster = read_remote_roster(root)
        match = resolve_remote_target(raw_target, roster) if roster else None
        if match is None:
            return None
        if match == "ambiguous":
            want = raw_target.strip().lstrip("@").partition("@")[0].lower()
            forms = ", ".join(form for r, form in zip(roster, remote_target_forms(roster, local_taken_forms(root)))
                              if want in _target_aliases(r))
            return _err(f"'{raw_target}' exists on several connected machines — disambiguate with one of: {forms}.")
        try:
            envelope = enqueue_envelope(root, target=match, message=content, sender_profile=me, sender_handle=_handle(me))
        except EnvelopeRefusedError as exc:
            # Fail fast: target definitively offline — nothing was queued.
            # Structured refusal so the agent can distinguish it from a resolution error ('runtime_offline'
            # per the #93091 reason enum).
            return json.dumps({"error": str(exc), "reason": exc.reason})
        label = f"@{match['handle']} on {match['connection_label'] or match['connection_id']}"
        if wake == "none":
            # The relay delivers the queued envelope on its own; only a reply needs the waiter.
            return json.dumps({"status": "queued", "delivery_id": envelope["id"], "to": label,
                               "expects_reply": False, "detail": _NOTIFY_ONLY_DETAIL})
        raw = _spawn_delivery(waiter_command(root, envelope), label, delivery_id=envelope["id"], task_id=task_id,
                              agent=agent, wake=wake)
        waiter_error = json.loads(raw).get("error")
        if not waiter_error:
            return raw
        # The envelope is already queued and the Desktop drains it on its own, so a waiter that
        # failed to start loses only the reply wake-up. Reporting a hard failure here makes the
        # sender resend and deliver the message twice. Same shape as the live-owner branch of
        # _start_delivery: queued + notification_error.
        return json.dumps({
            "status": "queued", "delivery_id": envelope["id"], "to": label, "notification_error": waiter_error,
            "detail": (f"Message queued for {label}; the relay delivers it on its own, but the reply "
                       "waiter did not start, so the reply will NOT wake you. Do NOT resend."),
        })
    except Exception:
        logger.debug("relay delivery attempt failed", exc_info=True)
        return None


def _dm_dir() -> Path:
    uid_getter = getattr(os, "getuid", None)
    uid = uid_getter() if callable(uid_getter) else None
    path = Path(tempfile.gettempdir()) / (f"{_DM_DIR_NAME}-{uid}" if uid is not None else _DM_DIR_NAME)
    path.mkdir(mode=0o700, exist_ok=True)
    # Shared POSIX temp roots need a per-user directory. Fail closed if an
    # attacker pre-created the expected path or replaced it with a symlink.
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise PermissionError(f"DM temp path is not a directory: {path}")
    if uid is not None and info.st_uid != uid:
        raise PermissionError(f"DM temp directory is owned by another user: {path}")
    if stat.S_IMODE(info.st_mode) != 0o700:
        path.chmod(0o700)
    return path


def cleanup_bot_dm_cache(max_age_hours: float = _DM_STALE_SECONDS / 3600, *, now: float | None = None) -> int:
    """Delete orphaned DM payload files older than *max_age_hours*; returns count.
    Same contract as the other ``cleanup_*_cache`` helpers (hourly gateway housekeeping);
    legacy temp-root locations from versions predating the dedicated directory are swept too."""
    cutoff = (time.time() if now is None else now) - max_age_hours * 3600
    temp_root = Path(tempfile.gettempdir())
    locations = [(temp_root, "hermes-dm-*.txt"), (temp_root, "hermes-relay-dm-*.txt")]
    with contextlib.suppress(OSError):
        dm_dir = _dm_dir()
        locations.append((dm_dir, "*.txt"))
        # Live-delivery intents (``<dm file>.live.json``, message plaintext included) outlive
        # their runner on purpose — a retry replays the same delivery id from them — so the
        # orphans of runners that never settled are swept here too.
        locations.append((dm_dir, "*.live.json"))
        locations.append((dm_dir, "fanout-*.json"))
    from tools.bot_relay import unlink_files_older_than

    return sum(unlink_files_older_than(d, pattern, cutoff) for d, pattern in locations)


def _unlink_dm_file(path: str) -> None:
    with contextlib.suppress(OSError):
        os.unlink(path)


def _write_dm_file(content: str) -> str:
    """The message rides a temp file — never inline shell text."""
    cleanup_bot_dm_cache()
    fd, path = tempfile.mkstemp(prefix="dm-", suffix=".txt", dir=_dm_dir(), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
    except BaseException:
        # If fdopen itself failed the raw descriptor is still ours; closing twice is harmless.
        with contextlib.suppress(OSError):
            os.close(fd)
        _unlink_dm_file(path)
        raise
    return path


def _delivery_lock(argv: list[str], *, stdin_file: bool):
    """Per-profile turn lock for a LOCAL teammate delivery: local and relay deliveries
    into one profile both run a Bot Chat turn here, so the turn window is serialized on
    ``tools.bot_relay``'s cross-process lock. Peer transports (stdin mode) are locked
    on the remote gateway by its own deliver path.

    See #93091.
    """
    # Match the CLI element by basename: argv[0] may be an absolute venv path
    # (service contexts lack PATH) and carries .exe on Windows; split on both separators.
    # Split on both separators so the shape matches regardless of which platform built the argv. See #93590.
    cli = (argv[0] if argv else "").rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    if stdin_file or len(argv) < 3 or cli not in ("hermes", "hermes.exe") or argv[1] != "-p":
        return contextlib.nullcontext()
    from tools.bot_mode_probe import _hermes_root
    from tools.bot_relay import acquire_turn_lock

    return acquire_turn_lock(_hermes_root(Path(_default_home())), argv[2])


def _live_intent_file(dm_file: str | os.PathLike) -> str:
    """The pinned live-delivery intent beside a DM file (the cache sweep globs ``*.live.json``)."""
    return f"{os.fspath(dm_file)}.live.json"


def _dm_delivery_id(dm_file: str | os.PathLike) -> str:
    """One delivery id per DM file: the dispatch ack, the live-owner intent and every retry
    of the runner derive it the same way, so the sender can correlate all of them."""
    return hashlib.sha256(str(Path(dm_file).resolve()).encode()).hexdigest()


class LiveTargetUnavailableError(RuntimeError):
    """The canonical Bot Chat owner could not be reached with anything admitted — before the
    intent was pinned, or on a dial that died before the deliver RPC went out: nothing was
    sent, so this is a definite no-send the sender may retry, not the ambiguous
    post-admission state."""

    reason = "runtime_unavailable"
    resend = "once the target is up"


class LiveTargetHeldError(LiveTargetUnavailableError):
    """The owner refused because another session writer holds the Bot Chat, and settled this
    delivery id as never sent (``bot_relay.settle``): every later attempt under the id gets that
    disposition, so this is a definite no-send too. A refusal without that settlement stays the
    ambiguous outcome."""

    resend = "after that session closes"

    @property
    def reason(self) -> str:
        from tools.bot_failure_reasons import HELD_NOT_SENT
        return HELD_NOT_SENT


def _unavailable_payload(dm_file: str, exc: LiveTargetUnavailableError) -> str:
    return json.dumps({"reason": exc.reason, "delivery_id": _dm_delivery_id(dm_file),
                       "error": f"{exc} — the message was NOT sent; it is safe to resend {exc.resend}."})


def _settled_no_send(dm_file: str, exc: LiveTargetUnavailableError) -> str:
    """The answer to a definite no-send, for both the tool and its runner: first the payload and
    any pinned intent go, so nothing local can rebuild or replay the delivery. A file that is still
    there is not settlement: the answer is then the unknown outcome, never "safe to resend"."""
    left = []
    for path in (_live_intent_file(dm_file), dm_file):
        _unlink_dm_file(path)
        if os.path.lexists(path):
            left.append(path)
    if left:
        return _live_outcome_unknown(dm_file, f"could not remove {left[0]}")
    return _unavailable_payload(dm_file, exc)


def _held_disposition(record: dict | None) -> bool:
    """The owner's terminal not-sent receipt for a delivery refused while another writer held the
    chat (``gateway/session_bot.py:_not_sent_record``)."""
    from tools.bot_failure_reasons import HELD_NOT_SENT
    return bool(record) and record.get("refused") is True and record.get("reason") == HELD_NOT_SENT


def _admit_live_dm(profile_home: Path | None, dm_file: str, author: Optional[dict] = None) -> dict | None:
    """Pin intent before admission; retries may inspect, never change transport."""
    from tools.bot_live_delivery import (PreDialError, deliver_to_live_owner, find_canonical_live_owner,
                                         read_delivery_result)
    from utils import fsync_directory

    intent: dict[str, Any]
    intent_path = Path(_live_intent_file(dm_file))
    pinned_now = False
    if intent_path.exists():
        intent = json.loads(intent_path.read_text(encoding="utf-8-sig"))
    else:
        assert profile_home is not None
        try:
            owner = find_canonical_live_owner(profile_home)
        except Exception as exc:
            # Discovery runs before the intent exists and never admits: definite, not ambiguous.
            raise LiveTargetUnavailableError(f"canonical Bot Chat target is unavailable ({exc})") from exc
        if owner is None:
            raise LiveTargetUnavailableError("canonical Bot Chat target is unavailable")
        intent = dict(owner=owner, message=Path(dm_file).read_text(encoding="utf-8-sig"),
                      delivery_id=_dm_delivery_id(dm_file),
                      **({"author": author} if author else {}))
        try:
            fd = os.open(intent_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            intent = json.loads(intent_path.read_text(encoding="utf-8-sig"))
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(intent, stream)
                stream.flush()
                os.fsync(stream.fileno())
            fsync_directory(intent_path.parent)
            pinned_now = True
    home = intent["owner"]["profile_home"]
    record = read_delivery_result(home, intent["delivery_id"])
    if record is None or intent["owner"].get("canonical"):
        from hermes_cli.gateway_client import GatewayRPCError
        from tools.bot_failure_reasons import HELD_NOT_SENT
        from tools.bot_live_delivery import settle_held_delivery

        identity = dict(delivery_id=intent["delivery_id"], author=intent.get("author"))
        try:
            record = deliver_to_live_owner(home, intent["owner"], intent["message"], **identity)
        except GatewayRPCError as exc:
            if str(exc) != HELD_NOT_SENT or exc.data.get("admitted") is not False:
                raise
            # Nothing was admitted while that refusal held the owner's lock, but a same-id attempt
            # may have admitted since: only the owner's settle decides, under that same lock.
            record = settle_held_delivery(home, intent["owner"], intent["message"], **identity)
        except PreDialError as exc:
            if not pinned_now:
                raise  # an earlier attempt under this intent may have admitted: outcome unknown
            # This call pinned the intent and its first dial died before the RPC went out: nothing
            # was admitted anywhere, so it is a definite no-send the sender may retry (#139303).
            _unlink_dm_file(_live_intent_file(dm_file))
            raise LiveTargetUnavailableError(f"live authority dial failed before admission ({exc})") from exc
    if _held_disposition(record):
        # The callers remove the intent and payload (``_settled_no_send``).
        raise LiveTargetHeldError("Bot Chat is held by another session writer")
    return record


def _wait_live_dm(home: str, delivery_id: str, *, dm_file: str | os.PathLike | None = None) -> int:
    from tools.bot_failure_reasons import RUNTIME_OFFLINE
    from tools.bot_live_delivery import await_delivery, cancel_queued_delivery, owner_holds_delivery

    deadline = time.monotonic() + _LIVE_WAIT_MAX_SECONDS
    missed = 0
    while True:
        record = await_delivery(home, delivery_id, max(0.0, min(_LIVE_WAIT_SECONDS, deadline - time.monotonic())))
        if record is None or record["status"] not in ("queued", "claimed") or time.monotonic() >= deadline:
            break
        try:
            held = owner_holds_delivery(home, record)
        except Exception:
            logger.debug("Could not check live DM owner", exc_info=True)
            held = None
        if held:
            missed = 0
            continue
        if held is False and record["status"] == "queued":
            # Under the same mailbox lock as claim, either cancellation wins or
            # the claim wins and we must keep waiting for its outcome.
            record = cancel_queued_delivery(
                home, delivery_id, error="Live Bot Chat owner is no longer available",
                reason=RUNTIME_OFFLINE)
            if record is None or record["status"] != "claimed":
                break
            missed = 0
            continue
        # A vanished consumer may leave a permanent claim, and an owner we cannot look up
        # may be gone: allow one more window for an in-flight turn, then release the runner.
        missed += 1
        if missed > 1:
            break
    status = record["status"] if record else "ambiguous"
    payload = {key: record[key] for key in ("reply", "error", "reason") if record and record.get(key)}
    payload.update(status=status, delivery_id=delivery_id)
    if status in ("queued", "claimed", "ambiguous"):
        payload["detail"] = "Delivery remains pending or its outcome is unknown. Do not resend; receipt is retained."
        state = _live_target_state(dm_file)
        if state:
            # A bound that lapses on a teammate parked on its own user is that, never a silent timeout.
            payload["target_state"] = state
    elif status == "settled" and dm_file is not None:
        # The intent carries the message plaintext so a retry can replay the SAME delivery id;
        # once the owner settled it nothing retries, so it goes along with the dm file (same
        # plaintext) — the live branch returns before _run_delivery's own unlink.
        _unlink_dm_file(_live_intent_file(dm_file))
        _unlink_dm_file(str(dm_file))
    print(json.dumps(payload))
    return 0 if status in ("settled", "queued", "claimed") else 1


def _live_target_state(dm_file: str | os.PathLike | None) -> Optional[str]:
    """The target Bot Chat's live state for a wait ending pending, from a same-id replay of the
    pinned intent (the owner answers it with the existing receipt). None when nothing answers:
    the state is a hint, and its absence must never change the delivery outcome."""
    if dm_file is None or not os.path.exists(_live_intent_file(dm_file)):
        return None
    try:
        return (_admit_live_dm(None, os.fspath(dm_file)) or {}).get("target_state")
    except Exception:
        logger.debug("Could not read the live DM target's state", exc_info=True)
        return None


def _local_delivery_home(argv: list[str]) -> Path | None:
    cli = (argv[0] if argv else "").rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    if len(argv) < 3 or cli not in ("hermes", "hermes.exe") or argv[1] != "-p":
        return None
    from tools.bot_mode_probe import _hermes_root, _roster

    return dict(_roster(_hermes_root(Path(_default_home())))).get(argv[2])


def _live_outcome_unknown(dm_file: str, cause: object) -> str:
    """The runner's stdout when a live admission may have happened but its outcome is unknown."""
    return json.dumps({"status": "ambiguous", "delivery_id": _dm_delivery_id(dm_file),
                       "error": f"Live admission outcome unknown: {cause}. Do not resend.",
                       "evidence_file": dm_file})


def _runner_argv(args: list[str]) -> tuple[Optional[str], str, str, list[str]] | None:
    """Split ``--run-delivery [--author <json>] <mode> <dm_file> <argv…>`` into
    ``(author_json, mode, dm_file, argv)``; None when malformed. Stdlib only: the boot-failure
    report runs it when no Hermes import is available."""
    if args[:1] != ["--run-delivery"]:
        return None
    rest, author = args[1:], None
    if rest[:1] == ["--author"]:
        if len(rest) < 2:
            return None
        author, rest = rest[1], rest[2:]
    if len(rest) < 2 or rest[0] not in ("stdin", "query-file"):
        return None
    return author, rest[0], rest[1], rest[2:]


def _run_delivery(argv: list[str], dm_file: str, *, stdin_file: bool,
                  profile_home: Path | None = None, author: Optional[dict] = None) -> int:
    """Admit local DMs only through the profile authority; retain uncertain intent.

    The optional peer stdin transport remains an explicit remote route, never a
    fallback after local admission. Background processes wait for delivery only.
    """
    # The live consumer owns turn admission; never compete for its CLI lease.
    if not stdin_file:
        home = profile_home or _local_delivery_home(argv)
        if home is not None or os.path.exists(_live_intent_file(dm_file)):
            try:
                record = _admit_live_dm(home, dm_file, author)
            except LiveTargetHeldError as exc:
                print(_settled_no_send(dm_file, exc))  # the owner settled the id: drop the payload too
                return 1
            except LiveTargetUnavailableError as exc:
                print(_unavailable_payload(dm_file, exc))
                return 1
            except Exception as exc:
                print(_live_outcome_unknown(dm_file, exc))
                return 1
            if record is not None:
                return _wait_live_dm(record["profile_home"], record["delivery_id"], dm_file=dm_file)
    if not stdin_file:
        print(json.dumps({"reason": "runtime_unavailable", "error": "No canonical Bot Chat authority; payload retained."}))
        return 1
    try:
        from tools.bot_relay import delivery_env

        env = delivery_env(author, profile_home if not stdin_file else None)
        with _delivery_lock(argv, stdin_file=stdin_file):
            # Keep the file open until the transport exits; cleanup occurs
            # after subprocess.run returns, not merely after stdin reaches EOF.
            with open(dm_file, "r", encoding="utf-8-sig") as stream:
                # Passing the file descriptor as stdin bypasses the BOM-aware decoder.
                return subprocess.run(argv, input=stream.read().encode("utf-8"), check=False, env=env).returncode
    finally:
        _unlink_dm_file(dm_file)


def _delivery_command(argv: list[str], dm_file: str, *, stdin_file: bool,
                      profile_home: Path | None = None, author: Optional[dict] = None) -> str:
    """Build an argv-safe command for the cleanup-owning background runner:
    ``--run-delivery [--author <json>] <mode> <dm_file> [--profile-home <path>] <argv...>``."""
    runner_argv = [sys.executable, str(Path(__file__).resolve()), "--run-delivery",
                   "stdin" if stdin_file else "query-file", dm_file]
    if profile_home is not None:
        runner_argv.extend(["--profile-home", str(Path(profile_home).resolve())])
    runner_argv.extend(argv)
    if sys.platform == "win32":
        # The tracked local backend uses Git Bash on native Windows: forward slashes keep drive
        # paths executable there; backslash paths are parsed as command names (exit 127).
        runner_argv = [part.replace("\\", "/") for part in runner_argv]
    if author:
        # Inserted after the slash rewrite: JSON escapes are backslashes too.
        runner_argv[3:3] = ["--author", json.dumps(author, separators=(",", ":"))]
    return shlex.join(runner_argv)


def _start_delivery(argv: list[str], content: str, label: str, *, stdin_file: bool,
                    task_id: Optional[str], agent: Any, profile_home: Path | None = None,
                    author: Optional[dict] = None, wake: Any = "notify") -> str:
    """Create a DM file and transfer its cleanup ownership to the runner."""
    dm_file = _write_dm_file(content)
    if profile_home is not None:
        try:
            record = _admit_live_dm(profile_home, dm_file, author)
        except LiveTargetUnavailableError as exc:
            return _settled_no_send(dm_file, exc)
        except Exception as exc:
            return json.dumps({"status": "ambiguous", "delivery_id": _dm_delivery_id(dm_file),
                "error": f"Live delivery admission could not be confirmed: {exc}. Do not resend.",
                "evidence_file": dm_file})
        if record is not None:
            if wake == "none":
                # The admission is durable at the owner and nothing waits for its reply: the intent and
                # payload exist only for a waiting runner's same-id replays.
                _unlink_dm_file(_live_intent_file(dm_file))
                _unlink_dm_file(dm_file)
                return json.dumps(_with_target_state(dict(
                    status=record["status"], delivery_id=record["delivery_id"], to=label, expects_reply=False,
                    detail=_NOTIFY_ONLY_DETAIL), record, label))
            command = _delivery_command(argv, dm_file, stdin_file=False, profile_home=profile_home, author=author)
            notification = json.loads(_spawn_delivery(command, label, delivery_id=record["delivery_id"],
                                                      task_id=task_id, agent=agent, wake=wake))
            result = dict(status=record["status"], delivery_id=record["delivery_id"], to=label,
                          detail="Durably queued for the live Bot Chat owner. Do NOT wait or resend; finish your turn.")
            if notification.get("error"):
                result["notification_error"] = notification["error"]
            elif notification.get("process_id"):
                result["process_id"] = notification["process_id"]
                result["reply_delivery"] = notification.get("reply_delivery", "notification")
                if result["reply_delivery"] == "poll":
                    # Same runner, same stdout-borne reply (#101142): a non-push sender must get
                    # the poll instruction here too, not 'finish your turn'.
                    result["detail"] = f"Durably queued for the live Bot Chat owner. Do NOT resend. {notification['detail']}"
            return json.dumps(_with_target_state(result, record, label))
    try:
        command = _delivery_command(argv, dm_file, stdin_file=stdin_file, profile_home=profile_home, author=author)
    except BaseException:
        _unlink_dm_file(dm_file)
        raise
    return _spawn_delivery(command, label, dm_file=dm_file, task_id=task_id, agent=agent, wake=wake)


_NOTIFY_ONLY_DETAIL = "Sent as notify-only: no reply comes back and nothing will wake you. Do not resend."
# What a sender should tell its user when the teammate's turn is parked on that teammate's own user.
_BLOCKED_DETAIL = {
    "waiting_approval": "is waiting for its user to approve a command in its own Bot Chat",
    "waiting_clarify": "is waiting for its user to answer a question in its own Bot Chat",
    "paused": "is not starting new turns right now (its gateway is restarting or updating)",
}


def _with_target_state(result: dict, record: dict, label: str) -> dict:
    """Carry the owner's live ``target_state`` into an ack; a blocked teammate is said so plainly."""
    state = record.get("target_state")
    if state:
        result["target_state"] = state
    if state in _BLOCKED_DETAIL:
        result["detail"] += (f" {label} {_BLOCKED_DETAIL[state]}, so its reply comes only after that: "
                             "tell your user now.")
    return result


def _spawn_delivery(command: str, label: str, *, dm_file: Optional[str] = None, delivery_id: Optional[str] = None,
                    task_id: Optional[str], agent: Any, wake: Any = "notify",
                    completion_chars: int = REPLY_COMPLETION_CHARS) -> str:
    """Launch the cleanup-owning runner and transfer file ownership on ack. ``dm_file``
    is None for relay deliveries (the waiter watches a reply file; envelope artifacts
    are owned/swept by ``tools/bot_relay.py``), which pass the envelope id as ``delivery_id``
    instead. The ack is ``queued`` + ``delivery_id`` like the live-owner branch: hand-off
    to a background process, never a delivery receipt. A list ``wake`` collects the runner for
    one fan-out process to join (it owns ``dm_file`` from here); ``"none"`` runs it unwatched."""
    if isinstance(wake, list):
        delivery_id = delivery_id or (_dm_delivery_id(dm_file) if dm_file else "")
        wake.append(dict(command=command, label=label, dm_file=dm_file, delivery_id=delivery_id))
        return json.dumps({"status": "queued", "delivery_id": delivery_id, "to": label})
    transferred = False
    try:
        from tools.terminal_tool import terminal_tool

        raw = terminal_tool(command, background=True, notify_on_complete=wake != "none", task_id=task_id,
                            workdir=str(Path(__file__).resolve().parent.parent), _host_local=True,
                            _completion_output_chars=completion_chars)
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            parsed = {}
        proc_id = parsed.get("session_id") or ""
        if parsed.get("error"):
            return _err(f"Delivery to {label} failed to start: {parsed['error']}")
        if parsed.get("status") == "pending_approval":
            # terminal_tool's approval gate answers with an EMPTY error and no session_id: the runner
            # never launched because nobody in this turn could approve its command.
            return _err(f"Delivery to {label} failed to start: its command needs terminal approval that nobody "
                        "in this turn can grant" + (", so nothing was sent. Approve it (or add it to "
                                                    "command_allowlist) and send again." if dm_file else "."))
        if not proc_id:
            return _err(f"Delivery to {label} failed to start: no process id returned")
        # From here the background runner owns the file (removed after the consumer finishes).
        transferred = True
        if wake == "none":
            return json.dumps({"status": "queued", "delivery_id": delivery_id or (_dm_delivery_id(dm_file) if dm_file
                               else ""), "to": label, "expects_reply": False, "detail": _NOTIFY_ONLY_DETAIL})
        if parsed.get("notify_on_complete") is False:
            # terminal_tool refused the completion promise: this session (api_server, one-shot
            # runner) cannot receive an async completion, so the recipient's reply would never
            # be injected here (#101142). Say so and name the return path the surface supports.
            detail = (f"Message handed to a background delivery process for {label}, but THIS session "
                      "cannot receive completion notifications, so the reply will NOT arrive on its own. "
                      f"Before ending your turn, retrieve the outcome with process(action='wait', "
                      f"session_id='{proc_id}') — its output is the reply (relay it, attributed to that "
                      "agent) or the delivery failure (report it; the message was NOT delivered); "
                      "if wait returns status=timeout, call wait again until the process exits.")
            if _persist_reply_when_done(proc_id, agent):
                detail += (" Its outcome is also saved into this session's transcript as a delivery row "
                           "when the process exits, so it survives even if the turn ends first.")
        else:
            detail = (f"Message queued for {label}: this acknowledges the hand-off to a "
                      "background delivery process, not a delivery receipt — do NOT wait or poll. "
                      "Finish your turn now; that process's completion notification carries the "
                      "delivery outcome — the reply (relay it then, attributed to that agent) or "
                      "the delivery failure (report it; the message was NOT delivered).")
        return json.dumps({
            "status": "queued",
            "delivery_id": delivery_id or (_dm_delivery_id(dm_file) if dm_file else ""),
            "to": label,
            "reply_delivery": "poll" if parsed.get("notify_on_complete") is False else "notification",
            "detail": detail,
            "process_id": proc_id,
            "queued_at": int(time.time()),
        })
    except Exception as exc:
        logger.error("message_agent delivery spawn failed: %s", exc, exc_info=True)
        return _err(f"Delivery to {label} could not be started: {exc}")
    finally:
        if dm_file and not transferred:
            _unlink_dm_file(dm_file)


def _persist_reply_when_done(proc_id: str, agent: Any) -> bool:
    """#101142 durable leg. A non-push sender (api_server, one-shot runner) gets no completion
    notification, so once the tracked runner exits its stdout — the recipient's reply or the
    delivery failure — is appended to the sender's session transcript as a DELIVERY row
    (``display_kind="process_complete"``, the shape push surfaces persist for the same
    completion; mirrors gateway.wake.persist_delegation_delivery). A sender that already read
    the outcome via process(action='wait'/'log') is not told twice. Returns False (nothing
    armed) when the sender has no session transcript or the process is not tracked here."""
    from tools.process_registry import process_registry

    db, session_id = getattr(agent, "_session_db", None), getattr(agent, "session_id", None)
    proc = process_registry.get(proc_id)
    if proc is None or not session_id or not callable(getattr(db, "append_message", None)):
        return False

    def _run() -> None:
        from tools.process_registry_notifications import (
            format_process_notification, process_completion_display_text,
        )

        proc._completion_event.wait()
        if process_registry.is_completion_consumed(proc_id):
            return
        evt = {"type": "completion", "session_id": proc_id, **process_registry._exit_snapshot(proc, "exited")}
        try:
            db.append_message(session_id, "user", content=format_process_notification(evt),
                              display_kind="process_complete",
                              display_metadata={"display_text": process_completion_display_text([evt])})
        except Exception as exc:
            logger.warning("message_agent: could not persist the reply of %s into session %s: %s",
                           proc_id, session_id, exc)

    threading.Thread(target=_run, name=f"message-agent-reply-{proc_id}", daemon=True).start()
    return True


def _wait_reply_main(reply_path: str, label: str, budget_seconds: str) -> int:
    """The relay reply waiter (``tools/bot_relay.waiter_command``): block until the sender-side
    reply file exists, print it as the completion notification the sender wakes on, exit 1 on a
    delivery error or when the budget runs out. Stdlib only: this runs as a background process
    from any bot turn, and the sender's completion notification is exactly its stdout."""
    try:
        deadline = time.time() + float(budget_seconds)
    except ValueError:
        return 2
    while time.time() < deadline:
        if os.path.exists(reply_path):
            with open(reply_path, encoding="utf-8-sig") as fh:
                d = json.load(fh)
            if d.get("error"):
                # Typed reason code rides ahead of the free text so the sender can branch on it
                # without parsing provider prose. See #93091.
                code = str(d.get("reason") or "").strip()
                tag = f" [reason: {code}]" if code else ""
                print(f"Delivery to {label} failed{tag}: {d['error']}")
                return 1
            print(f"Reply from {label}:")
            print(d.get("reply") or "(empty reply)")
            return 0
        # 250ms cadence: stat is cheap and a longer sleep is pure dead air.
        time.sleep(0.25)
    print(f"No reply from {label} within {budget_seconds}s. The message may still be delivered when "
          "the Desktop reconnects; do not resend blindly.")
    return 1


def _delivery_main(args: list[str]) -> int:
    """Runner entry for the argv ``_delivery_command`` and ``bot_relay.waiter_command`` build.
    Malformed argv exits 2 without touching the DM file."""
    if args[:1] == ["--wait-reply"]:
        return _wait_reply_main(*args[1:]) if len(args) == 4 else 2
    if args[:1] == ["--fan-out"]:
        from tools.bot_mode_dm_coordination import run_fan_out

        return run_fan_out(args[1]) if len(args) == 2 else 2
    parsed = _runner_argv(args)
    if parsed is None:
        return 2
    author_json, mode, dm_file, argv = parsed
    author = None
    if author_json is not None:
        from agent.turn_author import parse_turn_author

        author = parse_turn_author(author_json)
        if author is None:
            return 2
    try:
        profile_home = None
        if len(argv) >= 2 and argv[0] == "--profile-home":
            profile_home, argv = Path(argv[1]), argv[2:]
        return _run_delivery(argv, dm_file, stdin_file=mode == "stdin", profile_home=profile_home, author=author)
    except Exception as exc:
        # Every refusal ships a typed reason on stdout so the completion notification carries it
        # back to the sender (#93091): 'target_busy' from the queue's bounded wait, otherwise the
        # same vocabulary-guarded classification the relay lane applies.
        from tools.bot_failure_reasons import delivery_failure_reason

        print(json.dumps({"error": str(exc), "reason": delivery_failure_reason(exc)}))
        return 1


# agent-context helpers (mirror system_prompt.py's resolution)


def _agent_home(agent: Any) -> str:
    """The calling agent's OWN home (session-db derived), not ambient env."""
    with contextlib.suppress(Exception):
        db_path = getattr(getattr(agent, "_session_db", None), "db_path", None)
        if db_path:
            return str(Path(db_path).parent)
    return _default_home()


def _session_title(agent: Any) -> str:
    title = str(getattr(agent, "_session_title_hint", "") or "").strip()
    if title:
        return title
    with contextlib.suppress(Exception):
        sdb, sid = getattr(agent, "_session_db", None), getattr(agent, "session_id", None)
        if sdb and sid:
            return str(sdb.get_session_title(sid) or "").strip()
    return ""


if __name__ == "__main__":  # pragma: no cover - exercised as a background process
    # Spawned as a script with the sender's sys.executable, which under PM is the bare store
    # interpreter (dependencies are activated in-process, never inherited), so boot like every
    # entry point before the lazy Hermes imports. Run as a path, sys.path[0] is tools/.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    # Only the delivery lane boots: the reply waiter is stdlib only and its stdout is the
    # sender's wake-up, and any other argv exits 2 before a Hermes import.
    if sys.argv[1:2] == ["--run-delivery"]:
        try:
            import hermes_bootstrap
        except (Exception, SystemExit) as exc:
            # A pinned live intent means the sender may already have admitted this DM and was
            # told not to resend; a bare repair hint would read as "NOT delivered". Without an
            # intent nothing was handed over, so the plain failure is the truth. A relaunched
            # child (Windows) already printed its own outcome.
            booted = getattr(exc, "relaunched", False) or (isinstance(exc, SystemExit) and not exc.code)
            parsed = _runner_argv(sys.argv[1:])
            if not booted and parsed and os.path.exists(_live_intent_file(parsed[2])):
                print(_live_outcome_unknown(parsed[2], "the delivery runner could not activate "
                                                       "Hermes dependencies (see stderr)"))
            raise
    raise SystemExit(_delivery_main(sys.argv[1:]))
