"""Bot Mode agent-to-agent DM tool — ``message_agent``.

A structured, Bot-Chat-only tool that lets a Bot Mode agent message a
teammate agent (another Hermes profile on this install, or an agent on a
registered peer gateway) WITHOUT hand-assembling shell commands.

Why this exists (Aug 2026): the Bot Mode teammate protocol taught agents to
DM each other via a prompt-injected ``hermes -p <bot> chat ...`` shellout.
That transport works, but the *invocation* was fragile — quoting traps
(#91339/#91304), temp-file choreography, dead-profile races — and the
Desktop's remote-mention path forwarded raw user text verbatim (#91397).
``message_agent`` replaces the invocation with a real tool call: the message
is a parameter, the target is validated against the live roster, the
attribution prefix is applied server-side, and the reply arrives through the
existing background-process notification path (fire-and-forget, never
blocks the sender's turn).

Containment contract (MUST hold — reviewers check all three):
- The tool schema is injected ONLY into a bot's canonical "Bot Chat"
  session on Bot-Mode-managed installs — the exact same gate as the
  protocol section in ``tools/bot_mode_probe.py``. It is NOT registered in
  the global tool registry, is NOT part of any toolset, and never appears
  in CLI sessions, ordinary gateway chats, group-room member sessions
  (titled "Group: …"), cron agents, or subagents.
- Dispatch is title-gated again at execution time (defense in depth): a
  forged call from a session that shouldn't have the tool returns a
  structured error instead of delivering.
- Everything here is additive. The legacy protocol transports
  (``hermes -p`` / ``hermes peer dm``) keep working for older prompts.

The transports themselves are unchanged and proven:
- local teammate  → ``hermes -p <name> chat --in ~ -c "Bot Chat"
  --create-if-missing -Q --query-file <tmp>`` (one turn, reply on stdout)
- peer teammate   → ``hermes peer dm <peer>[/<name>] < <tmp>``

Both run through ``terminal_tool(background=True, notify_on_complete=True)``
so the reply lands as a completion notification on the sender's NEXT turn —
the same wake shape every Bot Mode agent already knows.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

MESSAGE_AGENT_TOOL_NAME = "message_agent"

# Message body cap — generous for real work products, small enough that a
# runaway paste can't turn one DM into a context bomb on the recipient.
MESSAGE_MAX_CHARS = 16000

# A runner normally owns and removes each file. This bounds the residual
# plaintext lifetime if the machine dies after background-spawn acknowledgement
# but before the runner reaches its ``finally`` block.
_DM_DIR_NAME = "hermes-dm"
_DM_STALE_SECONDS = 24 * 60 * 60

_PEER_TARGET_RE = re.compile(r"^([a-z0-9][a-z0-9_-]{0,63})/([a-zA-Z0-9][a-zA-Z0-9_-]{0,63})$")
_LOCAL_TARGET_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


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
                "delivery acknowledgement. It does NOT return their reply and you must "
                "not wait or poll for one — send it, finish your turn, and the reply "
                "arrives later as a background-process completion notification that "
                "wakes you. COMPOSE the message yourself: write what YOU want to say to "
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
                            "'<peer>' / '<peer>/<agent>' for a registered peer gateway."
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
                },
                "required": ["target", "message"],
            },
        },
    }


def ensure_message_agent_tool(agent: Any) -> bool:
    """Inject the ``message_agent`` schema into a Bot Chat agent's tool list.

    Called once per turn from the conversation loop. Idempotent and
    deterministic for the life of a session: the gate (canonical Bot Chat
    title on a Bot-Mode-managed install) is stable from the session's first
    turn, so the tool list is byte-identical across turns — prompt-cache
    safe. Every non-Bot-Chat session fails the gate on every turn and never
    sees the schema. Never raises.
    """
    try:
        if not getattr(agent, "_bot_mode_protocol", True):
            return False
        tools = getattr(agent, "tools", None)
        if tools:
            for tool in tools:
                if (
                    isinstance(tool, dict)
                    and tool.get("function", {}).get("name") == MESSAGE_AGENT_TOOL_NAME
                ):
                    return True
        from tools.bot_mode_probe import BOT_CHAT_TITLE, is_bot_mode_managed

        if _session_title(agent) != BOT_CHAT_TITLE:
            return False
        # Managed-install check, NOT section non-emptiness: a profile whose
        # SOUL.md carries the legacy plugin-appended protocol text gets an
        # empty section (dedupe) but must still receive the tool — otherwise
        # upgraded installs silently lose A2A messaging (Aug 2026).
        if not is_bot_mode_managed(_agent_home(agent)):
            return False
        if agent.tools is None:
            agent.tools = []
        agent.tools.append(message_agent_tool_schema())
        valid = getattr(agent, "valid_tool_names", None)
        if isinstance(valid, set):
            valid.add(MESSAGE_AGENT_TOOL_NAME)
        return True
    except Exception:  # pragma: no cover — must never break a turn
        logger.debug("ensure_message_agent_tool failed", exc_info=True)
        return False


# ── roster resolution ────────────────────────────────────────────────────────


def _hermes_root(home: Path) -> Path:
    if home.parent.name == "profiles":
        return home.parent.parent
    return home


def _self_profile_name(home: Path) -> str:
    if home.parent.name == "profiles":
        return home.name
    return "default"


def _local_roster(root: Path) -> list[str]:
    """Profile names on this install: default + every named profile."""
    names = ["default"]
    try:
        profiles = root / "profiles"
        if profiles.is_dir():
            for child in sorted(profiles.iterdir()):
                if child.is_dir():
                    names.append(child.name)
    except Exception:
        pass
    return names


def _peers(root: Path) -> list[str]:
    try:
        from tools.bot_mode_probe import _peers as _probe_peers

        return _probe_peers(root)
    except Exception:
        return []


def _handle(name: str) -> str:
    return "hermes" if name == "default" else name


def _resolve_local_name(target: str, roster: list[str]) -> Optional[str]:
    """Map a target handle to a profile name ('hermes' → 'default')."""
    want = target.strip()
    if not want:
        return None
    if want.lower() == "hermes":
        return "default" if "default" in roster else None
    for name in roster:
        if name.lower() == want.lower():
            return name
    return None


# ── the tool ─────────────────────────────────────────────────────────────────


def _err(message: str, *, roster: list[str] | None = None, peers: list[str] | None = None) -> str:
    from tools.bot_failure_reasons import classify_agent_error

    payload: dict[str, Any] = {"error": message, "reason": classify_agent_error(message)}
    if roster is not None:
        payload["teammates"] = roster
    if peers is not None:
        payload["peers"] = peers
    return json.dumps(payload)


def message_agent_tool(
    target: str = "",
    message: str = "",
    task_id: Optional[str] = None,
    agent: Any = None,
) -> str:
    """Deliver ``message`` to ``target``'s Bot Chat. Returns a JSON ack/error.

    ``agent`` is the calling AIAgent (threaded by the executor) — used for
    the Bot Chat gate, the sender identity, and the session key so the
    spawned transport is tracked against the right session.
    """
    # ── defense-in-depth gate: only a canonical Bot Chat may deliver ──
    home = _agent_home(agent)
    try:
        from tools.bot_mode_probe import BOT_CHAT_TITLE, is_bot_mode_managed

        title = _session_title(agent)
        if title != BOT_CHAT_TITLE:
            return _err(
                "message_agent is only available in a Bot Mode 'Bot Chat' session. "
                "This session is not one; do not retry."
            )
        if not is_bot_mode_managed(home):
            return _err(
                "This install is not Bot-Mode-managed (no bot roster); "
                "message_agent is unavailable. Do not retry."
            )
    except Exception as exc:  # pragma: no cover — defensive
        return _err(f"Bot Mode gate check failed: {exc}")

    root = _hermes_root(Path(home))
    me = _self_profile_name(Path(home))
    roster = _local_roster(root)
    peers = _peers(root)
    teammates = [_handle(n) for n in roster if n != me]

    body = str(message or "").strip()
    if not body:
        return _err("message is required — compose what you want to say to that agent.")
    if len(body) > MESSAGE_MAX_CHARS:
        return _err(
            f"message too long ({len(body)} chars > {MESSAGE_MAX_CHARS}). "
            "Send the essentials; share large content as a file path instead."
        )

    raw_target = str(target or "").strip().lstrip("@")
    if not raw_target:
        return _err("target is required.", roster=teammates, peers=peers)

    sender_handle = _handle(me)
    prefix = f"Message from 🤖 {sender_handle} (@{sender_handle}): "

    # ── peer target: '<peer>/<agent>' or a bare registered peer name ──
    peer_match = _PEER_TARGET_RE.match(raw_target)
    bare_peer = raw_target.lower() if raw_target.lower() in peers else None
    if peer_match or bare_peer:
        peer_name = peer_match.group(1) if peer_match else bare_peer
        peer_profile = peer_match.group(2) if peer_match else None
        if peer_name not in peers:
            return _err(
                f"No registered peer named '{peer_name}'.", roster=teammates, peers=peers
            )
        dm_target = f"{peer_name}/{peer_profile}" if peer_profile else peer_name
        label = f"@{peer_profile or peer_name} on peer '{peer_name}'"
        return _start_delivery(
            ["hermes", "peer", "dm", dm_target],
            prefix + body,
            label,
            stdin_file=True,
            task_id=task_id,
            agent=agent,
        )

    # ── local teammate ──
    if not _LOCAL_TARGET_RE.match(raw_target) and "@" not in raw_target:
        return _err(f"Invalid target: {raw_target!r}.", roster=teammates, peers=peers)
    resolved = _resolve_local_name(raw_target, roster) if _LOCAL_TARGET_RE.match(raw_target) else None
    if resolved is None:
        # ── cross-connection teammate (Desktop relay) ──
        # Every gateway connected to the user's Desktop is reachable: the
        # relay roster lists agents on the other connections; delivery rides
        # the Desktop's own persistent socket to that gateway.
        relayed = _try_relay_delivery(
            root, raw_target, body, me, sender_handle, task_id=task_id, agent=agent
        )
        if relayed is not None:
            return relayed
        return _err(
            f"No teammate named '{raw_target}' on this install, on a connected "
            "machine, or on a registered peer. Pick a name from the roster "
            "(roles are listed in your system prompt).",
            roster=teammates,
            peers=peers,
        )
    if resolved == me:
        # Same-name target on ANOTHER connection (e.g. this gateway's
        # 'default' messaging the cloud 'default') — try the relay before
        # calling it a self-message.
        relayed = _try_relay_delivery(
            root, raw_target, body, me, sender_handle, task_id=task_id, agent=agent
        )
        if relayed is not None:
            return relayed
        return _err("You can't message yourself. Pick a teammate from the roster.")

    return _start_delivery(
        [
            "hermes",
            "-p",
            resolved,
            "chat",
            "--in",
            "~",
            "-c",
            "Bot Chat",
            "--create-if-missing",
            "-Q",
        ],
        prefix + body,
        f"@{_handle(resolved)}",
        stdin_file=False,
        task_id=task_id,
        agent=agent,
    )


def _try_relay_delivery(
    root: Path,
    raw_target: str,
    body: str,
    me: str,
    sender_handle: str,
    *,
    task_id: Optional[str],
    agent: Any,
) -> Optional[str]:
    """Cross-connection delivery via the Desktop relay, or None if the
    target doesn't resolve against the relay roster.

    The envelope is queued on disk; the Desktop drains it over RPC and
    delivers on the target connection's own socket. A background waiter is
    spawned immediately so the relayed reply wakes the sender through the
    standard completion-notification path — identical UX to a local DM.
    """
    try:
        from tools.bot_relay import (
            EnvelopeRefusedError,
            enqueue_envelope,
            read_remote_roster,
            resolve_remote_target,
            waiter_command,
        )

        roster = read_remote_roster(root)
        if not roster:
            return None
        match = resolve_remote_target(raw_target, roster)
        if match is None:
            return None
        if match == "ambiguous":
            forms = ", ".join(
                f"{r['handle']}@{r['connection_id']}"
                for r in roster
                if r["handle"].lower() == raw_target.strip().lstrip("@").lower()
            )
            return _err(
                f"'{raw_target}' exists on several connected machines — "
                f"disambiguate with one of: {forms}."
            )
        try:
            envelope = enqueue_envelope(
                root,
                target=match,
                message=f"Message from 🤖 {sender_handle} (@{sender_handle}): {body}",
                sender_profile=me,
                sender_handle=sender_handle,
            )
        except EnvelopeRefusedError as exc:
            # Fail fast: target definitively offline — nothing was queued.
            # Structured refusal so the agent can distinguish it from a
            # resolution error ('runtime_offline' per the #93091 reason enum).
            return json.dumps({"error": str(exc), "reason": exc.reason})
        label = f"@{match['handle']} on {match['connection_label'] or match['connection_id']}"
        return _spawn_delivery(
            waiter_command(root, envelope), label, task_id=task_id, agent=agent
        )
    except Exception:
        logger.debug("relay delivery attempt failed", exc_info=True)
        return None


def _dm_dir() -> Path:
    uid_getter = getattr(os, "getuid", None)
    uid = uid_getter() if callable(uid_getter) else None
    dirname = f"{_DM_DIR_NAME}-{uid}" if uid is not None else _DM_DIR_NAME
    path = Path(tempfile.gettempdir()) / dirname
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


def cleanup_bot_dm_cache(
    max_age_hours: float = _DM_STALE_SECONDS / 3600, *, now: float | None = None
) -> int:
    """Delete orphaned DM payload files older than *max_age_hours*.

    Same contract as the other ``cleanup_*_cache`` helpers — returns the
    number of files removed — so the gateway housekeeping loop can prune
    this cache on the same hourly cadence as the media caches, even on
    installs that never send another DM (the in-band sweep in
    ``_write_dm_file`` only runs when a DM is written).
    """
    cutoff = (time.time() if now is None else now) - max_age_hours * 3600
    removed = 0
    # Include the legacy temp-root locations so upgrades clean files created
    # by versions predating the dedicated directory.
    temp_root = Path(tempfile.gettempdir())
    locations: list[tuple[Path, str]] = [
        (temp_root, "hermes-dm-*.txt"),
        (temp_root, "hermes-relay-dm-*.txt"),
    ]
    try:
        locations.append((_dm_dir(), "*.txt"))
    except OSError:
        pass
    for directory, pattern in locations:
        try:
            for candidate in directory.glob(pattern):
                try:
                    if candidate.is_file() and candidate.stat().st_mtime < cutoff:
                        candidate.unlink()
                        removed += 1
                except OSError:
                    pass
        except OSError:
            pass
    return removed


def _sweep_stale_dm_files(*, now: float | None = None) -> None:
    """Best-effort cleanup for files orphaned before their runner started."""
    cleanup_bot_dm_cache(now=now)


def _write_dm_file(content: str) -> str:
    """The message rides a temp file — never inline shell text."""
    _sweep_stale_dm_files()
    fd, path = tempfile.mkstemp(prefix="dm-", suffix=".txt", dir=_dm_dir(), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
    except BaseException:
        # fdopen owns the descriptor once it succeeds, but if fdopen itself
        # failed the raw descriptor is still ours. Closing twice is harmless.
        try:
            os.close(fd)
        except OSError:
            pass
        _unlink_dm_file(path)
        raise
    return path


def _unlink_dm_file(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _delivery_lock(argv: list[str], *, stdin_file: bool):
    """Per-profile turn lock context for a LOCAL teammate delivery (#93091).

    Local deliveries (``hermes -p <profile> chat …``) collide with relay
    deliveries into the same profile — both run a Bot Chat turn on this
    install — so the turn window is serialized on the shared cross-process
    lock in ``tools.bot_relay``. Peer transports (stdin mode) run on the
    remote gateway; their turn is locked THERE by its own deliver path.
    """
    # The CLI element is matched by basename: local_delivery_command now
    # resolves the venv-relative hermes next to this gateway's interpreter
    # (#93590 — service contexts lack PATH), so argv[0] may be an absolute
    # path (and on Windows carries the .exe suffix). Split on both
    # separators so the shape matches regardless of which platform built
    # the argv.
    cli = (argv[0] if argv else "").rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    if (
        stdin_file
        or len(argv) < 3
        or cli not in ("hermes", "hermes.exe")
        or argv[1] != "-p"
    ):
        return contextlib.nullcontext()
    from tools.bot_relay import acquire_turn_lock

    home = Path(os.getenv("HERMES_HOME") or os.path.expanduser("~/.hermes"))
    return acquire_turn_lock(_hermes_root(home), argv[2])


def _run_delivery(argv: list[str], dm_file: str, *, stdin_file: bool) -> int:
    """Run one DM transport and remove its plaintext file after consumption.

    The turn execution window (not the enqueue) holds the target profile's
    cross-process lock, so two deliveries into one profile queue instead of
    racing; a bounded wait ends in a structured 'target_busy' refusal.

    Local (query-file) turns get one policy-gated retry (#93091 item 5):
    transient failures re-run the same session; a context_overflow re-run
    lets the retried turn's pre-API compaction pass compact the Bot Chat
    transcript first (agent/conversation_loop.py) — the sanctioned
    compression lever; no fresh session is ever minted. Auth/quota/config
    failures never retry. Peer transports (stdin mode) retry on their own
    gateway's deliver path, not here.
    """
    try:
        with _delivery_lock(argv, stdin_file=stdin_file):
            if stdin_file:
                # Keep the file open until the transport exits; cleanup occurs
                # after subprocess.run returns, not merely after stdin reaches EOF.
                with open(dm_file, "r", encoding="utf-8") as stream:
                    return subprocess.run(argv, stdin=stream, check=False).returncode
            proc = subprocess.run(
                [*argv, "--query-file", dm_file],
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                from tools.bot_failure_reasons import (
                    RETRY_NONE,
                    classify_agent_error,
                    retry_action,
                )

                detail = (proc.stderr or proc.stdout or "").strip()[-500:]
                if retry_action(classify_agent_error(detail)) != RETRY_NONE:
                    proc = subprocess.run(
                        [*argv, "--query-file", dm_file],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
            # Re-emit the transport's streams: stdout is the reply text the
            # completion notification carries back to the sending agent.
            if proc.stdout:
                sys.stdout.write(proc.stdout)
                sys.stdout.flush()
            if proc.stderr:
                sys.stderr.write(proc.stderr)
                sys.stderr.flush()
            return proc.returncode
    finally:
        _unlink_dm_file(dm_file)


def _delivery_command(argv: list[str], dm_file: str, *, stdin_file: bool) -> str:
    """Build an argv-safe command for the cleanup-owning background runner."""
    runner_argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-delivery",
        "stdin" if stdin_file else "query-file",
        dm_file,
        *argv,
    ]
    if sys.platform == "win32":
        # The tracked local backend uses Git Bash on native Windows. Forward
        # slashes preserve native drive paths while remaining executable by
        # that shell; backslash-form paths are parsed as command names and die
        # with exit 127 before this runner starts.
        runner_argv = [part.replace("\\", "/") for part in runner_argv]
    return shlex.join(runner_argv)


def _start_delivery(
    argv: list[str],
    content: str,
    label: str,
    *,
    stdin_file: bool,
    task_id: Optional[str],
    agent: Any,
) -> str:
    """Create a DM file and transfer its cleanup ownership to the runner."""
    dm_file = _write_dm_file(content)
    try:
        command = _delivery_command(argv, dm_file, stdin_file=stdin_file)
    except BaseException:
        _unlink_dm_file(dm_file)
        raise
    return _spawn_delivery(
        command,
        label,
        dm_file=dm_file,
        task_id=task_id,
        agent=agent,
    )


def _spawn_delivery(
    command: str,
    label: str,
    *,
    dm_file: Optional[str] = None,
    task_id: Optional[str],
    agent: Any,
) -> str:
    """Launch the cleanup-owning runner and transfer file ownership on ack.

    ``dm_file`` is None for relay deliveries: the waiter command watches a
    reply file, and the envelope artifacts are owned and swept by
    ``tools/bot_relay.py`` — there is no plaintext DM tempfile to reclaim.
    """
    transferred = False
    try:
        from tools.terminal_tool import terminal_tool

        raw = terminal_tool(
            command,
            background=True,
            notify_on_complete=True,
            task_id=task_id,
            workdir=str(Path(__file__).resolve().parent.parent),
            _host_local=True,
        )
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            parsed = {}
        proc_id = parsed.get("session_id") or ""
        if parsed.get("error"):
            return _err(f"Delivery to {label} failed to start: {parsed['error']}")
        if not proc_id:
            return _err(f"Delivery to {label} failed to start: no process id returned")
        # From this point the background runner owns the file and removes it
        # only after the local query-file or peer stdin consumer has finished.
        transferred = True
        return json.dumps(
            {
                "status": "sent",
                "to": label,
                "detail": (
                    f"Message dispatched to {label}. This is asynchronous — do NOT wait "
                    "or poll. Finish your turn now; when the delivery completes, its "
                    "notification carries the reply — relay it then, attributed to "
                    "that agent."
                ),
                **({"process_id": proc_id} if proc_id else {}),
                "sent_at": int(time.time()),
            }
        )
    except Exception as exc:
        logger.error("message_agent delivery spawn failed: %s", exc, exc_info=True)
        return _err(f"Delivery to {label} could not be started: {exc}")
    finally:
        if dm_file and not transferred:
            _unlink_dm_file(dm_file)


def _delivery_main(args: list[str]) -> int:
    if len(args) < 3 or args[0] != "--run-delivery":
        return 2
    stdin_file = args[1] == "stdin"
    if not stdin_file and args[1] != "query-file":
        return 2
    dm_file = args[2]
    try:
        return _run_delivery(args[3:], dm_file, stdin_file=stdin_file)
    except Exception as exc:
        # 'target_busy' extends the #93091 item-1 structured refusal enum:
        # the queued delivery gave up after its bounded wait — surface the
        # structured payload on stdout so the completion notification carries
        # it back to the sending agent.
        reason = getattr(exc, "reason", "")
        if reason == "target_busy":
            print(json.dumps({"error": str(exc), "reason": "target_busy"}))
            return 1
        print(
            f"message_agent delivery failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


# ── agent-context helpers (mirror system_prompt.py's resolution) ─────────────


def _agent_home(agent: Any) -> str:
    """The calling agent's OWN home (session-db derived), not ambient env."""
    try:
        sdb = getattr(agent, "_session_db", None)
        db_path = getattr(sdb, "db_path", None)
        if db_path:
            return str(Path(db_path).parent)
    except Exception:
        pass
    return os.getenv("HERMES_HOME") or os.path.expanduser("~/.hermes")


def _session_title(agent: Any) -> str:
    title = str(getattr(agent, "_session_title_hint", "") or "").strip()
    if title:
        return title
    try:
        sdb = getattr(agent, "_session_db", None)
        sid = getattr(agent, "session_id", None)
        if sdb and sid:
            return str(sdb.get_session_title(sid) or "").strip()
    except Exception:
        pass
    return ""


if __name__ == "__main__":  # pragma: no cover - exercised as a background process
    raise SystemExit(_delivery_main(sys.argv[1:]))
