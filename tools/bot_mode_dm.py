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

import json
import logging
import os
import re
import shlex
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

MESSAGE_AGENT_TOOL_NAME = "message_agent"

# Message body cap — generous for real work products, small enough that a
# runaway paste can't turn one DM into a context bomb on the recipient.
MESSAGE_MAX_CHARS = 16000

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
                "the right recipient; targets: a teammate name (e.g. 'researcher'), or "
                "'<peer>/<agent>' for an agent on a registered peer gateway "
                "(e.g. 'spark/researcher', or just '<peer>' for the peer's main agent)."
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
        from tools.bot_mode_probe import BOT_CHAT_TITLE, get_bot_mode_protocol_section

        if _session_title(agent) != BOT_CHAT_TITLE:
            return False
        if not get_bot_mode_protocol_section(_agent_home(agent)):
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
    payload: dict[str, Any] = {"error": message}
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
        from tools.bot_mode_probe import BOT_CHAT_TITLE, get_bot_mode_protocol_section

        title = _session_title(agent)
        if title != BOT_CHAT_TITLE:
            return _err(
                "message_agent is only available in a Bot Mode 'Bot Chat' session. "
                "This session is not one; do not retry."
            )
        if not get_bot_mode_protocol_section(home):
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
        command = f"hermes peer dm {shlex.quote(dm_target)} < {shlex.quote(_write_dm_file(prefix + body))}"
        label = f"@{peer_profile or peer_name} on peer '{peer_name}'"
        return _spawn_delivery(command, label, task_id=task_id, agent=agent)

    # ── local teammate ──
    if not _LOCAL_TARGET_RE.match(raw_target):
        return _err(f"Invalid target: {raw_target!r}.", roster=teammates, peers=peers)
    resolved = _resolve_local_name(raw_target, roster)
    if resolved is None:
        return _err(
            f"No teammate named '{raw_target}' on this install. "
            "Pick a name from the roster (roles are listed in your system prompt).",
            roster=teammates,
            peers=peers,
        )
    if resolved == me:
        return _err("You can't message yourself. Pick a teammate from the roster.")

    dm_file = _write_dm_file(prefix + body)
    command = (
        f"hermes -p {shlex.quote(resolved)} chat --in ~ -c \"Bot Chat\" "
        f"--create-if-missing -Q --query-file {shlex.quote(dm_file)}"
    )
    return _spawn_delivery(command, f"@{_handle(resolved)}", task_id=task_id, agent=agent)


def _write_dm_file(content: str) -> str:
    """The message rides a temp file — never inline shell text."""
    fd, path = tempfile.mkstemp(prefix="hermes-dm-", suffix=".txt", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _spawn_delivery(command: str, label: str, *, task_id: Optional[str], agent: Any) -> str:
    """Run the delivery command tracked + background, notify on completion."""
    try:
        from tools.terminal_tool import terminal_tool

        raw = terminal_tool(
            command,
            background=True,
            notify_on_complete=True,
            task_id=task_id,
        )
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            parsed = {}
        proc_id = parsed.get("session_id") or ""
        if parsed.get("error"):
            return _err(f"Delivery to {label} failed to start: {parsed['error']}")
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
