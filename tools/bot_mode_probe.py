"""Bot Mode roster probe — canonical Bot Chat system prompt section.

When the desktop's Bot Mode manages this install, routed Bot Mode sessions
receive a short "Messaging other agents" section so the bot can receive
teammate DMs, reply with attribution, and hand off @mentions. Routed means:

- the canonical "Bot Chat" for any profile in a Bot-Mode-participating
  install, or
- a classified human messaging chat (Discord, Telegram, Slack, ...) routed to
  one of that install's real profiles.

Regular self-owned sessions (CLI, TUI, cron, subagents, ...), machine/API
adapters, paths outside the install's profile roster, and every session on an
unmanaged install never carry the section; the desktop's composer middleware
owns the @mention send path there.

The shared :func:`bot_mode_session_state` gate is used by the prompt,
schema-injection, and dispatch paths so defense in depth cannot drift.

This replaces the plugin-side SOUL.md backfill: the protocol is injected by
the core at prompt-build time instead of appended to user-authored SOUL
files.  If the profile's SOUL.md already carries the section (created by an
older plugin version), the probe stays silent so the text never doubles up.

Silent (returns ``""``) when:
- no profile on this install is Bot-Mode-managed (the dominant case),
- the current profile's SOUL.md already contains the protocol heading,
- anything at all goes wrong (never crash a prompt build).

Deterministic within a process: the result is computed once and cached, so
compression-triggered prompt rebuilds produce identical bytes.

Toggle via ``agent.bot_mode_protocol`` in config.yaml (default True).
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path

_PROTOCOL_HEADING = "## Messaging other agents"
_PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

# The canonical per-bot conversation title — the only session shape that
# receives the protocol section. Must match the desktop plugin's
# createCanonicalChat title and the `-c "Bot Chat"` resume target.
BOT_CHAT_TITLE = "Bot Chat"

_lock = threading.Lock()
_cached: dict[str, str] = {}
_session_state_lock = threading.Lock()
# Process bridge for agent recreation. The agent-local entry owns active-session
# stability; this LRU only preserves a bounded window across object recreation.
_SESSION_STATE_CACHE_MAX = 1024
_session_state_cache: dict[tuple[str, str, str], dict] = {}


def _hermes_root(home: Path) -> Path:
    """Root ~/.hermes for both the default profile and named profiles."""
    if home.parent.name == "profiles":
        return home.parent.parent
    return home


def _profile_name(home: Path) -> str:
    if home.parent.name == "profiles":
        return home.name
    return "default"


def _is_bot_managed(profile_dir: Path) -> bool:
    """True when profile.yaml carries a ui_meta['hermes-bots'] block.

    Cheap substring check before the YAML parse keeps the silent path fast.
    """
    meta = profile_dir / "profile.yaml"
    try:
        if not meta.is_file():
            return False
        raw = meta.read_text(encoding="utf-8", errors="replace")
        if "hermes-bots" not in raw:
            return False
        import yaml

        data = yaml.safe_load(raw)
        ui_meta = data.get("ui_meta") if isinstance(data, dict) else None
        return isinstance(ui_meta, dict) and isinstance(ui_meta.get("hermes-bots"), dict)
    except Exception:
        return False


def _absolute_without_symlink_resolution(path: Path) -> Path:
    """Absolute lexical path, preserving ``profiles/<name>`` containment."""
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _is_roster_profile_dir(root: Path, candidate: Path) -> bool:
    """Validate a security-filtered profile-roster directory.

    This follows ``profiles.list`` name/tombstone rules and additionally rejects
    symlinks. Resolving only after the lexical parent check prevents
    ``profiles/<name>`` links from escaping the install and being reclassified
    as another install's default profile.
    """
    try:
        root = _absolute_without_symlink_resolution(root)
        candidate = _absolute_without_symlink_resolution(candidate)
        if candidate == root:
            return root.is_dir()
        profiles = root / "profiles"
        if (
            candidate.parent != profiles
            or not _PROFILE_ID_RE.fullmatch(candidate.name)
            or candidate.is_symlink()
            or profiles.is_symlink()
            or not candidate.is_dir()
            or (profiles / ".deleted" / candidate.name).exists()
        ):
            return False
        profiles_real = profiles.resolve(strict=True)
        return candidate.resolve(strict=True) == profiles_real / candidate.name
    except (OSError, RuntimeError):
        return False


def _roster(root: Path) -> list[tuple[str, Path]]:
    """Valid default + named profile directories, with symlinks denied."""
    root = _absolute_without_symlink_resolution(root)
    entries: list[tuple[str, Path]] = []
    if _is_roster_profile_dir(root, root):
        entries.append(("default", root))
    try:
        profiles = root / "profiles"
        if profiles.is_dir() and not profiles.is_symlink():
            for child in sorted(profiles.iterdir()):
                if child.name != "default" and _is_roster_profile_dir(root, child):
                    entries.append((child.name, child))
    except OSError:
        pass
    return entries


def is_bot_mode_managed(home: str | os.PathLike | None = None) -> bool:
    """True when ANY profile on this install is Bot-Mode-managed.

    The tool-injection gate for ``message_agent`` — deliberately independent
    of :func:`get_bot_mode_protocol_section`'s emptiness: a profile whose
    SOUL.md carries the legacy plugin-appended protocol gets an empty
    section (text dedupe) but must still get the tool. Never raises.
    """
    try:
        resolved = Path(
            str(home) if home else (os.getenv("HERMES_HOME") or os.path.expanduser("~/.hermes"))
        )
        root = _hermes_root(resolved)
        return any(_is_bot_managed(d) for _n, d in _roster(root))
    except Exception:
        return False


def is_bot_mode_roster_profile(home: str | os.PathLike | None = None) -> bool:
    """True when ``home`` is a real profile in a participating install's roster.

    Bot Mode's roster is ``profiles.list``: every installed profile is a bot,
    while ``ui_meta['hermes-bots']`` is optional presentation data written only
    after customization. Named profiles must be valid, live immediate
    ``profiles/`` children; the install root is the implicit default profile.
    Never raises.
    """
    try:
        candidate = _absolute_without_symlink_resolution(
            Path(
                str(home)
                if home
                else (os.getenv("HERMES_HOME") or os.path.expanduser("~/.hermes"))
            )
        )
        root = _hermes_root(candidate)
        return _is_roster_profile_dir(root, candidate)
    except Exception:
        return False


# ── messaging-gateway session gate ───────────────────────────────────────────
#
# Only sources whose inbound unit is a human-authored conversation may expose
# cross-profile teammate routing. This is deliberately an allowlist: Platform
# also contains API endpoints, automation event streams, and agent-to-agent task
# protocols. A newly registered adapter therefore fails closed until its trust
# model is reviewed here.
_CANONICAL_BOT_CHAT_SESSION_SOURCES = frozenset({"", "cli", "tui", "desktop"})
_MESSAGING_GATEWAY_SESSION_SOURCES = frozenset({
    # Built-in adapters.
    "telegram", "discord", "whatsapp", "whatsapp_cloud", "slack", "signal",
    "mattermost", "matrix", "email", "sms", "dingtalk", "feishu", "wecom",
    "wecom_callback", "weixin", "bluebubbles", "qqbot", "yuanbao",
    # Bundled plugin adapters carrying human chats/messages.
    "buzz", "google_chat", "irc", "line", "photon", "simplex", "teams",
})


def _session_source(agent: object) -> str:
    try:
        from gateway.session_context import resolve_session_source

        return str(resolve_session_source(getattr(agent, "platform", None))).strip().lower()
    except Exception:
        # Resolver failures are authorization failures, never legacy CLI trust.
        return "__untrusted__"


def is_messaging_gateway_session(agent: object) -> bool:
    """True only for classified human messaging-gateway conversations.

    Machine/API surfaces (including A2A, Home Assistant, Raft, webhooks, and
    arbitrary future plugins) fail closed even when they are valid registered
    ``Platform`` values. Stable for a session's lifetime; never raises.
    """
    try:
        return _session_source(agent) in _MESSAGING_GATEWAY_SESSION_SOURCES
    except Exception:
        return False


def bot_mode_dispatch_authorized(
    agent: object, home: str | os.PathLike | None = None
) -> bool:
    """Live, fail-closed authorization for a ``message_agent`` delivery.

    Unlike the frozen presentation gate, this re-reads revocable config and
    roster state immediately before dispatch. A cached prompt/schema may remain
    byte-stable, but it cannot preserve delivery authority after Bot Mode is
    disabled, the sender leaves the roster, or the authoritative source changes.
    """
    try:
        if not bool(getattr(agent, "_bot_mode_protocol", True)):
            return False
        resolved = _absolute_without_symlink_resolution(
            Path(home if home is not None else _agent_home(agent))
        )
        if not is_bot_mode_managed(resolved) or not is_bot_mode_roster_profile(resolved):
            return False
        source = _session_source(agent)
        return (
            source in _CANONICAL_BOT_CHAT_SESSION_SOURCES
            and _session_title(agent) == BOT_CHAT_TITLE
        ) or source in _MESSAGING_GATEWAY_SESSION_SOURCES
    except Exception:
        return False


def _agent_home(agent: object) -> str:
    """The routed profile home: ContextVar first, shared DB only as fallback.

    Multiplex gateways bind the current profile with
    ``set_hermes_home_override`` while every agent may still point at the
    launch/default ``state.db``. Threads that lose that context fall back to
    the DB parent, matching :func:`agent.system_prompt._agent_home`.
    """
    try:
        from hermes_constants import get_hermes_home_override

        override = get_hermes_home_override()
        if override:
            return override
    except Exception:
        pass
    try:
        sdb = getattr(agent, "_session_db", None)
        db_path = getattr(sdb, "db_path", None)
        if db_path:
            return str(Path(db_path).parent)
    except Exception:
        pass
    return os.getenv("HERMES_HOME") or os.path.expanduser("~/.hermes")


def _session_title(agent: object) -> str:
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


def bot_mode_session_state(
    agent: object, home: str | os.PathLike | None = None
) -> dict:
    """The single Bot Mode routing answer, shared by every gate.

    Returns ``{"managed": bool, "session_kind": str | None}`` where
    ``session_kind`` is:

    - ``"bot_chat"``  — the canonical "Bot Chat" of a real profile in a
      Bot-Mode-participating install,
    - ``"gateway"``   — a classified human messaging chat routed to a real
      profile in that install,
    - ``None``        — gated: unmanaged installs, paths outside the profile
      roster, self-owned sessions, machine/API adapters, arbitrary sources.

    The answer is frozen on the agent by normalized source, and bridged across
    same-session agent recreation by a bounded process LRU keyed by profile
    home + persisted session identity + normalized source. The source component
    keeps the API/A2A deny boundary intact when one persisted session is resumed
    by agents on different adapters, and keeps a denied source from poisoning a
    trusted classification. Eviction only ends cross-object reuse; a live
    agent's local copy remains byte-stable for prompt/schema caching. Explicit
    ``home`` probes are uncached. Never raises; fails closed to ``None``.
    """
    cache_key = None
    cached = None
    source = ""
    try:
        if home is None:
            agent_cached = getattr(agent, "_bot_mode_session_state", None)
            if (
                isinstance(agent_cached, tuple)
                and len(agent_cached) == 2
                and isinstance(agent_cached[1], dict)
                and "session_kind" in agent_cached[1]
            ):
                # Agent-local reuse is only valid for the same normalized
                # source; a different platform adapter on the same persisted
                # session must reclassify (API/A2A stay denied, and a denied
                # source must not poison a trusted one).
                if agent_cached[0] == _session_source(agent):
                    return agent_cached[1]

        protocol_enabled = bool(getattr(agent, "_bot_mode_protocol", True))
        resolved = str(
            _absolute_without_symlink_resolution(
                Path(home if home else _agent_home(agent))
            )
        )
        title = _session_title(agent)
        session_id = str(getattr(agent, "session_id", "") or "")
        source = _session_source(agent)

        if home is None and session_id:
            # Session identity + normalized source own the frozen answer.
            # Title, platform-independent config, and metadata changes cannot
            # perturb a live conversation, but a different source on the same
            # persisted session reclassifies instead of reusing trust.
            cache_key = (resolved, session_id, source)
            with _session_state_lock:
                cached = _session_state_cache.pop(cache_key, None)
                if cached is not None:
                    # Dict insertion order is the LRU order. Refresh hits so
                    # active conversations survive traffic-driven eviction.
                    _session_state_cache[cache_key] = cached
            if cached is not None:
                try:
                    setattr(agent, "_bot_mode_session_state", (source, cached))
                except Exception:
                    pass
                return cached

        if not protocol_enabled:
            state = {"managed": False, "session_kind": None}
        else:
            managed = is_bot_mode_managed(resolved)
            roster_profile = is_bot_mode_roster_profile(resolved)
            if not managed:
                state = {"managed": False, "session_kind": None}
            elif (
                roster_profile
                and title == BOT_CHAT_TITLE
                and _session_source(agent) in _CANONICAL_BOT_CHAT_SESSION_SOURCES
            ):
                state = {"managed": True, "session_kind": "bot_chat"}
            elif roster_profile and is_messaging_gateway_session(agent):
                state = {"managed": True, "session_kind": "gateway"}
            else:
                state = {"managed": True, "session_kind": None}
    except Exception:
        state = {"managed": False, "session_kind": None}

    if home is None:
        if cache_key is not None:
            with _session_state_lock:
                existing = _session_state_cache.pop(cache_key, None)
                if existing is not None:
                    state = existing
                _session_state_cache[cache_key] = state
                limit = max(1, int(_SESSION_STATE_CACHE_MAX))
                while len(_session_state_cache) > limit:
                    _session_state_cache.pop(next(iter(_session_state_cache)), None)
        try:
            # Bind the agent-local copy to its normalized source so a later
            # call from a different platform adapter on the same persisted
            # session reclassifies instead of reusing trust across the
            # API/A2A deny boundary. Same-source recreation still hits the
            # fast path, keeping the prompt and tool schema byte-stable.
            setattr(agent, "_bot_mode_session_state", (source, state))
        except Exception:
            pass
    return state


def _soul_has_protocol(profile_dir: Path) -> bool:
    try:
        soul = profile_dir / "SOUL.md"
        return soul.is_file() and _PROTOCOL_HEADING in soul.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False


def _handle(name: str) -> str:
    # The mention middleware aliases the default profile as @hermes.
    return "hermes" if name == "default" else name


def _profile_role(profile_dir: Path) -> str:
    """A teammate's role line: Bot Mode title, else profile description.

    The ui_meta['hermes-bots'].title is the name the user gave the bot in
    Bot Mode; profile.yaml's description is the profile's stated purpose.
    Either one tells a teammate WHO to message for a given job. Bounded and
    single-line; empty when neither exists. Never raises.
    """
    meta = profile_dir / "profile.yaml"
    try:
        if not meta.is_file():
            return ""
        raw = meta.read_text(encoding="utf-8", errors="replace")
        import yaml

        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            return ""
        parts = []
        ui_meta = data.get("ui_meta")
        if isinstance(ui_meta, dict) and isinstance(ui_meta.get("hermes-bots"), dict):
            title = str(ui_meta["hermes-bots"].get("title") or "").strip()
            if title:
                parts.append(title)
        description = str(data.get("description") or "").strip()
        if description:
            parts.append(description)
        line = " — ".join(parts)
        return " ".join(line.split())[:160]
    except Exception:
        return ""


def _roster_lines(root: Path, me: str) -> list[str]:
    """One '- `@handle` — role' line per teammate (excluding ``me``)."""
    lines = []
    for name, profile_dir in _roster(root):
        if name == me:
            continue
        role = _profile_role(profile_dir)
        handle = _handle(name)
        lines.append(f"- `@{handle}`" + (f" — {role}" if role else ""))
    return lines


def _peers(root: Path) -> list[str]:
    """Registered peer gateway names (``hermes peer``), for the protocol text.

    Reads config.yaml directly (cheap, no config-loader import) — the section
    is optional and absent on most installs. Never raises.
    """
    try:
        cfg_path = root / "config.yaml"
        if not cfg_path.is_file():
            return []
        raw = cfg_path.read_text(encoding="utf-8", errors="replace")
        if "bot_peers" not in raw:
            return []
        import yaml

        data = yaml.safe_load(raw)
        peers = data.get("bot_peers") if isinstance(data, dict) else None
        if not isinstance(peers, dict):
            return []
        return sorted(str(name) for name in peers if str(name).strip())
    except Exception:
        return []


def _remote_paragraph(root: Path) -> str:
    """Protocol addendum for agents on OTHER connected machines.

    Fed by the Desktop relay roster (``tools/bot_relay.py``) — every gateway
    connected to the user's Desktop (local, remote URL, SSH, Hermes Cloud,
    docker) syncs its agents here, so bots can DM across machines with the
    same message_agent tool. Only rendered when the relay roster is
    non-empty.
    """
    try:
        from tools.bot_relay import read_remote_roster, remote_target_forms

        roster = read_remote_roster(root)
    except Exception:
        return ""
    if not roster:
        return ""
    lines = []
    for row, form in zip(roster, remote_target_forms(roster)):
        where = row["connection_label"] or row["connection_id"]
        role = " — ".join(p for p in (row["title"], row["description"]) if p)
        lines.append(
            f"- `@{form}` — on {where}" + (f" — {role}" if role else "")
        )
    return (
        "\n\nTeammates on OTHER connected machines (reachable through the "
        "Desktop relay — message them with message_agent exactly like local "
        "teammates; replies arrive as completion notifications the same "
        "way):\n" + "\n".join(lines)
    )


def _peer_paragraph(root: Path) -> str:
    """Protocol addendum for cross-machine DMs — only when peers exist."""
    peers = _peers(root)
    if not peers:
        return ""
    listed = ", ".join(f"`{p}`" for p in peers)
    return (
        "\n\nTeammates on OTHER machines: this install also has peer gateways "
        f"registered ({listed}). Message an agent on a peer the same way — "
        'message_agent with target "<peer>/<agent-name>" (or "<peer>" alone '
        "for the peer's main agent). Run `hermes peer list` for the live "
        "peer list."
    )


def _build_section(home: Path) -> str:
    root = _hermes_root(home)
    me = _profile_name(home)

    roster = _roster(root)
    if not any(_is_bot_managed(d) for _n, d in roster):
        return ""

    # An older plugin build may have appended the protocol to SOUL.md
    # already — never double it up.
    my_dir = home if me == "default" else root / "profiles" / me
    if _soul_has_protocol(my_dir):
        return ""

    handle = _handle(me)
    roster_block = "\n".join(_roster_lines(root, me)) or "- (no teammates yet)"

    return (
        f"{_PROTOCOL_HEADING}\n"
        "This install runs Bot Mode: each Hermes profile is an agent teammate with "
        'one canonical "Bot Chat" conversation, and you have the `message_agent` '
        "tool to DM any of them — from this messaging chat (Discord, Telegram, "
        "Slack, ...) or from your Bot Chat. It is FIRE-AND-FORGET: it delivers your "
        "message with your attribution prefixed automatically and returns an "
        "acknowledgement immediately — it never returns the reply. Send it, finish "
        "your turn, and the reply arrives later as a background-process completion "
        "notification that wakes you; relay it to the user then, attributed to that "
        "agent. COMPOSE every message yourself — say what YOU need from that agent; "
        "never forward the user's words verbatim, and never reveal private 1:1 chat "
        "content. When the user says \"ask <name>\" or \"tell <name> ...\", that is "
        "a handoff: pick the right teammate from the roster below, message them "
        "with message_agent, and report back naming which agent replied. Message "
        "ONE clearly relevant teammate; don't fan out to several unless the user "
        "explicitly asked.\n"
        f'When YOU receive a "Message from 🤖 <name> (@<handle>):" message, a '
        "teammate agent is talking to you (not the user): address them, reply "
        "concisely via message_agent to their handle, and if it is a pure FYI "
        "with nothing to add, staying silent is fine — never ping-pong "
        "acknowledgements.\n"
        f"You are `@{handle}`. Your teammates (live roster; roles from their "
        "profiles):\n"
        f"{roster_block}"
        + _remote_paragraph(root)
        + _peer_paragraph(root)
    )


def get_bot_mode_protocol_section(home: str | os.PathLike | None = None, *, force_refresh: bool = False) -> str:
    """Cached probe entry point — one filesystem pass per (process, home).

    ``home`` should be the AGENT'S OWN resolved home (session-db derived),
    not the ambient HERMES_HOME — build threads can lose the ContextVar
    override and the env var would then name the wrong profile.
    """
    resolved = str(home) if home else (os.getenv("HERMES_HOME") or os.path.expanduser("~/.hermes"))
    with _lock:
        if force_refresh or resolved not in _cached:
            try:
                _cached[resolved] = _build_section(Path(resolved))
            except Exception:
                _cached[resolved] = ""
        return _cached[resolved]


# ── capability epoch ─────────────────────────────────────────────────────────
#
# Bot Chat sessions are effectively eternal — the "new sessions come along
# often" assumption behind build-once system prompts does not hold. When the
# user changes a bot's capabilities (skills, toolsets, MCP servers, SOUL) or
# the teammate roster changes, they expect the change to work on the NEXT
# message. The fingerprint below hashes exactly that capability surface; the
# built Bot Chat prompt embeds it, and the restore path in
# agent/conversation_loop.py rebuilds the prompt when the stored epoch no
# longer matches the disk state. This is the /model exception applied to
# capabilities: a LOUD, USER-INITIATED, once-per-change cache break — never
# a per-turn drift (unchanged state hashes identically and the stored bytes
# are reused verbatim).

_EPOCH_PREFIX = "Capability epoch: "
_EPOCH_RE_TEXT = r"Capability epoch: ([0-9a-f]{12})"


def capability_fingerprint(home: str | os.PathLike | None = None) -> str:
    """12-hex digest of the capability surface for ``home``'s profile.

    Sources: the profile's disabled skills + enabled toolsets + MCP server
    config (config.yaml), SOUL.md bytes, installed skill names, and the
    Bot-Mode roster (managed profile names). Deliberately NOT cached — the
    whole point is detecting on-disk drift; callers compare it against the
    epoch embedded in a stored prompt. Never raises.
    """
    import hashlib
    import json

    resolved = Path(str(home) if home else (os.getenv("HERMES_HOME") or os.path.expanduser("~/.hermes")))
    surface: dict = {}
    try:
        # Canonical loader (managed overlay + env expansion + normalization),
        # scoped to the bot's home via the override the loaders already honor.
        from hermes_cli.config import load_config_readonly
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        token = set_hermes_home_override(str(resolved))
        try:
            cfg = load_config_readonly() or {}
        finally:
            reset_hermes_home_override(token)
        skills_cfg = cfg.get("skills") if isinstance(cfg.get("skills"), dict) else {}
        tools_cfg = cfg.get("tools") if isinstance(cfg.get("tools"), dict) else {}
        skills_cfg = skills_cfg or {}
        tools_cfg = tools_cfg or {}
        surface["disabled_skills"] = sorted(str(s).lower() for s in (skills_cfg.get("disabled") or []))
        surface["enabled_toolsets"] = sorted(str(t) for t in (tools_cfg.get("enabled_toolsets") or []))
        mcp = cfg.get("mcp_servers")
        surface["mcp"] = json.dumps(mcp, sort_keys=True, default=str) if isinstance(mcp, dict) else ""
    except Exception:
        pass
    try:
        soul = resolved / "SOUL.md"
        surface["soul"] = hashlib.sha256(soul.read_bytes()).hexdigest() if soul.is_file() else ""
    except Exception:
        surface["soul"] = ""
    try:
        names = []
        skills_root = resolved / "skills"
        if skills_root.is_dir():
            for skill_md in skills_root.glob("**/SKILL.md"):
                names.append(str(skill_md.parent.relative_to(skills_root)))
        surface["skills"] = sorted(names)
    except Exception:
        surface["skills"] = []
    try:
        root = _hermes_root(resolved)
        surface["roster"] = sorted(n for n, d in _roster(root) if _is_bot_managed(d))
        # Roles are part of the messaging surface: renaming a bot or editing
        # a profile description must refresh eternal Bot Chat prompts so the
        # roster block teammates pick recipients from stays current.
        surface["roster_roles"] = sorted(
            f"{n}:{_profile_role(d)}" for n, d in _roster(root)
        )
    except Exception:
        surface["roster"] = []
    # Protocol-text version salt: bumping this refreshes every eternal Bot
    # Chat prompt ONCE so existing bots adopt a new protocol section (e.g.
    # the v3 wording adding messaging-gateway chats as a message_agent
    # surface).
    surface["protocol_version"] = 3
    try:
        # Peer gateways are part of the messaging surface: registering one
        # must refresh eternal Bot Chat prompts so the cross-machine DM
        # paragraph appears on the next message.
        surface["peers"] = _peers(_hermes_root(resolved))
    except Exception:
        surface["peers"] = []
    try:
        # The Desktop relay roster is part of the messaging surface too:
        # connecting/disconnecting a machine, or agents appearing on one,
        # must refresh eternal Bot Chat prompts the same way.
        from tools.bot_relay import read_remote_roster

        surface["remote_roster"] = sorted(
            f"{r['connection_id']}:{r['profile']}:{r['title']}"
            for r in read_remote_roster(_hermes_root(resolved))
        )
    except Exception:
        surface["remote_roster"] = []
    try:
        blob = json.dumps(surface, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]
    except Exception:
        return "unavailable"


def epoch_line(home: str | os.PathLike | None = None) -> str:
    """The epoch stamp appended to a Bot Chat prompt."""
    return f"{_EPOCH_PREFIX}{capability_fingerprint(home)}"


def stored_prompt_has_bot_mode_protocol(stored_prompt: str) -> bool:
    """True only for prompts stamped with the generated Bot Mode protocol."""
    try:
        prompt = stored_prompt or ""
        return _EPOCH_PREFIX in prompt and _PROTOCOL_HEADING in prompt
    except Exception:
        return False


def stored_prompt_capability_stale(stored_prompt: str, home: str | os.PathLike | None = None) -> bool:
    """True when ``stored_prompt`` is a Bot Chat prompt whose embedded
    capability epoch no longer matches the current disk state.

    Non-Bot-Chat prompts (no epoch stamp) are never stale by this check.
    Fails closed to "not stale" — a broken probe must never turn into a
    rebuild-every-turn cache burner.
    """
    import re

    try:
        m = re.search(_EPOCH_RE_TEXT, stored_prompt or "")
        if not m:
            return False
        current = capability_fingerprint(home)
        if current == "unavailable":
            return False
        return m.group(1) != current
    except Exception:
        return False


def stored_bot_chat_prompt_needs_upgrade(stored_prompt: str, home: str | os.PathLike | None = None) -> bool:
    """True when a Bot Chat session's stored prompt PREDATES this feature.

    Legacy Bot Chats (created before bundling / this epoch mechanism)
    persisted prompts with no protocol section and no epoch stamp; without
    an explicit upgrade they would be stranded forever — the staleness check
    above only fires on stamped prompts. This is a one-time migration per
    legacy session: the caller must only invoke it for sessions titled
    "Bot Chat", and only rebuilds when the probe would actually emit a
    section (a profile whose SOUL.md already carries the legacy plugin-side
    append keeps its protocol-free prompt — rebuilding those would loop,
    since the probe stays silent and the rebuilt prompt would be unstamped
    again). Fails closed to "no upgrade".
    """
    try:
        if _EPOCH_PREFIX in (stored_prompt or ""):
            return False
        if _PROTOCOL_HEADING in (stored_prompt or ""):
            return False
        # Only upgrade when the rebuild would actually add the section —
        # this is what guarantees the rebuilt prompt carries a stamp and
        # the upgrade can never re-fire.
        return bool(get_bot_mode_protocol_section(home))
    except Exception:
        return False


def _reset_cache_for_tests() -> None:
    with _lock:
        _cached.clear()
    with _session_state_lock:
        _session_state_cache.clear()
