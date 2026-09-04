"""Bot Mode roster probe — canonical Bot Chat system prompt section.

When any profile carries ``ui_meta['hermes-bots']`` in profile.yaml (Bot-Mode-managed),
a bot's canonical "Bot Chat" session — ONLY that session (agent/system_prompt.py enforces
the ``BOT_CHAT_TITLE`` gate) — gets a "Messaging other agents" section. Silent (``""``)
when no profile is managed, when SOUL.md already carries the heading (legacy plugin text
must never double up), or on any error. Cached per (process, home) so compression rebuilds
produce identical bytes. Toggle: ``agent.bot_mode_protocol``. Also hosts path/roster
helpers shared by ``bot_mode_dm`` and ``bot_relay``.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

_PROTOCOL_HEADING = "## Messaging other agents"

# The only session title that receives the protocol section. Must match the
# desktop plugin's createCanonicalChat title and the `-c "Bot Chat"` resume target.
BOT_CHAT_TITLE = "Bot Chat"

_lock = threading.Lock()
_cached: dict[str, str] = {}


# ── shared path / roster helpers ─────────────────────────────────────────────


def _default_home() -> str:
    """Ambient HERMES_HOME (env, else ~/.hermes) as a string."""
    return os.getenv("HERMES_HOME") or os.path.expanduser("~/.hermes")


def _resolve_home(home: str | os.PathLike | None) -> Path:
    return Path(str(home) if home else _default_home())


def _swallow(fn, default):
    """``fn()`` or ``default`` on any exception — the probe must never crash a prompt build."""
    try:
        return fn()
    except Exception:
        return default


def _hermes_root(home: Path) -> Path:
    """Root ~/.hermes for both the default profile and named profiles."""
    return home.parent.parent if home.parent.name == "profiles" else home


def _profile_name(home: Path) -> str:
    return home.name if home.parent.name == "profiles" else "default"


def _handle(name: str) -> str:
    # The mention middleware aliases the default profile as @hermes.
    return "hermes" if name == "default" else name


def _roster(root: Path) -> list[tuple[str, Path]]:
    """(name, dir) for the default profile + every named profile, sorted."""
    profiles = root / "profiles"
    named = _swallow(lambda: [(c.name, c) for c in sorted(profiles.iterdir()) if c.is_dir()] if profiles.is_dir() else [], [])
    return [("default", root), *named]


def _read_yaml_dict(path: Path, needle: str | None = None) -> dict | None:
    """YAML mapping at ``path``, or None when missing / not a mapping / unreadable. ``needle``:
    cheap substring precheck that skips the YAML parse on the dominant (unmanaged) path."""
    def _load():
        if not path.is_file():
            return None
        raw = path.read_text(encoding="utf-8", errors="replace")
        if needle is not None and needle not in raw:
            return None
        import yaml

        data = yaml.safe_load(raw)
        return data if isinstance(data, dict) else None

    return _swallow(_load, None)


def _bots_meta(data: dict | None) -> dict | None:
    """The ``ui_meta['hermes-bots']`` block of a parsed profile.yaml, if a dict."""
    ui_meta = data.get("ui_meta") if data else None
    bots = ui_meta.get("hermes-bots") if isinstance(ui_meta, dict) else None
    return bots if isinstance(bots, dict) else None


def _is_bot_managed(profile_dir: Path) -> bool:
    return _bots_meta(_read_yaml_dict(profile_dir / "profile.yaml", "hermes-bots")) is not None


def _any_managed(root: Path) -> bool:
    return any(_is_bot_managed(d) for _n, d in _roster(root))


def is_bot_mode_managed(home: str | os.PathLike | None = None) -> bool:
    """True when ANY profile on this install is Bot-Mode-managed. Never raises. The
    ``message_agent`` injection gate — deliberately independent of the protocol section's
    emptiness: a SOUL.md carrying the legacy protocol gets an empty section but still gets the tool."""
    return _swallow(lambda: _any_managed(_hermes_root(_resolve_home(home))), False)


def _soul_has_protocol(profile_dir: Path) -> bool:
    soul = profile_dir / "SOUL.md"
    return _swallow(lambda: soul.is_file() and _PROTOCOL_HEADING in soul.read_text(encoding="utf-8", errors="replace"), False)


def _role_line(*parts: str) -> str:
    """'title — description' from the non-empty parts (either may be absent)."""
    return " — ".join(p for p in parts if p)


def _bullet(handle: str, *parts: str) -> str:
    """Roster line: '- `handle`' plus ' — part' for each non-empty part."""
    return _role_line(f"- `{handle}`", *parts)


def _profile_role(profile_dir: Path) -> str:
    """Teammate role line: Bot Mode title — profile description; tells a teammate
    WHO to message for a job. Single-line, ≤160 chars, "" when neither. Never raises."""
    def _role() -> str:
        data = _read_yaml_dict(profile_dir / "profile.yaml") or {}
        line = _role_line(str((_bots_meta(data) or {}).get("title") or "").strip(),
                          str(data.get("description") or "").strip())
        return " ".join(line.split())[:160]

    return _swallow(_role, "")


def _peers(root: Path) -> list[str]:
    """Registered peer gateway names (``hermes peer``) from config.yaml, read
    directly (no config-loader import; the section is absent on most installs). Never raises."""
    def _names() -> list[str]:
        peers = (_read_yaml_dict(root / "config.yaml", "bot_peers") or {}).get("bot_peers")
        return sorted(str(n) for n in peers if str(n).strip()) if isinstance(peers, dict) else []

    return _swallow(_names, [])


def _remote_roster(root: Path) -> list[dict]:
    """Desktop relay roster (``tools/bot_relay.py``); [] on any failure."""
    def _read():
        from tools.bot_relay import read_remote_roster

        return read_remote_roster(root)

    return _swallow(_read, [])


def _remote_paragraph(root: Path) -> str:
    """Addendum for agents on OTHER connected machines; only when the relay roster is non-empty."""
    roster = _remote_roster(root)
    if not roster:
        return ""
    from tools.bot_relay import remote_target_forms

    lines = [
        _bullet(f"@{form}", f"on {row['connection_label'] or row['connection_id']}", row["title"], row["description"])
        for row, form in zip(roster, remote_target_forms(roster))
    ]
    return (
        "\n\nTeammates on OTHER connected machines (reachable through the "
        "Desktop relay — message them with message_agent exactly like local "
        "teammates; replies arrive as completion notifications the same "
        "way):\n" + "\n".join(lines)
    )


def _peer_paragraph(root: Path) -> str:
    """Addendum for cross-machine DMs — only when peers exist."""
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
    if not _any_managed(root):
        return ""
    # An older plugin build may have appended the protocol to SOUL.md — never double it.
    if _soul_has_protocol(home if me == "default" else root / "profiles" / me):
        return ""

    roster_lines = [_bullet(f"@{_handle(name)}", _profile_role(d)) for name, d in _roster(root) if name != me]
    roster_block = "\n".join(roster_lines) or "- (no teammates yet)"

    return (
        f"{_PROTOCOL_HEADING}\n"
        "This install runs Bot Mode: each Hermes profile is an agent teammate with "
        'one canonical "Bot Chat" conversation, and you have the `message_agent` '
        "tool to DM any of them. It is FIRE-AND-FORGET: it delivers your message "
        "with your attribution prefixed automatically and returns an acknowledgement "
        "immediately — it never returns the reply. Send it, finish your turn, and "
        "the reply arrives later as a background-process completion notification "
        "that wakes you; relay it to the user then, attributed to that agent. "
        "COMPOSE every message yourself — say what YOU need from that agent; never "
        "forward the user's words verbatim, and never reveal private 1:1 chat "
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
        f"You are `@{_handle(me)}`. Your teammates (live roster; roles from their "
        "profiles):\n"
        f"{roster_block}"
        + _remote_paragraph(root)
        + _peer_paragraph(root)
    )


def get_bot_mode_protocol_section(home: str | os.PathLike | None = None, *, force_refresh: bool = False) -> str:
    """Cached probe entry point — one filesystem pass per (process, home). ``home`` should be
    the AGENT'S OWN resolved home (session-db derived), not ambient HERMES_HOME — build threads
    can lose the ContextVar override and the env var would then name the wrong profile."""
    resolved = str(_resolve_home(home))
    with _lock:
        if force_refresh or resolved not in _cached:
            _cached[resolved] = _swallow(lambda: _build_section(Path(resolved)), "")
        return _cached[resolved]


# ── capability epoch ─────────────────────────────────────────────────────────
# Bot Chat sessions are effectively eternal, so "build the prompt once" would strand
# capability changes (skills, toolsets, MCP, SOUL, roster, peers) forever. The fingerprint
# hashes exactly that surface; the built prompt embeds it and agent/conversation_loop.py
# rebuilds only when the stored epoch differs from disk — once per change, never per-turn drift.

_EPOCH_PREFIX = "Capability epoch: "
_EPOCH_RE_TEXT = r"Capability epoch: ([0-9a-f]{12})"


def capability_fingerprint(home: str | os.PathLike | None = None) -> str:
    """12-hex digest of the capability surface for ``home``'s profile: disabled skills +
    enabled toolsets + MCP config, SOUL.md bytes, installed skill names, the Bot-Mode roster
    (+ roles), peers and the relay roster. Deliberately NOT cached — the point is detecting
    on-disk drift against a stored prompt's epoch. Never raises ("unavailable" on failure)."""
    import hashlib
    import json

    resolved = _resolve_home(home)
    root = _hermes_root(resolved)
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
        surface["disabled_skills"] = sorted(str(s).lower() for s in (skills_cfg.get("disabled") or []))
        surface["enabled_toolsets"] = sorted(str(t) for t in (tools_cfg.get("enabled_toolsets") or []))
        mcp = cfg.get("mcp_servers")
        surface["mcp"] = json.dumps(mcp, sort_keys=True, default=str) if isinstance(mcp, dict) else ""
    except Exception:
        pass

    def _soul() -> str:
        soul = resolved / "SOUL.md"
        return hashlib.sha256(soul.read_bytes()).hexdigest() if soul.is_file() else ""

    def _skills() -> list[str]:
        skills_root = resolved / "skills"
        if not skills_root.is_dir():
            return []
        return sorted(str(p.parent.relative_to(skills_root)) for p in skills_root.glob("**/SKILL.md"))

    surface["soul"] = _swallow(_soul, "")
    surface["skills"] = _swallow(_skills, [])
    try:
        roster = _roster(root)
        surface["roster"] = sorted(n for n, d in roster if _is_bot_managed(d))
        # Roles are part of the messaging surface: renaming a bot or editing a
        # description must refresh the roster block teammates pick recipients from.
        surface["roster_roles"] = sorted(f"{n}:{_profile_role(d)}" for n, d in roster)
    except Exception:
        surface["roster"] = []
    # Protocol-text version salt: bumping it refreshes every eternal Bot Chat
    # prompt ONCE so existing bots adopt a new protocol section.
    surface["protocol_version"] = 2
    # Peer gateways and the Desktop relay roster are part of the messaging
    # surface too: registering a peer or (dis)connecting a machine must show up.
    surface["peers"] = _peers(root)
    surface["remote_roster"] = sorted(
        f"{r['connection_id']}:{r['profile']}:{r['title']}" for r in _remote_roster(root)
    )
    return _swallow(
        lambda: hashlib.sha256(json.dumps(surface, sort_keys=True).encode("utf-8")).hexdigest()[:12],
        "unavailable",
    )


def epoch_line(home: str | os.PathLike | None = None) -> str:
    """The epoch stamp appended to a Bot Chat prompt."""
    return f"{_EPOCH_PREFIX}{capability_fingerprint(home)}"


def stored_prompt_capability_stale(stored_prompt: str, home: str | os.PathLike | None = None) -> bool:
    """True when ``stored_prompt`` is a Bot Chat prompt whose embedded epoch no
    longer matches disk. Unstamped prompts are never stale. Fails closed to
    "not stale" — a broken probe must not become a rebuild-every-turn cache burner."""
    import re

    m = re.search(_EPOCH_RE_TEXT, stored_prompt or "")
    if not m:
        return False
    current = _swallow(lambda: capability_fingerprint(home), "unavailable")
    return current != "unavailable" and m.group(1) != current


def stored_bot_chat_prompt_needs_upgrade(stored_prompt: str, home: str | os.PathLike | None = None) -> bool:
    """True when a Bot Chat session's stored prompt PREDATES the epoch mechanism. Legacy
    prompts carry neither section nor stamp, so the staleness check (stamped only) would strand
    them forever. The caller must only ask for sessions titled "Bot Chat"; we rebuild only when
    the probe would actually emit a section — a SOUL.md carrying the legacy protocol yields an
    empty section, and rebuilding would mint another unstamped prompt and loop. Fails closed."""
    text = stored_prompt or ""
    if _EPOCH_PREFIX in text or _PROTOCOL_HEADING in text:
        return False
    return _swallow(lambda: bool(get_bot_mode_protocol_section(home)), False)


def _reset_cache_for_tests() -> None:
    with _lock:
        _cached.clear()
