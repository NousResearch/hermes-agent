"""Bot Mode roster probe — canonical Bot Chat system prompt section.

When the desktop's Bot Mode manages this install (any profile carries a
``ui_meta['hermes-bots']`` block in its profile.yaml), a bot's canonical
"Bot Chat" session — and ONLY that session — gets a short "Messaging other
agents" section so the bot can receive teammate DMs, reply with attribution,
and hand off @mentions.  Regular sessions never carry the section; the
desktop's composer middleware owns the @mention send path there.

The caller (agent/system_prompt.py) enforces the session-title gate against
``BOT_CHAT_TITLE``; this module answers "is this install Bot-Mode-managed,
and what should the section say for this profile".

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
import threading
from pathlib import Path

_PROTOCOL_HEADING = "## Messaging other agents"

# The canonical per-bot conversation title — the only session shape that
# receives the protocol section. Must match the desktop plugin's
# createCanonicalChat title and the `-c "Bot Chat"` resume target.
BOT_CHAT_TITLE = "Bot Chat"

_lock = threading.Lock()
_cached: dict[str, str] = {}


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


def _roster(root: Path) -> list[tuple[str, Path]]:
    """(name, dir) for the default profile + every named profile."""
    entries: list[tuple[str, Path]] = [("default", root)]
    try:
        profiles = root / "profiles"
        if profiles.is_dir():
            for child in sorted(profiles.iterdir()):
                if child.is_dir():
                    entries.append((child.name, child))
    except Exception:
        pass
    return entries


def _soul_has_protocol(profile_dir: Path) -> bool:
    try:
        soul = profile_dir / "SOUL.md"
        return soul.is_file() and _PROTOCOL_HEADING in soul.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False


def _handle(name: str) -> str:
    # The mention middleware aliases the default profile as @hermes.
    return "hermes" if name == "default" else name


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
    teammates = ", ".join(f"`{n}`" for n, _d in roster if n != me) or "(none yet)"

    return (
        f"{_PROTOCOL_HEADING}\n"
        "This install runs Bot Mode: each Hermes profile is an agent teammate with "
        'one canonical "Bot Chat" conversation. To message a teammate, run on the '
        "terminal tool (background=true, notify_on_complete=true), then finish your "
        "turn — the reply arrives later as a new message:\n"
        "```\n"
        f'hermes -p <agent-name> chat --in ~ -c "Bot Chat" -Q -q "Message from 🤖 {handle} (@{handle}): your message"\n'
        "```\n"
        f'Always open with the "Message from 🤖 {handle} (@{handle}):" prefix so they '
        "know who is talking. When YOU receive a message with that prefix, you are "
        "being messaged by a teammate agent — address them (not the user) and reply "
        "concisely. When the user says \"ask <name>\" or \"tell <name> ...\", that is a "
        "handoff: message that agent, wait for the reply, and report back, saying "
        "which agent it came from. Run `hermes profile list` for the LIVE teammate "
        f"list before a handoff. Teammates at session start: {teammates}."
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


def _reset_cache_for_tests() -> None:
    with _lock:
        _cached.clear()
