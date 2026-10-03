"""Bot Mode roster, policy, and session-scoped prompt projection.

Desktop metadata or ``agent.bot_mode.enabled`` opts an install in. Canonical Bot Chats and
trusted human messaging-gateway sessions receive a session-stable protocol section and the
``message_agent`` schema; machine/API/finite gateway sources fail closed. Older desktop builds appended
a frozen copy to SOUL.md, which ``strip_legacy_protocol`` removes at load time. This module also
owns the directed local roster and shared path helpers used by ``bot_mode_dm`` and ``bot_relay``.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

_PROTOCOL_HEADING = "## Messaging other agents"
# The legacy section through the next H2 heading (or EOF), plus the blank lines before it.
_LEGACY_PROTOCOL_RE = re.compile(r"\n*" + re.escape(_PROTOCOL_HEADING) + r"[ \t]*\n.*?(?=\n## |\Z)", re.S)


def strip_legacy_protocol(text: str) -> str:
    """SOUL text without the plugin-era "## Messaging other agents" section (idempotent)."""
    return _LEGACY_PROTOCOL_RE.sub("\n", text).strip() + "\n" if _PROTOCOL_HEADING in text else text


_USER_PROTOCOL_HEADING = "## Bot Mode: messaging other agents"
_SESSION_AUTH_CONFIG_KEY = "_bot_mode_authorized"
_user_surface_cached: dict[str, tuple[str, str]] = {}
_PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

# The only session title that receives the protocol section. Must match the
# desktop plugin's createCanonicalChat title and the `-c "Bot Chat"` resume target.
BOT_CHAT_TITLE = "Bot Chat"

_lock = threading.Lock()
_cached: dict[str, tuple[str, str]] = {}
_session_state_lock = threading.Lock()
_SESSION_STATE_CACHE_MAX = 1024
_session_state_cache: dict[tuple[str, str, str, str], dict] = {}


# ── shared path / roster helpers ─────────────────────────────────────────────


def _default_home() -> str:
    """Ambient process HERMES_HOME (env, else the platform default) as a string."""
    from hermes_constants import get_process_hermes_home
    return str(get_process_hermes_home())


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


def _bot_mode_config(root: Path) -> dict[str, Any] | None:
    """Root declarative policy; missing is legacy, unreadable is denied."""
    from hermes_cli.config import InvalidUserConfigError
    from hermes_cli.config_bot_mode import load_bot_mode_config
    import hermes_yaml as yaml

    try:
        return load_bot_mode_config(root)
    except (OSError, UnicodeError, ValueError, yaml.YAMLError, InvalidUserConfigError):
        return None


def _configured_targets(root: Path, me: str) -> set[str] | None:
    """Allowed targets for ``me``; absent roster is unrestricted, invalid policy denies."""
    cfg = _bot_mode_config(root)
    if cfg is None or cfg.get("enabled") is False:
        return set()
    if "roster" not in cfg:
        return None
    roster = cfg["roster"]
    if not isinstance(roster, list):
        return set()
    targets: set[str] = set()
    for row in roster:
        if not isinstance(row, dict) or not isinstance(row.get("from"), str):
            return set()
        source = row["from"].strip()
        if not _PROFILE_ID_RE.fullmatch(source):
            return set()
        raw_targets = row.get("to")
        if isinstance(raw_targets, str):
            raw_targets = [raw_targets]
        if not isinstance(raw_targets, list) or any(
            not isinstance(target, str) or not _PROFILE_ID_RE.fullmatch(target.strip())
            for target in raw_targets
        ):
            return set()
        normalized_source = "default" if source == "hermes" else source
        if normalized_source == me:
            targets.update(
                "default" if target.strip() == "hermes" else target.strip()
                for target in raw_targets
            )
    return targets


def _is_bot_enabled(profile_dir: Path) -> bool:
    from hermes_cli.profile_bot_policy import read_bot_enabled

    return read_bot_enabled(profile_dir)


def allowed_local_profile_names(home: str | os.PathLike | None = None) -> list[str]:
    """Live intersection of real profiles, target enablement, and directed policy."""
    try:
        resolved = _resolve_home(home)
        root, me = _hermes_root(resolved), _profile_name(resolved)
        configured = _configured_targets(root, me)
        return [
            name for name, directory in _roster(root)
            if name != me
            and _is_bot_enabled(directory)
            and (configured is None or name in configured)
        ]
    except Exception:
        return []


def _handle(name: str) -> str:
    # The mention middleware aliases the default profile as @hermes.
    return "hermes" if name == "default" else name


def _roster(root: Path) -> list[tuple[str, Path]]:
    """(name, dir) for the default profile + every live named profile, sorted. Same identity
    predicate as ``profile list``: infra dirs (``sessions/``, ``logs/``) and tombstones are not
    teammates (#99392), and neither is a marker-carrying dir whose name is not a profile id —
    a parked backup or staging dir must never become a ``message_agent`` target (#116905)."""
    from hermes_constants import PROFILE_ID_RE, named_profile_is_live

    profiles = root / "profiles"
    named = _swallow(
        lambda: [
            (c.name, c)
            for c in sorted(profiles.iterdir())
            if c.name != "default" and PROFILE_ID_RE.match(c.name) and named_profile_is_live(c)
        ]
        if profiles.is_dir()
        else [],
        [],
    )
    return [("default", root), *named]


def _read_yaml_dict(path: Path, needle: str | None = None) -> dict | None:
    """YAML mapping at ``path``, or None when missing / not a mapping / unreadable. ``needle``:
    cheap substring precheck that skips the YAML parse on the dominant (unmanaged) path."""
    def _load():
        if not path.is_file():
            return None
        raw = path.read_text(encoding="utf-8-sig", errors="replace")
        if needle is not None and needle not in raw:
            return None
        import hermes_yaml as yaml

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
    cfg = _bot_mode_config(root)
    if cfg is None or cfg.get("enabled") is False:
        return False
    return cfg.get("enabled") is True or any(_is_bot_managed(d) for _n, d in _roster(root))


def is_bot_mode_managed(home: str | os.PathLike | None = None) -> bool:
    """True when ANY profile on this install is Bot-Mode-managed. Never raises. The
    ``message_agent`` injection gate — deliberately independent of the protocol section's
    emptiness: a SOUL.md carrying the legacy protocol gets an empty section but still gets the tool."""
    return _swallow(lambda: _any_managed(_hermes_root(_resolve_home(home))), False)


def is_bot_mode_roster_profile(home: str | os.PathLike | None = None) -> bool:
    """Whether ``home`` is a non-symlinked profile in this install's live roster."""
    try:
        candidate = Path(os.path.abspath(os.fspath(_resolve_home(home).expanduser())))
        if candidate.is_symlink() or not candidate.is_dir():
            return False
        root = _hermes_root(candidate)
        return any(
            candidate == Path(os.path.abspath(os.fspath(directory.expanduser())))
            for _name, directory in _roster(root)
        )
    except Exception:
        return False


_CANONICAL_BOT_CHAT_SESSION_SOURCES = frozenset({"", "cli", "tui", "desktop"})
_MESSAGING_GATEWAY_SESSION_SOURCES = frozenset({
    "telegram", "discord", "whatsapp", "whatsapp_cloud", "slack", "signal",
    "mattermost", "matrix", "email", "sms", "dingtalk", "feishu", "wecom",
    "wecom_callback", "weixin", "bluebubbles", "qqbot", "yuanbao",
    "buzz", "google_chat", "irc", "line", "photon", "simplex", "teams",
})


def _session_source(agent: object) -> str:
    try:
        from gateway.session_context import resolve_session_source

        return resolve_session_source(getattr(agent, "platform", None))
    except Exception:
        return "__untrusted__"


def is_messaging_gateway_session(agent: object) -> bool:
    """True only for explicitly classified human messaging surfaces."""
    from gateway.session_context import bound_gateway_session_identity

    source = _session_source(agent)
    session_key = str(getattr(agent, "_gateway_session_key", "") or "").strip()
    return (
        source in _MESSAGING_GATEWAY_SESSION_SOURCES
        and bool(session_key)
        and bound_gateway_session_identity() == (source, session_key)
    )


def _agent_home(agent: object) -> str:
    """Resolve the routed profile before falling back to the shared SessionDB."""
    try:
        from hermes_constants import get_hermes_home_override

        override = get_hermes_home_override()
        if override:
            return str(override)
    except Exception:
        pass
    try:
        db_path = getattr(getattr(agent, "_session_db", None), "db_path", None)
        if db_path:
            return str(Path(db_path).parent)
    except Exception:
        pass
    return _default_home()


def _session_title(agent: object) -> str:
    title = str(getattr(agent, "_session_title_hint", "") or "").strip()
    if title:
        return title
    try:
        session_db = getattr(agent, "_session_db", None)
        session_id = getattr(agent, "session_id", None)
        if session_db and session_id:
            return str(session_db.get_session_title(session_id) or "").strip()
    except Exception:
        pass
    return ""


def _internal_session(agent: object) -> bool:
    from agent.delegation_context import is_delegated_child_process_context
    from gateway.session_context import get_session_env
    from utils import is_truthy_value

    return (
        is_truthy_value(get_session_env("HERMES_CRON_SESSION", ""))
        or is_delegated_child_process_context()
    )


def _single_query_session() -> bool:
    from gateway.session_context import get_session_env
    from utils import is_truthy_value

    return is_truthy_value(get_session_env("HERMES_SINGLE_QUERY_SESSION", ""))


def _eligible_session_kind(agent: object, home: Path) -> str | None:
    """Source/profile shape eligible for a frozen Bot Mode presentation decision."""
    if not is_bot_mode_roster_profile(home):
        return None
    source, title = _session_source(agent), _session_title(agent)
    if title == BOT_CHAT_TITLE and source in _CANONICAL_BOT_CHAT_SESSION_SOURCES:
        return "bot_chat"
    if _single_query_session():
        return None
    return "gateway" if is_messaging_gateway_session(agent) else None


def _session_authorization_identity(agent: object) -> dict[str, str]:
    from gateway.session_context import bound_gateway_session_identity

    bound = bound_gateway_session_identity()
    source = _session_source(agent)
    return {
        "source": source,
        "gateway_session_key": (
            bound[1] if bound is not None and bound[0] == source else ""
        ),
    }


def _session_authorization_decision(agent: object, *, authorized: bool) -> dict[str, Any]:
    return {
        **_session_authorization_identity(agent),
        "authorized": bool(authorized),
    }


def _new_session_authorization_state(agent: object, *, authorized: bool) -> dict[str, Any]:
    decision = _session_authorization_decision(agent, authorized=authorized)
    identity = {
        key: decision[key]
        for key in ("source", "gateway_session_key")
    }
    return {"version": 1, "active": identity, "decisions": [decision]}


def _validated_session_authorization_state(
    value: object,
) -> tuple[tuple[str, str], list[dict[str, Any]], dict[tuple[str, str], bool]] | None:
    """Validate the persisted state as one active identity plus unique decisions."""
    if (
        not isinstance(value, dict)
        or value.get("version") != 1
        or not isinstance(value.get("active"), dict)
        or not isinstance(value.get("decisions"), list)
    ):
        return None

    def _identity(candidate: object) -> tuple[str, str] | None:
        if not isinstance(candidate, dict):
            return None
        source = candidate.get("source")
        session_key = candidate.get("gateway_session_key")
        if not isinstance(source, str) or not source or not isinstance(session_key, str):
            return None
        return source, session_key

    active = _identity(value["active"])
    if active is None:
        return None
    decisions: list[dict[str, Any]] = []
    by_identity: dict[tuple[str, str], bool] = {}
    for raw in value["decisions"]:
        identity = _identity(raw)
        if (
            identity is None
            or not isinstance(raw.get("authorized"), bool)
            or identity in by_identity
        ):
            return None
        decision = {
            "source": identity[0],
            "gateway_session_key": identity[1],
            "authorized": raw["authorized"],
        }
        decisions.append(decision)
        by_identity[identity] = decision["authorized"]
    if active not in by_identity:
        return None
    return active, decisions, by_identity


def _persisted_session_authorization(
    agent: object,
) -> tuple[bool | None, bool, bool, bool]:
    """Return current decision, state validity, prompt decision, and identity match."""
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    getter = getattr(session_db, "get_session", None)
    if not session_id or not callable(getter):
        return None, False, False, False
    try:
        row = getter(session_id)
    except Exception:
        return False, True, False, False
    if not isinstance(row, dict) or row.get("system_prompt") is None:
        return None, False, False, False
    raw_config = row.get("model_config")
    try:
        import json

        config = json.loads(raw_config) if isinstance(raw_config, str) else raw_config
    except (TypeError, ValueError):
        return False, True, False, False
    if isinstance(config, dict) and _SESSION_AUTH_CONFIG_KEY in config:
        parsed = _validated_session_authorization_state(
            config[_SESSION_AUTH_CONFIG_KEY],
        )
        if parsed is None:
            return False, True, False, False
        active, _decisions, by_identity = parsed
        current = _session_authorization_identity(agent)
        current_identity = (current["source"], current["gateway_session_key"])
        current_decision = by_identity.get(current_identity)
        prompt_authorized = by_identity[active]
        active_matches = active == current_identity
        return current_decision, True, prompt_authorized, active_matches
    return False, False, False, False


def _live_session_state(agent: object, home: Path) -> dict:
    managed = is_bot_mode_managed(home)
    state = {"managed": managed, "session_kind": None}
    if not getattr(agent, "_bot_mode_protocol", True) or _internal_session(agent):
        return state
    if not managed or not _is_bot_enabled(home):
        return state
    state["session_kind"] = _eligible_session_kind(agent, home)
    return state


def bot_mode_session_state(agent: object, home: str | os.PathLike | None = None) -> dict:
    """Frozen prompt/schema classification keyed by profile, session, and source."""
    cache_key = None
    identity = None
    try:
        if not getattr(agent, "_bot_mode_protocol", True) or _internal_session(agent):
            return {"managed": False, "session_kind": None}
        resolved = str(Path(home if home is not None else _agent_home(agent)).absolute())
        source = _session_source(agent)
        if (
            source in _MESSAGING_GATEWAY_SESSION_SOURCES
            and not is_messaging_gateway_session(agent)
        ):
            return {"managed": False, "session_kind": None}
        identity = (
            resolved,
            str(getattr(agent, "session_id", "") or ""),
            source,
            str(getattr(agent, "_gateway_session_key", "") or ""),
        )
        persisted, structured, _prompt_authorized, _active_matches = (
            _persisted_session_authorization(agent)
        )
        if structured:
            state = {
                "managed": bool(persisted),
                "session_kind": (
                    _eligible_session_kind(agent, Path(resolved))
                    if persisted else None
                ),
            }
        else:
            if home is None:
                agent_cached = getattr(agent, "_bot_mode_session_state", None)
                if (
                    isinstance(agent_cached, tuple)
                    and len(agent_cached) == 2
                    and agent_cached[0] == identity
                    and isinstance(agent_cached[1], dict)
                ):
                    return agent_cached[1]
                if identity[1]:
                    cache_key = identity
                    with _session_state_lock:
                        cached = _session_state_cache.pop(cache_key, None)
                        if cached is not None:
                            _session_state_cache[cache_key] = cached
                    if cached is not None:
                        setattr(agent, "_bot_mode_session_state", (identity, cached))
                        return cached
            legacy_canonical = (
                persisted is False
                and _session_title(agent) == BOT_CHAT_TITLE
                and source in _CANONICAL_BOT_CHAT_SESSION_SOURCES
            )
            state = (
                _live_session_state(agent, Path(resolved))
                if persisted is None or legacy_canonical
                else {"managed": False, "session_kind": None}
            )
    except Exception:
        state = {"managed": False, "session_kind": None}

    if home is None and identity is not None:
        if cache_key is not None:
            with _session_state_lock:
                state = _session_state_cache.setdefault(cache_key, state)
                while len(_session_state_cache) > max(1, int(_SESSION_STATE_CACHE_MAX)):
                    _session_state_cache.pop(next(iter(_session_state_cache)))
        try:
            setattr(agent, "_bot_mode_session_state", (identity, state))
        except Exception:
            pass
    return state


def bot_mode_dispatch_authorized(
    agent: object, home: str | os.PathLike | None = None,
) -> bool:
    """Live fail-closed authorization used immediately before delivery."""
    try:
        resolved = Path(home if home is not None else _agent_home(agent)).absolute()
        return _live_session_state(agent, resolved)["session_kind"] is not None
    except Exception:
        return False


def bot_mode_cached_prompt_needs_rebuild(agent: object) -> bool:
    """Whether the persisted prompt belongs to another or invalid authority identity."""
    try:
        _decision, structured, _prompt_authorized, active_matches = (
            _persisted_session_authorization(agent)
        )
        return structured and not active_matches
    except Exception:
        return False


def persist_bot_mode_session_authorization(agent: object) -> None:
    """Persist this identity's first schema decision and mark its prompt as active."""
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    getter = getattr(session_db, "get_session_model_config_value", None)
    patcher = getattr(session_db, "patch_session_model_config", None)
    if (
        not session_id
        or not getattr(agent, "_session_db_created", False)
        or not callable(getter)
        or not callable(patcher)
    ):
        return
    missing = object()
    existing = getter(session_id, _SESSION_AUTH_CONFIG_KEY, missing)
    desired = _session_authorization_decision(
        agent,
        authorized=(
            "message_agent" in (getattr(agent, "valid_tool_names", None) or ())
        ),
    )
    identity_keys = ("source", "gateway_session_key")
    identity = tuple(desired[key] for key in identity_keys)
    parsed = _validated_session_authorization_state(existing)
    if (
        parsed is not None
        and parsed[0] == identity
        and getattr(agent, "_bot_mode_authorization_recorded_for", None)
        == (session_id, *identity)
    ):
        return
    decisions = list(parsed[1]) if parsed is not None else []
    current = next(
        (
            decision for decision in decisions
            if isinstance(decision, dict)
            and isinstance(decision.get("authorized"), bool)
            and all(decision.get(key) == desired[key] for key in identity_keys)
        ),
        None,
    )
    if current is None:
        decisions.append(desired)
        decisions = decisions[-16:]
        current = desired
    active = {key: desired[key] for key in identity_keys}
    updated = {"version": 1, "active": active, "decisions": decisions}
    if updated != existing:
        patcher(session_id, {_SESSION_AUTH_CONFIG_KEY: updated})
    agent._bot_mode_authorization_recorded_for = (session_id, *identity)


def _role_line(*parts: str) -> str:
    """'title — description' from the non-empty parts (either may be absent)."""
    return " — ".join(p for p in parts if p)


def _bullet(handle: str, *parts: str) -> str:
    """Roster line: '- `handle`' plus ' — part' for each non-empty part."""
    return _role_line(f"- `{handle}`", *parts)


def _profile_role(profile_dir: Path) -> str:
    """Teammate role line: Bot Mode title — profile description; tells a teammate
    WHO to message for a job. A friendly ``display_name`` (``hermes profile rename``) that
    differs from both the folder id and the title leads the line, so an untagged
    "talk to Scribe" maps to the folder handle without a disk search (#100671).
    Single-line, ≤160 chars, "" when nothing. Never raises."""
    def _role() -> str:
        data = _read_yaml_dict(profile_dir / "profile.yaml") or {}
        title = str((_bots_meta(data) or {}).get("title") or "").strip()
        display = str(data.get("display_name") or "").strip()
        if display.lower() in (profile_dir.name.lower(), title.lower()):
            display = ""
        line = _role_line(display, title, str(data.get("description") or "").strip())
        return " ".join(line.split())[:160]

    return _swallow(_role, "")


def _friendly_names(profile_dir: Path) -> tuple[str, str]:
    """(Bot Mode title, profile.yaml ``display_name``) for a profile, "" when unset. Never raises."""
    def _read() -> tuple[str, str]:
        data = _read_yaml_dict(profile_dir / "profile.yaml") or {}
        return (str((_bots_meta(data) or {}).get("title") or "").strip(),
                str(data.get("display_name") or "").strip())

    return _swallow(_read, ("", ""))


def _display_name(name: str, profile_dir: Path) -> str:
    """Human-facing sender name, in the Desktop's ``botFriendlyNames`` order: Bot Mode title,
    then profile.yaml ``display_name`` (``hermes profile rename``), else the @handle — the
    renamed primary signs as ``Maia (@hermes)``, not ``hermes (@hermes)`` (#89720)."""
    return next((n for n in _friendly_names(profile_dir) if n), None) or _handle(name)


# Tokens the Desktop mention parser reserves; a bot titled "Hermes" never hijacks @hermes.
_RESERVED_ALIASES = frozenset({"all", "everyone", "user", "default", "hermes"})


def alias_forms(value: str) -> set[str]:
    """Lower-cased mention forms of a friendly name, mirroring the Desktop's
    ``mentionNameForms``: slugified (``"Dr. Foo"`` → ``dr-foo``, what autocomplete inserts)
    and collapsed (``drfoo``). Reserved tokens and empty forms are dropped."""
    name = str(value or "").strip().lower()
    slug = re.sub(r"[^a-z0-9_-]+", "-", name).strip("-")
    collapsed = re.sub(r"[^a-z0-9_-]+", "", name)
    return {f for f in (slug, collapsed)
            if f and re.fullmatch(r"[a-z0-9][a-z0-9_-]*", f) and f not in _RESERVED_ALIASES}


def local_alias_map(root: Path) -> dict[str, set[str]]:
    """``alias form → {folder ids}`` for every local profile's friendly names (profile.yaml
    ``display_name`` and the Bot Mode title). Folder ids themselves are not aliases: the
    caller matches those first, so a target that is an exact folder id always addresses that
    folder — a friendly name colliding with ANOTHER folder id never steals it. Ambiguity
    (one alias form shared by several profiles) surfaces as a multi-id set. Never raises."""
    def _build() -> dict[str, set[str]]:
        aliases: dict[str, set[str]] = {}
        for name, profile_dir in _roster(root):
            for form in set().union(*(alias_forms(f) for f in _friendly_names(profile_dir))):
                aliases.setdefault(form, set()).add(name)
        return aliases

    return _swallow(_build, {})


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


def local_taken_forms(root: Path) -> set[str]:
    """Bare forms this gateway's own profiles answer to (handles + friendly-name slugs); a remote
    row must not be offered under any of them, since local resolution wins (``_resolve_local_name``)."""
    return {_handle(name) for name, _d in _roster(root)} | set(local_alias_map(root))


def _remote_paragraph(root: Path, *, telegram: bool = False) -> str:
    """Addendum for agents on OTHER connected machines; only when the relay roster is non-empty."""
    roster = _remote_roster(root)
    if not roster:
        return ""
    from tools.bot_relay import remote_target_forms

    lines = [
        _bullet(
            f"{_sigil(telegram)}{form}",
            f"on {row['connection_label'] or row['connection_id']}",
            row["title"],
            row["description"],
        )
        for row, form in zip(roster, remote_target_forms(roster, local_taken_forms(root)))
    ]
    return (
        "\n\nTeammates on OTHER connected machines (reachable through the "
        "Desktop relay — message them with message_agent exactly like local "
        "teammates; replies arrive as completion notifications the same "
        "way, or via reply_delivery=\"poll\" as below):\n" + "\n".join(lines)
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

    allowed = allowed_local_profile_names(home)
    roster_lines = [
        _bullet(f"@{_handle(name)}", _profile_role(directory))
        for name, directory in _roster(root)
        if name in allowed
    ]
    roster_block = "\n".join(roster_lines) or "- (no teammates yet)"

    return (
        f"{_PROTOCOL_HEADING}\n"
        "This install runs Bot Mode: each Hermes profile is an agent teammate with "
        'one canonical "Bot Chat" conversation, and you have the `message_agent` '
        "tool to DM any of them. It is FIRE-AND-FORGET: it delivers your message "
        "with your attribution prefixed automatically and returns an acknowledgement "
        "immediately — it never returns the reply. Send it, finish your turn, and "
        "the reply arrives later as a background-process completion notification "
        "that wakes you; relay it to the user then, attributed to that agent — unless "
        "the ack returns reply_delivery=\"poll\", in which case follow its "
        "process(action=\"wait\") instruction before ending the turn. "
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


def _roster_render_fingerprint(home: Path) -> str:
    """Live policy key for new prompt renders; deliberately not a capability epoch."""
    import hashlib
    import json

    root = _hermes_root(home)
    surface = {
        "policy": _bot_mode_config(root),
        "targets": allowed_local_profile_names(home),
        "sender_enabled": _is_bot_enabled(home),
    }
    return _swallow(
        lambda: hashlib.sha256(
            json.dumps(surface, sort_keys=True, default=str).encode("utf-8"),
        ).hexdigest()[:12],
        "unavailable",
    )


def get_bot_mode_protocol_section(home: str | os.PathLike | None = None, *, force_refresh: bool = False) -> str:
    """Cached probe entry point — one filesystem pass per (process, home). ``home`` should be
    the AGENT'S OWN resolved home (session-db derived), not ambient HERMES_HOME — build threads
    can lose the ContextVar override and the env var would then name the wrong profile."""
    resolved = str(_resolve_home(home))
    epoch = f"{capability_fingerprint(resolved)}:{_roster_render_fingerprint(Path(resolved))}"
    with _lock:
        cached = _cached.get(resolved)
        if force_refresh or cached is None or cached[0] != epoch:
            cached = (epoch, _swallow(lambda: _build_section(Path(resolved)), ""))
            _cached[resolved] = cached
        return cached[1]


def _sigil(telegram: bool) -> str:
    """Use an inert profile marker where ``@word`` addresses a real account."""
    return "$" if telegram else "@"


def _soul_has_user_protocol(profile_dir: Path) -> bool:
    try:
        soul = profile_dir / "SOUL.md"
        return soul.is_file() and _USER_PROTOCOL_HEADING in soul.read_text(
            encoding="utf-8", errors="replace",
        )
    except Exception:
        return False


def _build_user_surface_section(home: Path, platform: str = "") -> str:
    """Protocol for a trusted human messaging session."""
    telegram = str(platform or "").strip().lower() == "telegram"
    root, me = _hermes_root(home), _profile_name(home)
    profile_dir = home if me == "default" else root / "profiles" / me
    if _soul_has_user_protocol(profile_dir):
        return ""
    sigil = _sigil(telegram)
    roster_lines = [
        _bullet(f"{sigil}{_handle(name)}", _profile_role(directory))
        for name, directory in _roster(root)
        if name in allowed_local_profile_names(home)
    ]
    roster_block = "\n".join(roster_lines) or "- (no local teammates allowed)"
    telegram_rule = (
        " This chat is on Telegram: `$profile` is the internal message_agent target. "
        "Use an actual Telegram @username only when the channel context lists it as a "
        "same-group peer, and never both call message_agent and visibly mention that peer "
        "for the same handoff."
        if telegram else ""
    )
    return (
        f"{_USER_PROTOCOL_HEADING}\n"
        "This messaging session can use Bot Mode. Named Hermes profiles are agent "
        "teammates and `message_agent` is the reliable handoff path. It returns a "
        "dispatch acknowledgement, then the delivery outcome arrives through the "
        "session's completion path. Compose the message yourself, include the concrete "
        "request and relevant context, and contact one relevant teammate unless the user "
        f"explicitly asks for fan-out.{telegram_rule}\n"
        f"You are `{sigil}{_handle(me)}`. Allowed teammates:\n"
        + roster_block
        + _remote_paragraph(root, telegram=telegram)
        + _peer_paragraph(root)
    )


def get_bot_mode_user_protocol_section(
    home: str | os.PathLike | None = None, *, force_refresh: bool = False, platform: str = "",
) -> str:
    """Cached messaging-session protocol, separated by Telegram addressing semantics."""
    resolved = str(_resolve_home(home))
    cache_key = f"{resolved}\ttelegram" if str(platform).strip().lower() == "telegram" else resolved
    epoch = f"{capability_fingerprint(resolved)}:{_roster_render_fingerprint(Path(resolved))}"
    with _lock:
        cached = _user_surface_cached.get(cache_key)
        if force_refresh or cached is None or cached[0] != epoch:
            cached = (
                epoch,
                _swallow(
                    lambda: _build_user_surface_section(Path(resolved), platform=platform),
                    "",
                ),
            )
            _user_surface_cached[cache_key] = cached
        return cached[1]


def invalidate_bot_mode_prompt_cache(
    home: str | os.PathLike | None = None,
) -> None:
    """Drop rendered roster text after a capability epoch change.

    Session authorization stays frozen; only the rebuilt prompt's live presentation changes.
    """
    resolved = str(_resolve_home(home))
    with _lock:
        _cached.pop(resolved, None)
        _user_surface_cached.pop(resolved, None)
        _user_surface_cached.pop(f"{resolved}\ttelegram", None)


# ── capability epoch ─────────────────────────────────────────────────────────
# Bot Chat sessions are effectively eternal, so "build the prompt once" would strand
# capability changes (skills, toolsets, MCP, SOUL, roster, peers, model capability
# overrides that change the prompt) forever. The fingerprint hashes exactly that
# surface; the built prompt embeds it and agent/conversation_loop.py rebuilds only
# when the stored epoch differs from disk — once per change, never per-turn drift.

_EPOCH_PREFIX = "Capability epoch: "
_EPOCH_RE_TEXT = r"Capability epoch: ([0-9a-f]{12})"


def _model_prompt_capability_surface(model_cfg: object) -> dict:
    """``model.*`` overrides whose flip changes a rebuilt prompt, coerced like their consumers.

    ``supports_vision`` uses image routing's strict bool so YAML ``yes`` and ``true`` share
    one epoch. ``context_length`` is the cap ``build_system_prompt_parts`` uses to truncate
    context files. Routing keys (provider, default, base_url) are identity lines, not this
    surface — and no model id is special-cased.
    """
    from agent.image_routing import _coerce_capability_bool

    if not isinstance(model_cfg, dict):
        model_cfg = {}
    raw_ctx = model_cfg.get("context_length")
    ctx = None
    # bool is an int subclass; ``context_length: true`` is not a window.
    if isinstance(raw_ctx, bool):
        ctx = None
    elif isinstance(raw_ctx, int):
        ctx = raw_ctx if raw_ctx > 0 else None
    elif isinstance(raw_ctx, str) and raw_ctx.strip().isdigit():
        parsed = int(raw_ctx.strip())
        ctx = parsed if parsed > 0 else None
    return {
        "supports_vision": _coerce_capability_bool(model_cfg.get("supports_vision")),
        "context_length": ctx,
    }


def capability_fingerprint(home: str | os.PathLike | None = None) -> str:
    """12-hex digest of the capability surface for ``home``'s profile: disabled skills +
    enabled toolsets + MCP config, model capability overrides that change the prompt
    (``supports_vision``, ``context_length``), SOUL.md bytes, installed skill names, the
    Bot-Mode roster (+ roles), peers and the relay roster. Deliberately NOT cached — the
    point is detecting on-disk drift against a stored prompt's epoch. Never raises
    ("unavailable" on failure)."""
    import hashlib
    import json

    resolved = _resolve_home(home)
    root = _hermes_root(resolved)
    surface: dict = {}
    try:
        # Canonical loader (managed overlay + env expansion + normalization),
        # scoped to the bot's home via the override the loaders already honor.
        from agent.skill_utils import parse_config_string_list
        from hermes_cli.config import load_config_readonly
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        token = set_hermes_home_override(str(resolved))
        try:
            cfg = load_config_readonly() or {}
        finally:
            reset_hermes_home_override(token)
        skills_cfg = cfg.get("skills") if isinstance(cfg.get("skills"), dict) else {}
        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        surface["model_capabilities"] = _model_prompt_capability_surface(model_cfg)
        surface["disabled_skills"] = sorted(str(s).lower() for s in (skills_cfg.get("disabled") or []))
        # The live selection is platform_toolsets.<platform> minus agent.disabled_toolsets;
        # tools.enabled_toolsets is written by no surface, so watching it left Bot Chats
        # blind to `hermes tools enable/disable` (#124211). Raw slices: an edit that leaves
        # the effective selection unchanged costs one spurious rebuild at most.
        agent_cfg = cfg.get("agent") if isinstance(cfg.get("agent"), dict) else {}
        surface["platform_toolsets"] = json.dumps(cfg.get("platform_toolsets") or {}, sort_keys=True, default=str)
        surface["disabled_toolsets"] = sorted(parse_config_string_list(agent_cfg.get("disabled_toolsets")))
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
        # The same walk every other reader of this tree uses (skills_list/skill_view, the prompt's
        # skills index, skill_count): it prunes EXCLUDED_SKILL_DIRS — ``.archive``, ``.curator_backups``,
        # ``node_modules`` … — and each skill's support dirs. A raw ``**/SKILL.md`` glob counted files
        # the model can never invoke, so archiving a skill, or the curator writing a backup, flipped the
        # epoch and forced every Bot Chat to rebuild a system prompt whose skills index had not changed.
        # iter_skill_index_files is also org-token-gated, so the epoch moves on an org switch as well —
        # intended: a different org sees a different skills index, so it needs a different prompt.
        from agent.skill_utils import iter_skill_index_files

        return sorted(str(p.parent.relative_to(skills_root))
                      for p in iter_skill_index_files(skills_root, "SKILL.md"))

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
    surface["protocol_version"] = 3
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
    """Whether an authorized session's prompt predates the capability epoch."""
    if _EPOCH_PREFIX in (stored_prompt or ""):
        return False
    return _swallow(lambda: bool(get_bot_mode_protocol_section(home)), False)


def _reset_cache_for_tests() -> None:
    with _lock:
        _cached.clear()
        _user_surface_cached.clear()
    with _session_state_lock:
        _session_state_cache.clear()
