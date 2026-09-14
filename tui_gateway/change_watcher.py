"""Skin + config-change watcher: on-disk signatures for skin/pet/cron/sessions/platforms/
pairing/bot-relay state and the broadcast loop that pushes *.changed events. Bodies are
rebound onto server.py's globals at install time (method_ctx.bind_module)."""

from __future__ import annotations

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


def resolve_skin() -> dict:
    try:
        from hermes_cli.skin_engine import init_skin_from_config, get_active_skin
        init_skin_from_config(_load_cfg())
        skin = get_active_skin()
        # light/dark are paired palettes: the TUI prefers the block matching terminal polarity.
        return {
            "name": skin.name, "colors": skin.colors,
            "light_colors": skin.light_colors, "dark_colors": skin.dark_colors,
            "branding": skin.branding, "banner_logo": skin.banner_logo,
            "banner_hero": skin.banner_hero, "tool_prefix": skin.tool_prefix,
            "help_header": (skin.branding or {}).get("help_header", "")}
    except Exception:
        return {}


# (name, user-file mtime) of the last skin broadcast: ``skin.changed`` fires on a name
# switch OR a live color edit of the active skin, and nothing else.
_last_skin_sig: tuple[str, float | None] | None = None


def _watcher_home() -> Path:
    """Active profile home for the change watcher's signature probes."""
    override = get_hermes_home_override()
    return Path(override if isinstance(override, str) and override else _hermes_home)


def _watcher_mtime_ns(path: Path):
    """``st_mtime_ns`` of ``path``, or None when it cannot be stat'ed."""
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _home_mtime_ns(*parts: str):
    return _watcher_mtime_ns(_watcher_home().joinpath(*parts))


def _newest_mtime_ns(paths) -> int | None:
    """Max ``st_mtime_ns`` across ``paths`` (unstat-able ignored); None when none stat'ed."""
    return max((m for m in map(_watcher_mtime_ns, paths) if m is not None), default=None)


def _skin_sig() -> tuple[str, float | None]:
    """(active skin name, its user-file mtime). Built-ins have no file, so only
    their name moves; a user skin's mtime lets an in-place color edit repaint too."""
    name = str((_load_cfg().get("display") or {}).get("skin") or "default")
    try:
        return name, (_watcher_home() / "skins" / f"{name}.yaml").stat().st_mtime
    except OSError:
        return name, None


def _note_skin_broadcast() -> None:
    """Sync the baseline after the /skin RPC emits so the watcher doesn't re-broadcast it."""
    global _last_skin_sig
    with contextlib.suppress(Exception):
        _last_skin_sig = _skin_sig()


def _broadcast_skin_if_changed() -> None:
    """Emit ``skin.changed`` when the active skin moved, via the SAME live path as
    ``/skin`` so every surface repaints. The check is a dict lookup + one stat."""
    global _last_skin_sig
    with contextlib.suppress(Exception):
        sig = _skin_sig()
        if sig == _last_skin_sig:
            return
        _last_skin_sig = sig
        _broadcast_global_event("skin.changed", resolve_skin())


def _active_pet():
    """(pet, scale) when an enabled pet with an existing sheet is selected, else None."""
    enabled, pet, scale = _pet_active_selection()
    return (pet, scale) if enabled and pet is not None and pet.exists else None


def _pet_sig() -> tuple:
    """(slug, spritesheet revision, scale) of the active pet — ("off",) when none."""
    display = _load_cfg().get("display") or {}
    pet_cfg = display.get("pet") if isinstance(display.get("pet"), dict) else {}
    if not pet_cfg or not is_truthy_value(pet_cfg.get("enabled"), default=False):
        return ("off",)
    try:
        if active := _active_pet():
            pet, scale = active
            return (pet.slug, _pet_sheet_revision(pet.spritesheet), scale)
    except Exception:  # noqa: BLE001 - cosmetic, never break the watcher
        pass
    return ("off",)


def _pet_changed_payload() -> dict:
    """``pet.info.meta``-shaped payload so the renderer can decide whether to refetch sprites."""
    try:
        if active := _active_pet():
            pet, scale = active
            return {"enabled": True, "slug": pet.slug, "displayName": pet.display_name,
                    "scale": scale, "spritesheetRevision": _pet_sheet_revision(pet.spritesheet)}
    except Exception:  # noqa: BLE001 - cosmetic, never break the watcher
        pass
    return {"enabled": False}


def _sessions_sig():
    """Newest mtime across state.db + WAL: the one thing messaging-gateway turns and cron runs
    all move. Served sibling profile homes are probed too, else a routed Bot Chat never refreshes.

    signal. Messaging-gateway turns and cron runs are written by OTHER processes that never touch this
    gateway's transports; the shared SQLite file is the one thing they all move (#58671). A backend serving
    several profiles owns one store per profile, so every served sibling home is
    """
    return _newest_mtime_ns(
        root / name
        for root in (_watcher_home(), *_served_profile_homes)
        for name in ("state.db", "state.db-wal"))


def _pairing_sig():
    """Newest mtime across every profile's pairing ledgers (legacy ``pairing/`` and
    ``platforms/pairing/``): the gateway process writes pending codes, so the files are the only
    shared signal (a pairing request moves nothing in gateway_state.json)."""
    home = _watcher_home()
    roots = [home / "pairing", home / "platforms" / "pairing"]
    with contextlib.suppress(OSError):
        for profile_dir in (home / "profiles").iterdir():
            roots += [profile_dir / "pairing", profile_dir / "platforms" / "pairing"]
    entries = []
    for root in roots:
        with contextlib.suppress(OSError):
            # Only the ledgers: _rate_limits.json moves on every unauthorized DM.
            entries += [
                e for e in root.iterdir() if e.name.endswith(("-pending.json", "-approved.json"))]
    return _newest_mtime_ns(entries)


# Newest outbox-envelope mtime EVER seen (monotone): a drain empties the outbox,
# and falling back to None would fire a spurious pending event after every drain.
_bot_relay_outbox_seen = 0


def _bot_relay_outbox_sig():
    """Newest mtime across pending bot-relay outbox envelopes (monotone). Written by the AGENT
    process, so the files are the only shared signal; the Desktop reacts with a debounced drain.

    Envelopes are written by the AGENT process (``message_agent`` → ``tools.bot_relay.enqueue_envelope``) —
    a different process that never touches this gateway's transports — so the files are the only shared
    signal, exactly like the pairing store. See #92760, #93091.
    """
    global _bot_relay_outbox_seen
    home = _watcher_home()
    root = home.parent.parent if home.parent.name == "profiles" else home
    with contextlib.suppress(OSError):
        for entry in (root / "bot_relay" / "outbox").iterdir():
            if entry.name.endswith(".json"):
                _bot_relay_outbox_seen = max(_bot_relay_outbox_seen, _watcher_mtime_ns(entry) or 0)
    return _bot_relay_outbox_seen or None


def _desktop_room_mailbox_sig():
    """Watch messaging commands for classic Desktop rooms across served profiles."""
    home = _watcher_home()
    root = home.parent.parent if home.parent.name == "profiles" else home
    return _watcher_mtime_ns(root / "desktop_room_mailbox.pending")


# event → (check interval, signature fn, payload fn). Signatures are stat-cheap; the interval
# keeps pricier probes (pet resolves the sheet off disk) off the 0.5s tick. cron/jobs.json
# moves on edits AND scheduler ticks; gateway_state.json is where the messaging gateway
# persists platform connect/disconnect/health (the Messaging page's status signal).
_CHANGE_WATCHES: dict[str, tuple[float, Any, Any]] = {
    "pet.changed": (2.0, _pet_sig, _pet_changed_payload),
    "cron.changed": (1.0, lambda: _home_mtime_ns("cron", "jobs.json"), lambda: {}),
    "sessions.changed": (0.5, _sessions_sig, lambda: {}),
    "platforms.changed": (2.0, lambda: _home_mtime_ns("gateway_state.json"), lambda: {}),
    "pairing.changed": (2.0, _pairing_sig, lambda: {}),
    # 1s so a queued DM envelope reaches the Desktop's push-triggered drain fast.
    "bot_relay.outbox.pending": (1.0, _bot_relay_outbox_sig, lambda: {}),
    "desktop_rooms.commands.pending": (1.0, _desktop_room_mailbox_sig, lambda: {})}

# state.db moves on every append of a streaming turn and gateway_state.json on
# in-flight bookkeeping; the floor coalesces bursts to one broadcast per window,
# trailing edge included (a floored change keeps its old signature, re-fires later).
_CHANGE_BROADCAST_FLOOR_S = {"sessions.changed": 2.0, "platforms.changed": 5.0}

_change_sigs: dict[str, Any] = {}
_change_checked_at: dict[str, float] = {}
_change_broadcast_at: dict[str, float] = {}


def _broadcast_watched_changes(now: float | None = None) -> None:
    """One pass: recompute due signatures, broadcast events whose signature moved.
    First sighting seeds silently so a gateway boot never fires a refresh storm."""
    now = time.monotonic() if now is None else now
    for event, (interval, sig_fn, payload_fn) in _CHANGE_WATCHES.items():
        if now - _change_checked_at.get(event, -interval) < interval:
            continue
        _change_checked_at[event] = now
        try:
            sig = sig_fn()
        except Exception:  # noqa: BLE001 - a broken probe must not kill the loop
            continue
        if event not in _change_sigs:
            _change_sigs[event] = sig
            continue
        floor = _CHANGE_BROADCAST_FLOOR_S.get(event, 0.0)
        if sig == _change_sigs[event]:
            continue
        if floor and now - _change_broadcast_at.get(event, -floor) < floor:
            continue  # floored: old signature stays so it re-fires when the window opens
        _change_sigs[event] = sig
        _change_broadcast_at[event] = now
        with contextlib.suppress(Exception):
            _broadcast_global_event(event, payload_fn())


_skin_watcher_started = False


def _ensure_skin_watcher() -> None:
    """Start the process's one change watcher (named for its original skin-only duty): cheap
    on-disk signatures → broadcast events, so changes go live without client polling. Idempotent."""
    global _skin_watcher_started
    if _skin_watcher_started:
        return
    _skin_watcher_started = True
    _note_skin_broadcast()  # seed the baseline so only a real change repaints

    def _loop() -> None:
        while True:
            time.sleep(0.5)
            _broadcast_skin_if_changed()
            _broadcast_watched_changes()
    threading.Thread(target=_loop, name="hermes-change-watcher", daemon=True).start()


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
