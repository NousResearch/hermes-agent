"""Skin + config-change watcher: on-disk signatures for skin/pet/cron/sessions/platforms/
pairing/bot-relay state and the broadcast loop that pushes *.changed events. Bodies are
rebound onto server.py's globals at install time (method_ctx.bind_module)."""

from __future__ import annotations

import itertools
from pathlib import Path

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


def resolve_skin() -> dict:
    return skin_payload(_load_cfg())


def skin_payload(config: dict) -> dict:
    """The ``skin.changed`` / ``gateway.ready`` skin for ``config``'s ``display.skin`` (``{}`` when the
    skin engine fails). Pure of server state: the shared owner calls it under the profile's scope."""
    try:
        from hermes_cli.skin_engine import init_skin_from_config, get_active_skin
        init_skin_from_config(config)
        skin = get_active_skin()
        # light/dark are paired palettes: the TUI prefers the block matching terminal polarity.
        return {
            "name": skin.name, "colors": skin.colors,
            "light_colors": skin.light_colors, "dark_colors": skin.dark_colors,
            "branding": skin.branding, "banner_logo": skin.banner_logo,
            "banner_hero": skin.banner_hero, "tool_prefix": skin.tool_prefix,
            "help_header": (skin.branding or {}).get("help_header", ""),
            # Raw user CSS for the desktop GUI's <style> tag (32 KiB cap in the skin engine).
            "customCSS": skin.custom_css}
    except Exception:  # health: allow BLE001 -- moved unchanged from resolve_skin (main keeps the same handler): a broken skin file boots the default theme
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


def _files_sig(paths) -> tuple:
    """Sorted ``(path, mtime_ns, size)`` per stat-able file in ``paths``.

    For multi-file signatures, unlike a newest-mtime max: a file added, removed or rewritten in
    the same timestamp tick as the current newest one (coarse-mtime filesystems) still moves it.
    """
    entries = []
    for path in paths:
        with contextlib.suppress(OSError):
            st = path.stat()
            entries.append((str(path), st.st_mtime_ns, st.st_size))
    return tuple(sorted(entries))


def _skin_sig() -> tuple[str, float | None]:
    """(active skin name, its user-file mtime). Built-ins have no file, so only
    their name moves; a user skin's mtime lets an in-place color edit repaint too."""
    name = str((_load_cfg().get("display") or {}).get("skin") or "default")
    try:
        return name, (_watcher_home() / "skins" / f"{name}.yaml").stat().st_mtime
    except OSError:
        return name, None


# A multiplexed gateway serves other profiles' TUIs from this one watcher thread (no profile
# override): their baselines live here, keyed by home, and their repaints go to their own peers only
# (main ran one gateway per profile, so a work TUI never saw the launch home's skin).
_skin_sig_by_home: dict[str, tuple[str, float | None]] = {}


def _skin_home_for(profile_id) -> str | None:
    """The profile home a peer's skin follows, or None for the launch home (the ambient scope)."""
    with contextlib.suppress(Exception):
        home = Path(str(profile_id))
        if home.is_absolute() and home.is_dir() and home.resolve() != Path(_hermes_home).resolve():
            return str(home)
    return None


def _under_home(home: str, fn):
    token = set_hermes_home_override(home)
    try:
        return fn()
    finally:
        reset_hermes_home_override(token)


def _note_skin_broadcast() -> None:
    """Sync the baseline after the /skin RPC emits so the watcher doesn't re-broadcast it."""
    global _last_skin_sig
    with contextlib.suppress(Exception):
        if home := _skin_home_for(get_hermes_home_override()):
            _skin_sig_by_home[home] = _skin_sig()
        else:
            _last_skin_sig = _skin_sig()


def _skin_peers_by_home() -> dict[str, list]:
    with _live_transports_lock:
        targets = list(_live_transports)
    peers: dict[str, list] = {}
    for transport in targets:
        if home := getattr(transport, "skin_home", None):
            peers.setdefault(home, []).append(transport)
    return peers


def _broadcast_skin_if_changed() -> None:
    """Emit ``skin.changed`` when the active skin moved, via the SAME live path as
    ``/skin`` so every surface repaints. The check is a dict lookup + one stat per home."""
    global _last_skin_sig
    by_home = _skin_peers_by_home()
    with contextlib.suppress(Exception):
        sig = _skin_sig()
        if sig != _last_skin_sig:
            _last_skin_sig = sig
            _broadcast_global_event("skin.changed", resolve_skin(),
                                    skip={peer for peers in by_home.values() for peer in peers})
    for home, peers in by_home.items():
        with contextlib.suppress(Exception):
            sig = _under_home(home, _skin_sig)
            seen = _skin_sig_by_home.get(home)
            _skin_sig_by_home[home] = sig
            if seen is None or sig == seen:
                continue  # first sight seeds: the attach's runtime.describe already carried this skin
            frame = _event_frame("skin.changed", "", _under_home(home, resolve_skin))
            for peer in peers:
                with contextlib.suppress(Exception):
                    peer.write(frame)


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
    except Exception:
        pass
    return ("off",)


def _pet_changed_payload() -> dict:
    """``pet.info.meta``-shaped payload so the renderer can decide whether to refetch sprites."""
    try:
        if active := _active_pet():
            pet, scale = active
            return {"enabled": True, "slug": pet.slug, "displayName": pet.display_name,
                    "scale": scale, "spritesheetRevision": _pet_sheet_revision(pet.spritesheet)}
    except Exception:
        pass
    return {"enabled": False}


# last_activity_at / _description are left out on purpose: the session activity heartbeat restamps
# them mid-turn and on idle ticks, so hashing them re-fired sessions.changed every heartbeat window
# (#98005). Liveness comes from session.active_list, and a real turn still moves message_count.
_SESSION_SIGNATURE_FIELDS = (
    "id", "source", "session_key", "display_name", "model", "parent_session_id",
    "started_at", "ended_at", "end_reason", "message_count", "tool_call_count",
    "cwd", "git_branch", "git_repo_root", "title", "title_source", "profile_name",
    "archived", "pinned", "hidden", "last_read_at", "handoff_state",
)
# path -> (database/WAL mtime, session-table digest, whether that is a fallback rather than a
# read answer). The mtime guard keeps the normal 0.5 s watch pass stat-only; SQLite is read only
# after another process actually commits (and for ``utils.RACY_MTIME_WINDOW_NS`` after it, see
# _settled_mtime).
_sessions_db_sig_cache: dict[str, tuple[object, tuple | None, bool]] = {}


_MTIME_FALLBACK = "mtime-fallback"


def _db_content_sig(db_path: Path, cache: dict, read, *, fallback=lambda mtime: None):
    """``read(conn, mtime)`` over a read-only connection, cached behind the DB/WAL mtime so a watch pass
    stays stat-only until another process commits. ``read`` returns ``(signature, is_fallback)``. A failed
    read after a good answer keeps it and leaves the cached mtime stale so the next pass re-reads (answer ->
    fallback -> answer would broadcast twice); otherwise the answer is ``fallback(mtime)``."""
    mtime = _newest_mtime_ns((db_path, db_path.with_name(f"{db_path.name}-wal")))
    cache_key = str(db_path)
    cached = cache.get(cache_key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    if not db_path.exists():
        cache[cache_key] = (mtime, None, True)
        return None
    conn = None
    try:
        from hermes_state import _connect_tracked_db
        from hermes_state_holders import read_only_db_uri
        conn = _connect_tracked_db(read_only_db_uri(db_path), tracking_path=db_path, uri=True, timeout=0.05)
        signature, is_fallback = read(conn, mtime)
    except Exception:  # health: allow BLE001 -- any failed read (busy, locked, foreign schema) falls back like a busy one
        if cached is not None and not cached[2]:
            return cached[1]
        signature, is_fallback = fallback(mtime), True
    finally:
        if conn is not None:
            conn.close()
    cache[cache_key] = (_settled_mtime(mtime), signature, is_fallback)
    return signature


def _session_rows_sig(conn, mtime):
    import hashlib
    available = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
    fields = tuple(field for field in _SESSION_SIGNATURE_FIELDS if field in available)
    if not fields:
        # A state.db without a readable sessions table (foreign schema, legacy or transient file): keep
        # the old mtime contract so any move still wakes the sidebar instead of silently never broadcasting.
        return (_MTIME_FALLBACK, mtime), True
    order = " ORDER BY id" if "id" in available else ""
    digest = hashlib.blake2b(digest_size=16)
    ids = set()
    id_index = fields.index("id") if "id" in fields else None
    for row in conn.execute(f"SELECT {', '.join(fields)} FROM sessions{order}"):
        if id_index is not None:
            ids.add(row[id_index])
        digest.update(repr(tuple(row)).encode("utf-8", "backslashreplace"))
        digest.update(b"\0")
    return (fields, digest.digest(), frozenset(ids)), False


def _session_db_content_sig(db_path: Path):
    """Digest list/transcript-relevant session rows, excluding unrelated tables.

    ``gateway_heartbeats`` shares state.db and writes every minute. Using the
    database mtime directly therefore emits sessions.changed while no session
    changed (#98005). Cache behind the DB/WAL mtime, then inspect only the
    sessions columns that drive Desktop projections. Legacy stores safely use
    the subset of columns they have. A failed read after a good digest keeps it:
    digest -> mtime -> digest would broadcast twice. A freshly computed digest
    also carries session IDs so a committed disappearance can emit
    ``sessions.deleted`` without waiting for the generic refresh floor.
    """
    cached = _sessions_db_sig_cache.get(str(db_path))
    previous = cached[1] if cached is not None else None
    signature = _db_content_sig(
        db_path, _sessions_db_sig_cache, _session_rows_sig,
        fallback=lambda mtime: (_MTIME_FALLBACK, mtime))
    # Cache hits and locked reads return the same signature object. Only two
    # readable ID snapshots can prove disappearance; None and fallbacks cannot.
    if isinstance(previous, tuple) and isinstance(signature, tuple) and signature is not previous:
        _broadcast_session_deletions(db_path.parent, previous, signature)
    return signature


def _broadcast_session_deletions(home: Path, previous, current) -> None:
    """Only two readable ID snapshots prove deletion; never a missing/locked store.

    Publish independently of the generic refresh floor: Desktop can evict exact
    IDs without interrupting typing or rehydrating a live transcript.
    """
    if not isinstance(previous, tuple) or not isinstance(current, tuple):
        return
    if len(previous) != 3 or len(current) != 3:
        return
    if "id" not in previous[0] or "id" not in current[0]:
        return
    deleted = sorted(previous[2] - current[2])
    if deleted:
        profile = home.name if home.parent.name == "profiles" else "default"
        _broadcast_global_event("sessions.deleted", {"session_ids": deleted, "profile": profile})


# Filesystem timestamps come from a coarse clock (~4 ms ticks; whole seconds on some filesystems) and
# SQLite commits in place, so a commit landing in the tick of a read leaves the DB/WAL mtime unchanged.
# A digest read while the mtime is that fresh must not become a cache hit ("racily clean", as in git's
# index): it is cached under a key no mtime equals, so the next tick re-reads the rows.
_RACY_MTIME = object()


def _settled_mtime(mtime):
    """``mtime`` when it is old enough to trust as a cache key, else a never-matching sentinel."""
    from utils import mtime_is_racy  # function-local: bodies are rebound onto server.py's globals
    if mtime is None or mtime_is_racy(mtime):
        return _RACY_MTIME
    return mtime


def _watched_homes() -> tuple[Path, ...]:
    """The watcher home and every sibling profile home this gateway serves."""
    return (_watcher_home(), *_served_profile_homes)


def _sessions_sig():
    """Session-table content across the active and served profile stores.

    Messaging-gateway turns and cron runs are written by other processes that
    never touch this gateway's transports, so their session rows are the shared
    change signal. Hashing only those rows avoids false Desktop refreshes from
    unrelated state.db writes such as gateway heartbeats.
    """
    return tuple(
        _session_db_content_sig(root / "state.db")
        for root in _watched_homes()
    )


def _projects_sig():
    """Per-file stat signature of projects.db (+ WAL) for the watcher home and every served
    sibling profile. The CLI and other windows write projects.db directly — nothing in
    their process touches this gateway's transports — so the file is the only shared
    signal, exactly like state.db (#53046, #56757)."""
    return _files_sig(
        root / name
        for root in _watched_homes()
        for name in ("projects.db", "projects.db-wal"))


def _pairing_sig():
    """Per-file stat signature of every profile's pairing ledgers (legacy ``pairing/`` and
    ``platforms/pairing/``): the gateway process writes pending codes, so the files are the only
    shared signal (a pairing request moves nothing in gateway_state.json)."""
    entries = []
    for root in _pairing_roots(_watcher_home()):
        with contextlib.suppress(OSError):
            # Only the ledgers: _rate_limits.json moves on every unauthorized DM.
            entries += [
                e for e in root.iterdir() if e.name.endswith(("-pending.json", "-approved.json"))]
    return _files_sig(entries)


# Live-profile pairing roots, cached on (home, ``profiles/`` dir mtime) with a TTL. The liveness
# probe costs ~14 stats per profile; on the 2 s tick that was the watcher's share of the idle
# profile-tree burn (#114041 §2/§3). The profile SET only moves when a dir is added or removed —
# which bumps the parent's mtime — while a marker/tombstone landing inside one is caught by the TTL.
_PAIRING_ROOTS_TTL_S = 30.0
_pairing_roots_cache: tuple[Path, int | None, float, list] | None = None


def _pairing_roots(home: Path) -> list:
    global _pairing_roots_cache
    profiles_dir = home / "profiles"
    dir_mtime, now = _watcher_mtime_ns(profiles_dir), time.monotonic()
    cached = _pairing_roots_cache
    if cached is not None and cached[0] == home and cached[1] == dir_mtime and now - cached[2] < _PAIRING_ROOTS_TTL_S:
        return cached[3]
    from hermes_constants import named_profile_is_live
    roots = [home / "pairing", home / "platforms" / "pairing"]
    with contextlib.suppress(OSError):
        for profile_dir in profiles_dir.iterdir():
            if named_profile_is_live(profile_dir):
                roots += [profile_dir / "pairing", profile_dir / "platforms" / "pairing"]
    _pairing_roots_cache = (home, dir_mtime, now, roots)
    return roots


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


def _room_rows_sig(conn, _mtime) -> dict:
    """Per live room in one store: what ``groups.log`` and ``groups.state`` read from it."""
    from gateway.hosted_room_driver import room_change_rows
    return room_change_rows(conn), False


_room_store_sig_cache: dict[str, tuple[object, dict | None, bool]] = {}


def _groups_sig() -> dict:
    """``{room_id: signature}`` across every room store this process serves (each room service's store,
    each served home's session store, the install's shared room store) plus the status only this
    process holds: each service's ``live_status_sig`` (approvals, route states, stalls, blocked rooms) and
    member progress (``live_progress``: tool, live confirmation, progress), signed once. Key ``""`` is
    process-wide (a worker started or stopped). Cron, the CLI and other gateways append to
    these stores from their own processes, so the files are the shared signal."""
    from gateway.hosted_rooms import default_db_path
    from tui_gateway.hosted_room_service import live_room_services
    services = live_room_services()
    stores = {Path(path).resolve(): None for path in (
        *(root / "state.db" for root in _watched_homes()), default_db_path(),
        *(service.db_path for service in services))}
    rooms: dict[str, list] = {}
    for store in stores:
        for room_id, part in (_db_content_sig(store, _room_store_sig_cache, _room_rows_sig) or {}).items():
            rooms.setdefault(room_id, []).append((str(store), part))
    for service in services:
        with contextlib.suppress(Exception):  # health: allow BLE001 -- one broken service must not blind the rest
            for room_id, part in service.live_status_sig().items():
                rooms.setdefault(room_id, []).append(("live", part))
    # Member progress is one process-wide map that every service's status reads: signed once.
    from tui_gateway.hosted_room_member_activity import live_progress
    for room_id, *progress in live_progress():
        rooms.setdefault(room_id, []).append(("progress", *progress))
    # Each part starts with its kind and a key unique within the room (store path, "live", "progress" + task):
    # plain tuple order, no repr per part per tick.
    return {room_id: tuple(sorted(parts)) for room_id, parts in rooms.items()}


_groups_woken_heads: dict[str, dict] = {}


def _wake_room_drivers() -> None:
    """Wake the room service whose store holds a room whose log grew, so a post another process appended
    (a cron job, the CLI) is answered now instead of at the driver's 5 s idle poll. Only ``next_seq``
    counts: task and status moves are the driver's own. Its own appends wake it once more (one empty
    pass, at most one per watch tick); ``wakeup`` only sets an event. Reads the store signatures the
    groups watch already cached: no store access of its own."""
    from tui_gateway.hosted_room_service import live_room_services
    heads = {store: {room_id: part.next_seq for room_id, part in rows.items()}
             for store, (_mtime, rows, _fallback) in _room_store_sig_cache.items() if rows}
    moved = {store for store, rooms in heads.items() if _groups_woken_heads.get(store, rooms) != rooms}
    _groups_woken_heads.clear()
    _groups_woken_heads.update(heads)
    for service in live_room_services():
        with contextlib.suppress(Exception):  # health: allow BLE001 -- a stopped service must not stop the watcher
            if str(Path(service.db_path).resolve()) in moved:
                service.wakeup()


# One ``groups.changed`` frame goes to every socket a window holds open to this process; its id (this process's
# replay epoch, the token gateway.ready already carries, + a counter) tells a view that a copy is one it already
# read for, and nothing else.
_groups_broadcast_count = itertools.count(1)


def _groups_changed_payload(before: dict, after: dict) -> dict:
    """``{"room_ids": [...]}``: the rooms whose log or status moved since the last broadcast, so an open
    view of another room does not re-read. No ``room_ids`` (every view refreshes) when process-wide state
    moved. ``broadcast_id`` names this one broadcast, unique per process lifetime."""
    moved = {room_id for room_id in before.keys() | after.keys() if before.get(room_id) != after.get(room_id)}
    scope = {} if "" in moved else {"room_ids": sorted(moved)}
    from tui_gateway.event_replay import replay_epoch
    return {**scope, "broadcast_id": f"{replay_epoch()}:{next(_groups_broadcast_count)}"}


def _skills_sig():
    """The skill roots' stat signature for every scope a session resolved commands in (each profile
    home and session repo, not only the launch one: ``agent.skill_commands.published_scopes_signature``):
    a skill installed or removed from another terminal (``hermes skills install``, a dropped SKILL.md,
    nested under a category) moves it, and attached clients refetch ``commands.catalog`` from this
    gateway instead of rescanning disk."""
    from agent.skill_commands import published_scopes_signature
    return published_scopes_signature()


# event → (check interval, signature fn, payload fn). Signatures are stat-cheap; the interval
# keeps pricier probes (pet resolves the sheet off disk) off the 0.5s tick. cron/jobs.json
# moves on edits AND scheduler ticks; gateway_state.json is where the messaging gateway
# persists platform connect/disconnect/health (the Messaging page's status signal).
_CHANGE_WATCHES: dict[str, tuple[float, Any, Any] | tuple[float, Any, Any, Any]] = {
    "pet.changed": (2.0, _pet_sig, _pet_changed_payload),
    "cron.changed": (1.0, lambda: _home_mtime_ns("cron", "jobs.json"), dict),
    "sessions.changed": (0.5, _sessions_sig, dict),
    # Projects created/switched by CLI or agent tooling write projects.db without any
    # state.db movement, so sessions.changed never fires and the desktop Projects
    # sidebar goes stale until a manual refresh (#56757).
    "platforms.changed": (2.0, lambda: _home_mtime_ns("gateway_state.json"), dict),
    "projects.changed": (2.0, _projects_sig, dict),
    "pairing.changed": (2.0, _pairing_sig, dict),
    "skills.changed": (2.0, _skills_sig, dict),
    # 1s so a queued DM envelope reaches the Desktop's push-triggered drain fast.
    "bot_relay.outbox.pending": (1.0, _bot_relay_outbox_sig, dict),
    # An open room view re-reads its log delta + status on this; its slow poll is only the backstop.
    # Its payload names what moved: an optional 4th slot, ``fn(previous broadcast signature, current one)``.
    "groups.changed": (0.5, _groups_sig, dict, _groups_changed_payload)}

# state.db moves on every append of a streaming turn and gateway_state.json on
# in-flight bookkeeping; the floor coalesces bursts to one broadcast per window,
# trailing edge included (a floored change keeps its old signature, re-fires later).
_CHANGE_BROADCAST_FLOOR_S = {"sessions.changed": 2.0, "platforms.changed": 5.0, "groups.changed": 1.0}

_change_sigs: dict[str, Any] = {}
_change_checked_at: dict[str, float] = {}
_change_broadcast_at: dict[str, float] = {}


def _broadcast_watched_changes(now: float | None = None) -> None:
    """One pass: recompute due signatures, broadcast events whose signature moved.
    First sighting seeds silently so a gateway boot never fires a refresh storm."""
    now = time.monotonic() if now is None else now
    for event, (interval, sig_fn, payload_fn, *diff_fn) in _CHANGE_WATCHES.items():
        if now - _change_checked_at.get(event, -interval) < interval:
            continue
        _change_checked_at[event] = now
        try:
            sig = sig_fn()
        except Exception:
            continue
        if event not in _change_sigs:
            _change_sigs[event] = sig
            continue
        floor = _CHANGE_BROADCAST_FLOOR_S.get(event, 0.0)
        if sig == _change_sigs[event]:
            continue
        if floor and now - _change_broadcast_at.get(event, -floor) < floor:
            continue  # floored: old signature stays so it re-fires when the window opens
        before, _change_sigs[event] = _change_sigs[event], sig
        _change_broadcast_at[event] = now
        with contextlib.suppress(Exception):
            _broadcast_global_event(event, diff_fn[0](before, sig) if diff_fn else payload_fn())


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
            # A failed wake must never end this thread: every push (skin, sessions, groups...) runs on it.
            with contextlib.suppress(Exception):  # health: allow BLE001 -- the driver's own idle poll still runs
                _wake_room_drivers()
    threading.Thread(target=_loop, name="hermes-change-watcher", daemon=True).start()


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
