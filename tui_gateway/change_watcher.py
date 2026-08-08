"""Change watcher for the tui_gateway server.

Extracted from :mod:`tui_gateway.server` (god-file slice R2-S1, epic #78647 /
#78630). Owns the process's one change watcher: cheap on-disk signature probes
that broadcast ``skin.changed`` / ``pet.changed`` / ``cron.changed`` /
``sessions.changed`` / ``platforms.changed`` / ``pairing.changed`` global
events so a skin Hermes activates, a pet ``/pet`` adopts, a cron the scheduler
fires, or a messaging turn another process writes goes live on every surface
within a couple seconds.

Seam contract
-------------
Shared server state stays in :mod:`tui_gateway.server` and is read through the
module object at call time (patch-inert for the test suite): the registry
(``server._CHANGE_WATCHES``, ``server._CHANGE_BROADCAST_FLOOR_S``,
``server._change_sigs``, ``server._change_checked_at``,
``server._change_broadcast_at``, ``server._last_skin_sig``) plus
``server._hermes_home``, ``server._cfg_cache``, ``server._load_cfg``,
``server._broadcast_global_event``, ``server._pet_active_selection`` and
``server._pet_sheet_revision``. Cluster-mate reads that tests patch on the
server module (``server._skin_sig``, ``server.resolve_skin``) go through the
module object at call time for the same reason.

``server`` is imported at the END of this module (not the top) on purpose:
``tui_gateway.server`` re-exports this module's names at module level, so a
top-of-file ``from tui_gateway import server`` would deadlock the
change_watcher-first import order. Importing at the end keeps both orders
clean: this module's functions only touch ``server`` at call time.
"""

import threading
import time
from pathlib import Path

from hermes_constants import get_hermes_home_override

def resolve_skin() -> dict:
    try:
        from hermes_cli.skin_engine import init_skin_from_config, get_active_skin

        init_skin_from_config(server._load_cfg())
        skin = get_active_skin()
        return {
            "name": skin.name,
            "colors": skin.colors,
            # Paired palettes: the TUI detects the terminal's polarity and
            # prefers the matching hand-tuned block over adapting `colors`.
            "light_colors": skin.light_colors,
            "dark_colors": skin.dark_colors,
            "branding": skin.branding,
            "banner_logo": skin.banner_logo,
            "banner_hero": skin.banner_hero,
            "tool_prefix": skin.tool_prefix,
            "help_header": (skin.branding or {}).get("help_header", ""),
        }
    except Exception:
        return {}


def _skin_sig() -> tuple[str, float | None]:
    """(active skin name, its user-file mtime). Built-ins have no file, so only
    their name moves; a user skin's mtime lets an in-place color edit repaint too."""
    name = str((server._load_cfg().get("display") or {}).get("skin") or "default")
    override = get_hermes_home_override()
    home = override if isinstance(override, str) and override else server._hermes_home
    try:
        mtime: float | None = (Path(home) / "skins" / f"{name}.yaml").stat().st_mtime
    except OSError:
        mtime = None
    return name, mtime


def _note_skin_broadcast() -> None:
    """Sync the reconcile baseline after the /skin RPC emits, so the per-tool
    check doesn't re-broadcast the skin /skin just applied."""
    try:
        server._last_skin_sig = server._skin_sig()
    except Exception:
        pass


def _broadcast_skin_if_changed() -> None:
    """Emit ``skin.changed`` when the active skin moved — the agent switched it
    (``hermes config set display.skin``) OR edited the active skin's colors in
    place ("I don't like that coral" → tweak the YAML).

    Routes through the SAME live path as ``/skin`` so every surface (TUI + desktop)
    repaints, no slash command. The signature check is a dict lookup + one stat,
    so polling it is ~free.
    """
    try:
        sig = server._skin_sig()
    except Exception:
        return
    if sig == server._last_skin_sig:
        return
    server._last_skin_sig = sig
    try:
        server._broadcast_global_event("skin.changed", server.resolve_skin())
    except Exception:
        pass


def _watcher_home() -> Path:
    """Active profile home for the change watcher's signature probes."""
    override = get_hermes_home_override()
    return Path(override if isinstance(override, str) and override else server._hermes_home)


def _pet_sig() -> tuple:
    """(slug, spritesheet revision, scale) of the active pet — ("off",) when none.

    Cheap by construction: config comes from the mtime-cached ``_load_cfg`` and
    the sheet revision is one stat. Moves when ``/pet`` (de)activates a pet, the
    hatch flow rebuilds a sheet, or the scale changes."""
    display = server._load_cfg().get("display") or {}
    pet_cfg = display.get("pet") if isinstance(display.get("pet"), dict) else {}
    if not pet_cfg or not pet_cfg.get("enabled"):
        return ("off",)
    try:
        enabled, pet, scale = server._pet_active_selection()
        if not enabled or pet is None or not pet.exists:
            return ("off",)
        return (pet.slug, server._pet_sheet_revision(pet.spritesheet), scale)
    except Exception:  # noqa: BLE001 - cosmetic, never break the watcher
        return ("off",)


def _pet_changed_payload() -> dict:
    """``pet.info.meta``-shaped payload for ``pet.changed`` — enough for the
    renderer to decide whether the heavy sprite payload needs a refetch."""
    try:
        enabled, pet, scale = server._pet_active_selection()
        if not enabled or pet is None or not pet.exists:
            return {"enabled": False}
        return {
            "enabled": True,
            "slug": pet.slug,
            "displayName": pet.display_name,
            "scale": scale,
            "spritesheetRevision": server._pet_sheet_revision(pet.spritesheet),
        }
    except Exception:  # noqa: BLE001 - cosmetic, never break the watcher
        return {"enabled": False}


def _cron_sig():
    """mtime of the profile's cron/jobs.json — moves on create/edit/pause/
    remove AND on scheduler tick bookkeeping (last_run/next_run)."""
    try:
        return (_watcher_home() / "cron" / "jobs.json").stat().st_mtime_ns
    except OSError:
        return None


def _sessions_sig():
    """Newest mtime across state.db and its WAL — the cross-process change
    signal. Messaging-gateway turns and cron runs are written by OTHER
    processes that never touch this gateway's transports; the shared SQLite
    file is the one thing they all move (#58671)."""
    home = _watcher_home()
    sig = None
    for name in ("state.db", "state.db-wal"):
        try:
            mtime = (home / name).stat().st_mtime_ns
        except OSError:
            continue
        sig = mtime if sig is None else max(sig, mtime)
    return sig


def _platforms_sig():
    """mtime of gateway_state.json — the messaging gateway process persists
    platform connect/disconnect/health there, so its movement is the
    "connection status changed" signal for the Messaging page."""
    try:
        return (_watcher_home() / "gateway_state.json").stat().st_mtime_ns
    except OSError:
        return None


def _pairing_sig():
    """Newest mtime across every profile's pairing store.

    An unknown DMer's pending code is written by the messaging gateway — a
    DIFFERENT process that never touches this gateway's transports — so the
    files are the only shared signal. ``platforms.changed`` cannot stand in
    for this: it tracks connect/disconnect/health, and a pairing request
    moves nothing in gateway_state.json.
    """
    home = _watcher_home()
    sig = None
    # Global store (legacy `pairing/` and consolidated `platforms/pairing/`)
    # plus every named profile's own — the Messaging page can be scoped to any
    # of them, and a request landing in a profile store must still tick.
    roots = [home / "pairing", home / "platforms" / "pairing"]
    try:
        for profile_dir in (home / "profiles").iterdir():
            roots.append(profile_dir / "pairing")
            roots.append(profile_dir / "platforms" / "pairing")
    except OSError:
        pass

    for root in roots:
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            # Only the pending/approved ledgers — _rate_limits.json moves on
            # every unauthorized DM, including ones that produce no new row.
            if not entry.name.endswith(("-pending.json", "-approved.json")):
                continue
            try:
                mtime = entry.stat().st_mtime_ns
            except OSError:
                continue
            sig = mtime if sig is None else max(sig, mtime)
    return sig


def _broadcast_watched_changes(now: float | None = None) -> None:
    """One pass over ``_CHANGE_WATCHES``: recompute due signatures, broadcast
    the events whose signature moved. First sighting seeds silently so a
    gateway boot never fires a spurious refresh storm."""
    now = time.monotonic() if now is None else now
    for event, (interval, sig_fn, payload_fn) in server._CHANGE_WATCHES.items():
        if now - server._change_checked_at.get(event, -interval) < interval:
            continue
        server._change_checked_at[event] = now
        try:
            sig = sig_fn()
        except Exception:  # noqa: BLE001 - a broken probe must not kill the loop
            continue
        if event not in server._change_sigs:
            server._change_sigs[event] = sig
            continue
        if sig == server._change_sigs[event]:
            continue
        floor = server._CHANGE_BROADCAST_FLOOR_S.get(event, 0.0)
        if floor and now - server._change_broadcast_at.get(event, -floor) < floor:
            # Floored: leave the old signature in place so the change re-fires
            # once the window opens (the trailing edge of the burst).
            continue
        server._change_sigs[event] = sig
        server._change_broadcast_at[event] = now
        try:
            server._broadcast_global_event(event, payload_fn())
        except Exception:  # noqa: BLE001
            pass


_skin_watcher_started = False


def _ensure_skin_watcher() -> None:
    """Watch cheap on-disk signatures and broadcast change events — so a skin
    Hermes activates, a pet ``/pet`` adopts, a cron the scheduler fires, or a
    messaging turn another process writes goes live on every surface within a
    couple seconds, on its own, with no client-side poll in the loop.
    Idempotent; started at gateway.ready. (Named for its original skin-only
    duty; it is the process's one change watcher.)"""
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


# Imported last: see module docstring ("Seam contract") for why a top-of-file
# import would deadlock the change_watcher-first import order.
from tui_gateway import server  # noqa: E402
