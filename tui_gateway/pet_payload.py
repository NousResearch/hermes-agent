"""Pet payload / spritesheet build helpers (extracted from tui_gateway.server).

Slice R3-S1 of the tui_gateway/server.py god-file extraction - epic #78647,
target #78630. Byte-verbatim move of the pet payload/spritesheet cluster
(window 7989-8282); server.py re-exports every name so bare-name consumers
(HandlerRegistry-installed pet handlers, the R1 ``pet.changed`` watcher, and
test monkeypatches) keep resolving through the server namespace unchanged.
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)


def _pet_frame_counts(spritesheet) -> dict:
    """Real (padding-trimmed) frame count per state, for the desktop canvas.

    Fail-open: a decode hiccup returns ``{}`` and the canvas falls back to its
    static ``framesPerState`` rather than breaking the (cosmetic) pet.
    """
    try:
        from agent.pet import render

        return render.state_frame_counts(str(spritesheet))
    except Exception:  # noqa: BLE001 - cosmetic, never break the surface
        return {}


_pet_payload_cache_lock = threading.Lock()
_pet_payload_cache: dict[tuple, dict] = {}


def _pet_sheet_revision(spritesheet) -> str:
    """Stable revision id for one spritesheet file."""
    try:
        stat = spritesheet.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"
    except Exception:  # noqa: BLE001 - cosmetic, never break the surface
        return "0:0"


def _pet_payload_cache_key(pet, *, scale: float) -> tuple | None:
    """Cache key for the expensive sprite payload build."""
    try:
        stat = pet.spritesheet.stat()
    except Exception:  # noqa: BLE001
        return None
    return (
        str(pet.spritesheet),
        stat.st_mtime_ns,
        stat.st_size,
        pet.slug,
        pet.display_name,
        round(scale, 4),
    )


def _clone_pet_payload(payload: dict) -> dict:
    """Shallow-clone cached payloads so callers can't mutate shared state."""
    out = dict(payload)
    if isinstance(payload.get("framesByState"), dict):
        out["framesByState"] = dict(payload["framesByState"])
    if isinstance(payload.get("framesByRow"), dict):
        out["framesByRow"] = dict(payload["framesByRow"])
    if isinstance(payload.get("stateRows"), list):
        out["stateRows"] = list(payload["stateRows"])
    return out


def _pet_row_frame_counts(spritesheet) -> dict:
    """Real frame count per concrete spritesheet row name."""
    try:
        from PIL import Image

        from agent.pet import constants, render

        with Image.open(spritesheet) as opened:
            image = opened.convert("RGBA")
        cols = max(1, image.width // constants.FRAME_W)
        row_count = max(1, image.height // constants.FRAME_H)
        rows = constants.state_rows_for_grid(row_count)
        out: dict[str, int] = {}
        for row_idx, name in enumerate(rows[:row_count]):
            top = row_idx * constants.FRAME_H
            count = 0
            for col in range(cols):
                left = col * constants.FRAME_W
                frame = image.crop((left, top, left + constants.FRAME_W, top + constants.FRAME_H))
                if render._frame_is_blank(frame):
                    break
                count += 1
            out[name] = count
        return out
    except Exception:  # noqa: BLE001 - cosmetic, never break the surface
        return {}


def _pet_config_scale() -> float:
    """Configured ``display.pet.scale`` (or the engine default), never raises."""
    from agent.pet import constants

    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        display = cfg.get("display", {}) if isinstance(cfg.get("display"), dict) else {}
        pet_cfg = display.get("pet", {}) if isinstance(display.get("pet"), dict) else {}
        return float(pet_cfg.get("scale", constants.DEFAULT_SCALE) or constants.DEFAULT_SCALE)
    except Exception:  # noqa: BLE001
        return constants.DEFAULT_SCALE


def _pet_sprite_payload(pet, *, scale: float) -> dict:
    """Build the renderer payload (spritesheet bytes + geometry) for *pet*.

    Shared by ``pet.info`` (the active mascot) and ``pet.hatch`` (the unadopted
    preview) so both feed the desktop canvas / TUI from one shape.
    """
    import base64

    from agent.pet import constants

    cache_key = _pet_payload_cache_key(pet, scale=scale)
    if cache_key is not None:
        with _pet_payload_cache_lock:
            cached = _pet_payload_cache.get(cache_key)
        if cached is not None:
            return _clone_pet_payload(cached)

    raw = pet.spritesheet.read_bytes()
    suffix = pet.spritesheet.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/webp"
    payload = {
        "slug": pet.slug,
        "displayName": pet.display_name,
        "mime": mime,
        "spritesheetBase64": base64.standard_b64encode(raw).decode("ascii"),
        "spritesheetRevision": _pet_sheet_revision(pet.spritesheet),
        "frameW": constants.FRAME_W,
        "frameH": constants.FRAME_H,
        "framesPerState": constants.FRAMES_PER_STATE,
        "framesByState": _pet_frame_counts(pet.spritesheet),
        "framesByRow": _pet_row_frame_counts(pet.spritesheet),
        "loopMs": constants.LOOP_MS,
        "scale": scale,
        "stateRows": _pet_state_rows(pet.spritesheet),
    }
    if cache_key is not None:
        with _pet_payload_cache_lock:
            _pet_payload_cache[cache_key] = payload
            while len(_pet_payload_cache) > 8:
                _pet_payload_cache.pop(next(iter(_pet_payload_cache)))
    return _clone_pet_payload(payload)


def _pet_active_selection():
    """Resolve configured active pet + scale from config."""
    from agent.pet import constants, store

    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        display = cfg.get("display", {}) if isinstance(cfg.get("display"), dict) else {}
        pet_cfg = display.get("pet", {}) if isinstance(display.get("pet"), dict) else {}
    except Exception:
        pet_cfg = {}

    enabled = bool(pet_cfg.get("enabled"))
    configured_slug = str(pet_cfg.get("slug", "") or "")
    pet = store.resolve_active_pet(configured_slug) if enabled else None
    scale = float(pet_cfg.get("scale", constants.DEFAULT_SCALE) or constants.DEFAULT_SCALE)
    return enabled, pet, scale


def _pet_state_rows(spritesheet) -> list[str]:
    """Row taxonomy for the concrete active pet sheet.

    Hermes has to support both the legacy 8-row petdex atlas and the current
    Codex/petdex 9-row atlas. The desktop canvas gets this list and indexes it
    with the same `PetState` names the Python renderer uses.
    """
    try:
        from PIL import Image

        from agent.pet import constants

        with Image.open(spritesheet) as image:
            row_count = max(1, image.height // constants.FRAME_H)
        return list(constants.state_rows_for_grid(row_count))
    except Exception:  # noqa: BLE001 - cosmetic, never break the surface
        from agent.pet import constants

        return list(constants.STATE_ROWS)


def _pet_gen_root():
    """Profile-scoped staging dir for in-progress generation drafts."""
    from hermes_constants import get_hermes_home

    root = get_hermes_home() / "cache" / "pet-gen"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _pet_gen_sweep(root, *, max_age_s: float = 3600.0) -> None:
    """Drop stale draft staging dirs so cache never grows unbounded."""
    import shutil
    import time

    try:
        now = time.time()
        for child in root.iterdir():
            if child.is_dir() and now - child.stat().st_mtime > max_age_s:
                shutil.rmtree(child, ignore_errors=True)
    except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
        logger.debug("pet-gen sweep failed: %s", exc)


def _pet_png_data_uri(path, *, max_px: int = 160) -> str:
    """Downscaled PNG data URI for a draft image (small preview payload)."""
    import base64
    import io

    from PIL import Image

    with Image.open(path) as opened:
        img = opened.convert("RGBA")
    img.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.standard_b64encode(buf.getvalue()).decode("ascii")


# Cooperative cancellation for the heavy pet generation paths. The client's Stop
# aborts its RPC immediately, but the worker-pool generation keeps running unless
# told to stop — pet.cancel flips a token's flag, which generate_base_drafts /
# hatch_pet poll between provider calls to skip work they haven't started.
_pet_cancel_lock = threading.Lock()
_pet_cancelled: set[str] = set()
_PET_REFERENCE_MIME_EXT = {
    "png": "png",
    "jpeg": "jpg",
    "jpg": "jpg",
    "webp": "webp",
    "gif": "gif",
}
try:
    _PET_REFERENCE_MAX_BYTES = max(
        1,
        int(os.environ.get("HERMES_PET_REFERENCE_MAX_BYTES") or str(16 * 1024 * 1024)),
    )
except (TypeError, ValueError):
    _PET_REFERENCE_MAX_BYTES = 16 * 1024 * 1024


def _pet_reference_images_from_data_url(ref_raw: str, stage) -> list:
    """Decode + validate a reference-image data URL into the stage dir."""
    import base64
    import binascii
    import re as _re

    match = _re.match(r"^data:image/([a-zA-Z0-9.+-]+);base64,(.*)$", ref_raw, _re.DOTALL)
    if not match:
        raise ValueError("invalid reference image format")

    mime = match.group(1).lower()
    ext = _PET_REFERENCE_MIME_EXT.get(mime)
    if ext is None:
        raise ValueError("unsupported reference image type")

    payload = "".join(match.group(2).split())
    approx = (len(payload) * 3) // 4
    if approx > _PET_REFERENCE_MAX_BYTES:
        raise ValueError("reference image too large")

    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid reference image data") from exc

    if len(raw) > _PET_REFERENCE_MAX_BYTES:
        raise ValueError("reference image too large")

    ref_path = stage / f"reference.{ext}"
    ref_path.write_bytes(raw)
    return [ref_path]


def _pet_cancel_arm(token: str) -> None:
    """Clear a stale cancel flag at the start of a generate/hatch run."""
    with _pet_cancel_lock:
        _pet_cancelled.discard(token)


def _pet_cancel_request(token: str) -> None:
    with _pet_cancel_lock:
        _pet_cancelled.add(token)


def _pet_is_cancelled(token: str) -> bool:
    with _pet_cancel_lock:
        return token in _pet_cancelled


def _pet_cancel_release(token: str) -> None:
    with _pet_cancel_lock:
        _pet_cancelled.discard(token)
