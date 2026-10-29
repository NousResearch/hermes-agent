"""Pure companion spritesheet helpers (revision id, per-row frame counts, row taxonomy, payload clone) for
the ``companion.*`` RPCs in ``tui_gateway.server``."""

from __future__ import annotations

import contextlib


def _companion_sheet_revision(spritesheet) -> str:
    """Stable revision id for one spritesheet file."""
    with contextlib.suppress(Exception):
        stat = spritesheet.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"
    return "0:0"


def _clone_companion_payload(payload: dict) -> dict:
    """Shallow-clone cached payloads so callers can't mutate shared state."""
    out = dict(payload)
    for key, kind in (("framesByState", dict), ("framesByRow", dict), ("stateRows", list)):
        if isinstance(payload.get(key), kind):
            out[key] = kind(payload[key])
    return out


def _companion_row_frame_counts(spritesheet) -> dict:
    """Real frame count per concrete spritesheet row name."""
    with contextlib.suppress(Exception):
        from PIL import Image
        from agent.companion import constants, render
        with Image.open(spritesheet) as opened:
            image = opened.convert("RGBA")
        W, H = constants.FRAME_W, constants.FRAME_H
        cols = max(1, image.width // W)
        row_count = max(1, image.height // H)
        rows = constants.state_rows_for_grid(row_count)
        out: dict[str, int] = {}
        for row_idx, name in enumerate(rows[:row_count]):
            top = row_idx * H
            blank = lambda col: render._frame_is_blank(image.crop((col * W, top, col * W + W, top + H)))
            out[name] = next((col for col in range(cols) if blank(col)), cols)  # frames before the first blank cell
        return out
    return {}


def _companion_state_rows(spritesheet) -> list[str]:
    """Row taxonomy for the concrete sheet (legacy 8-row or current 9-row atlas), in the renderer's `CompanionState` names."""
    from agent.companion import constants
    with contextlib.suppress(Exception):
        from PIL import Image
        with Image.open(spritesheet) as image:
            row_count = max(1, image.height // constants.FRAME_H)
        return list(constants.state_rows_for_grid(row_count))
    return list(constants.STATE_ROWS)
