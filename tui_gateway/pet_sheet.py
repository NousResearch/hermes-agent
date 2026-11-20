"""Pure pet spritesheet helpers (revision id, row taxonomy + per-row frame counts) for the ``pet.*`` RPCs
in ``tui_gateway.server``."""

from __future__ import annotations

import contextlib


def _pet_sheet_revision(spritesheet) -> str:
    """Revision id for the sheet *and* its ``pet.json``, so editing either refreshes the renderer."""
    from agent.pet.behaviors import pet_revision  # never raises
    return pet_revision(spritesheet)


def _pet_rows(spritesheet, rows) -> tuple[list[str], dict]:
    """(stateRows, framesByRow): the sheet's row names (``PetSpec.rows``) and each row's real frame count;
    ``(constants.STATE_ROWS, {})`` when the sheet is unreadable."""
    from agent.pet import constants
    with contextlib.suppress(Exception):
        from PIL import Image
        from agent.pet import render
        with Image.open(spritesheet) as opened:
            image = opened.convert("RGBA")
        W, H = constants.FRAME_W, constants.FRAME_H
        cols = max(1, image.width // W)
        out: dict[str, int] = {}
        for row_idx, name in enumerate(rows[:max(1, image.height // H)]):
            top = row_idx * H
            blank = lambda col: render._frame_is_blank(image.crop((col * W, top, col * W + W, top + H)))
            out[name] = next((col for col in range(cols) if blank(col)), cols)  # frames before the first blank cell
        return list(rows), out
    return list(constants.STATE_ROWS), {}
