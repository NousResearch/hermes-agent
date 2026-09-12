"""Thumbnail of a bot's screen: one JPEG grab of the profile's Xvnc display.

Feeds the Screen hero in Hermes Desktop (the big preview at the top of a bot's pane). Read-only:
it never touches the lease, so a human in control is not disturbed and the bot is not blocked.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Optional

from tools.bot_desktop import runtime

THUMB_MAX = (960, 600)


def thumbnail_data_url(max_size: tuple[int, int] = THUMB_MAX, quality: int = 72) -> Optional[str]:
    """``data:image/jpeg;base64,...`` of the running screen, or ``None`` when no screen is up."""
    env = runtime.published_env()
    display = env.get("DISPLAY")
    if not display or runtime._launcher_pid() is None:
        return None
    from PIL import ImageGrab  # Pillow is a hard dependency; import lazily to keep status calls cheap

    # Xlib reads XAUTHORITY from the process env; the launcher publishes a per-profile cookie file.
    previous = os.environ.get("XAUTHORITY")
    if env.get("XAUTHORITY"):
        os.environ["XAUTHORITY"] = env["XAUTHORITY"]
    try:
        image = ImageGrab.grab(xdisplay=display)
    finally:
        if previous is None:
            os.environ.pop("XAUTHORITY", None)
        else:
            os.environ["XAUTHORITY"] = previous
    image.thumbnail(max_size)
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
