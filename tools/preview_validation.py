"""Raw native-drag boundary shared by tool dispatch and the preview wire."""

import math
from collections.abc import Mapping


def preview_drag_error(args: Mapping) -> str | None:
    """Validate before argument repair; omission is different from explicit null."""
    action = args.get("action")
    verb = action.strip().lower() if isinstance(action, str) else action
    if verb != "drag":
        return "dx and dy are only valid for drag." if "dx" in args or "dy" in args else None
    if set(args) - {"action", "ref", "selector", "dx", "dy"}:
        return "drag accepts only action, exactly one ref or selector, dx and dy."
    targets = [key for key in ("ref", "selector") if key in args]
    if len(targets) != 1 or not isinstance(args[targets[0]], str) or not args[targets[0]].strip():
        return "drag needs exactly one non-empty ref or selector."
    for axis in ("dx", "dy"):
        value = args.get(axis)
        if type(value) not in (int, float) or not -2000 <= value <= 2000 or not math.isfinite(value):
            return "drag dx and dy must be finite numbers within -2000 to 2000 CSS pixels."
    if args["dx"] == 0 and args["dy"] == 0:
        return "drag needs a non-zero movement."
    return None
