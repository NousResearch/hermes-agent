"""zoom / invoke_menu action handlers for the computer_use tool (rows in ``tool._ACTIONS``)."""

from __future__ import annotations

import json


def _do_zoom(backend, action, args, fence=lambda: None, **_):
    region = args.get("region")
    if not (isinstance(region, list) and len(region) == 4 and all(isinstance(v, (int, float)) for v in region)
            and region[0] < region[2] and region[1] < region[3]):
        return json.dumps({"error": "zoom requires `region` = [x1, y1, x2, y2] in capture pixels with x1<x2, y1<y2"})
    cap = backend.zoom(tuple(region))
    fence()
    from tools.computer_use.tool import _capture_response  # late: tool.py imports this module
    # No session_id: a zoom crop must never be deduped against (or replace) the full capture's frame.
    return _capture_response(cap)


def _do_invoke_menu(backend, action, args, **_):
    path = args.get("menu_path")
    if not (isinstance(path, list) and path and all(isinstance(s, str) and s.strip() for s in path)):
        return json.dumps({"error": "invoke_menu requires `menu_path`, e.g. [\"File\", \"Export\", \"PDF…\"]"})
    res = backend.invoke_menu([s.strip() for s in path])
    if res.ok and args.get("capture_after"):
        return res  # _dispatch takes the follow-up capture
    # Flat payload, no escalate verdict: the driver resolves the path through accessibility only, and the pixel
    # rungs that verdict recommends are exactly the fallback a refused menu path must not take.
    optional = {k: v for k, v in (("message", res.message), ("code", res.code)) if v}
    return json.dumps({"ok": res.ok, "action": res.action, **optional})
