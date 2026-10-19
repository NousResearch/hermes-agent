"""Where an extension page landed: the check after the fact for the browser extension lane.

``_extension_url_guard`` (in ``tools.browser_tool``) judges the URL that was asked for, before anything is
sent. A redirect, a clicked link or a submitted form can end somewhere else, so the URL the controller reports
back is judged too, by the same policy, for every result that carries a page address (navigate, back, click,
press, type, fill, select, hover, scroll, snapshot, the page a batch ended on and each tab in a tab list).
A page that fails is withheld whole, and the tab that is showing it is remembered: the extension cannot blank
a tab the way the built-in browser does, so every later call on that tab is refused until it moves to an
allowed page or is closed.

Facade-owned helpers are read through ``_bt`` (``tools.browser_tool``, resolved per call) — no import cycle.
"""

import json
import logging
import threading
import time
import urllib.parse
from collections import OrderedDict
from typing import Any, Optional

from tools import browser_tool_args as _args
from tools.browser_extension_router import RouteContext
from tools.browser_tool_origin import origin_module as _origin

logger = logging.getLogger(__name__)

# What a withheld batch result keeps: how far the batch got, never anything read from the page.
_BATCH_PROGRESS_KEYS = ("status", "stopped_reason", "completed", "remaining")
_STEP_KEYS = ("index", "action", "status", "error")
# Calls that can run on a withheld tab: they leave the page (or the tab) rather than use it. Everything else
# (snapshot, screenshot, click, type, scroll, ...) is refused there.
_LEAVING_ACTIONS = frozenset({
    "browser_navigate", "browser_back", "browser_tab_create", "browser_tabs", "browser_tab_close",
    "browser_tab_activate", "browser_tab_group", "browser_tab_ungroup",
})
_LEAVING_STEPS = frozenset({"navigate", "back"})
# A new tab's address is only known once its navigation commits; ask the browser for it a few times.
_SETTLE_POLLS = 6
_SETTLE_INTERVAL_S = 0.5
_UNSETTLED_URLS = frozenset({"", "about:blank"})
# Bounds on what is remembered: lanes, and withheld tabs per lane.
_MAX_LANES = 256
_MAX_TABS = 64


class _Withheld:
    """The tabs, per lane, that last showed a page the policy withheld: ``{lane key: {tab id or None: host}}``.
    ``None`` is the current tab, which a call that names no tab acts on."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lanes: OrderedDict[tuple, OrderedDict[Optional[int], str]] = OrderedDict()

    def add(self, key: tuple, tab: Optional[int], host: str) -> None:
        with self._lock:
            tabs = self._lanes.setdefault(key, OrderedDict())
            tabs.pop(tab, None)
            tabs[tab] = host
            while len(tabs) > _MAX_TABS:
                tabs.popitem(last=False)
            self._lanes.move_to_end(key)
            while len(self._lanes) > _MAX_LANES:
                self._lanes.popitem(last=False)

    def discard(self, key: tuple, tab: Optional[int]) -> None:
        with self._lock:
            tabs = self._lanes.get(key)
            if tabs is not None:
                tabs.pop(tab, None)
                if not tabs:
                    del self._lanes[key]

    def host(self, key: tuple, tab: Optional[int]) -> Optional[str]:
        with self._lock:
            return (self._lanes.get(key) or {}).get(tab)

    def any(self, key: tuple) -> bool:
        with self._lock:
            return bool(self._lanes.get(key))

    def clear(self) -> None:
        with self._lock:
            self._lanes.clear()


_withheld = _Withheld()


def reset_withheld() -> None:
    """Forget every withheld tab (tests; a new process starts empty)."""
    _withheld.clear()


def _tab_arg(args: dict[str, Any]) -> Optional[int]:
    tab = args.get("tab_id")
    return tab if isinstance(tab, int) and not isinstance(tab, bool) and tab > 0 else None


def _acted_on_tab(args: dict[str, Any]) -> Optional[int]:
    """The tab a call runs in: ``None``, the current tab (no action takes a tab to act in yet)."""
    return None


def _decode(result: Any) -> Optional[dict[str, Any]]:
    payload = result
    if isinstance(result, str):
        try:
            payload = json.loads(result)
        except ValueError:
            return None
    return payload if isinstance(payload, dict) else None


def _landed_url(action: str, payload: dict[str, Any]) -> str:
    """The URL an extension result says the call landed on, or ``""`` when it reports none."""
    if action == "browser_tab_create":
        payload = payload.get("tab") if isinstance(payload.get("tab"), dict) else {}
    url = payload.get("url")
    return url if isinstance(url, str) else ""


def _policy_error(url: str) -> Optional[dict]:
    """The block the landed-URL policy puts on ``url``: the cloud-metadata floor and the operator's website
    policy, but not the private-address rule, which the extension lane skips on purpose."""
    return _origin()._url_policy_error(url, auto_local=True) if url else None


def _blocked_host(landed: str, policy_error: dict) -> str:
    """The host to name: the one the website policy matched, else the landed URL's. A hostname and nothing
    else, so a page cannot put prose in front of the model through its address."""
    try:
        host = (policy_error.get("blocked_by_policy") or {}).get("host") or urllib.parse.urlsplit(landed).hostname or ""
    except ValueError:  # a URL too malformed to name a host
        return ""
    return "".join(ch for ch in str(host) if ch.isascii() and (ch.isalnum() or ch in ".:-_"))[:253]


_STAYS_PUT = (" The tab is still on that page and Hermes will not read or act on it: go to another page "
              "(browser_navigate or browser_back) or close it (browser_tab_close).")


def redirect_blocked_result(action: str, payload: dict[str, Any], landed: str, policy_error: dict,
                            *, moved_back: bool = False) -> dict[str, Any]:
    """What the model gets in place of a page that landed on a blocked address. It names the host and nothing
    else the page or its URL said: no path, query, title, snapshot or text."""
    host = _blocked_host(landed, policy_error)
    named = f" ({host})" if host else ""
    result: dict[str, Any] = {
        "success": False,
        "code": "redirect_blocked",
        "error": f"Blocked: the page landed on a blocked address{named}. Its content was withheld, and the action "
                 "itself has already run. Do not continue on this page: do not act on it or read it.",
        "blocked_host": host,
    }
    if moved_back:
        result["error"] += " The tab was taken back to the previous page."
    elif action == "browser_tab_create":
        result["error"] += " The tab stays open; browser_tab_close closes it. Hermes will not read or act on it."
    else:
        result["error"] += _STAYS_PUT
    matched = policy_error.get("blocked_by_policy")
    if isinstance(matched, dict):
        result["blocked_by_policy"] = {key: matched.get(key) for key in ("host", "rule", "source")}
    if action == "browser_tab_create":
        tab_id = (payload.get("tab") or {}).get("id")
        if isinstance(tab_id, int) and not isinstance(tab_id, bool):
            result["tab_id"] = tab_id
    elif action == "browser_batch":
        result.update({key: payload[key] for key in _BATCH_PROGRESS_KEYS if key in payload})
        if isinstance(payload.get("steps"), list):
            result["steps"] = [{key: step[key] for key in _STEP_KEYS if key in step}
                               for step in payload["steps"] if isinstance(step, dict)]
    return result


def _tab_listing_url(listing: Any, tab_id: int) -> str:
    payload = _decode(listing) or {}
    for tab in payload.get("tabs") if isinstance(payload.get("tabs"), list) else []:
        if isinstance(tab, dict) and tab.get("id") == tab_id:
            return tab["url"] if isinstance(tab.get("url"), str) else ""
    return ""


def _settle_new_tab(route: RouteContext, tab_id: int) -> str:
    """The address a new tab committed to. The controller reports a created tab the moment it exists, before
    any redirect has run, so the tab is read back until it has an address (``""`` when it never does or the
    controller cannot list tabs)."""
    for attempt in range(_SETTLE_POLLS):
        if attempt:
            time.sleep(_SETTLE_INTERVAL_S)
        try:
            url = _tab_listing_url(route.send("browser_tabs", {}), tab_id)
        except Exception:
            logger.debug("could not read back new tab %s", tab_id, exc_info=True)
            return ""
        if url not in _UNSETTLED_URLS:
            return url
    return ""


def _move_back(route: RouteContext, tab: Optional[int]) -> bool:
    """Take a tab off a withheld page with one browser_back, the nearest thing to the built-in lane's
    about:blank (the extension only navigates to http(s) pages). True when it landed on an allowed page."""
    try:
        outcome = _decode(route.send("browser_back", {})) or {}
    except Exception:
        logger.debug("could not move a withheld tab off its page", exc_info=True)
        return False
    landed = _landed_url("browser_back", outcome)
    return bool(landed) and _policy_error(landed) is None


def _moves_the_page(action: str, args: dict[str, Any], payload: dict[str, Any]) -> bool:
    """Whether this call may itself have taken the tab to the page it reports (so moving back undoes it). A
    read, or a tab that was already there, is not moved: it is only remembered."""
    if action in ("browser_navigate", "browser_back"):
        return True
    if action == "browser_batch":
        steps = args.get("actions") if isinstance(args.get("actions"), list) else []
        return any(_args.step_action(step) in _LEAVING_STEPS for step in steps)
    return payload.get("navigated") is True


def _guard_tab_list(key: tuple, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The tab list with each tab on a blocked address replaced by its id and host, or ``None`` when no tab
    was. Also keeps the withheld tabs current: a tab seen on an allowed address is no longer withheld."""
    tabs = payload.get("tabs")
    if not isinstance(tabs, list):
        return None
    listed, changed = [], False
    for tab in tabs:
        url = tab.get("url") if isinstance(tab, dict) and isinstance(tab.get("url"), str) else ""
        tab_id = tab.get("id") if isinstance(tab, dict) and isinstance(tab.get("id"), int) \
            and not isinstance(tab.get("id"), bool) else None
        error = _policy_error(url)
        if error is None:
            if tab_id is not None and url:
                _withheld.discard(key, tab_id)
            listed.append(tab)
            continue
        host = _blocked_host(url, error)
        if tab_id is not None:
            _withheld.add(key, tab_id, host)
        listed.append({"id": tab.get("id"), "active": tab.get("active"), "withheld": True, "blocked_host": host})
        changed = True
    return {**payload, "tabs": listed} if changed else None


def extension_landing_guard(action: str, args: dict[str, Any], result: Any, route: Optional[RouteContext] = None) -> Any:
    """Router result guard for the extension lane. Any address the controller reports (the page a navigation,
    click, key press or read ended on, a new tab's, each tab in a tab list) goes through the policy the
    built-in lane applies after a redirect: the cloud-metadata floor and the operator's website policy, but
    not the private-address rule, which the extension lane skips on purpose. A page that fails it is withheld
    whole, so nothing it showed reaches the model, and its tab is remembered so later calls on it are refused
    (see :func:`extension_withheld_guard`). Anything that reports no address passes through untouched."""
    payload = _decode(result)
    if payload is None:
        return result
    key = route.key if route is not None else ()
    tab = _acted_on_tab(args)
    _bt = _origin()
    dump = (lambda value: _bt._dumps(value)) if isinstance(result, str) else (lambda value: value)

    if action == "browser_tab_close":
        _withheld.discard(key, _tab_arg(args))
        return result
    if action == "browser_tabs":
        listed = _guard_tab_list(key, payload)
        return result if listed is None else dump(listed)

    landed = _landed_url(action, payload)
    if action == "browser_tab_create":
        new_tab = payload.get("tab") if isinstance(payload.get("tab"), dict) else {}
        tab = new_tab.get("id") if isinstance(new_tab.get("id"), int) and not isinstance(new_tab.get("id"), bool) else None
        if not landed and tab is not None and route is not None:
            landed = _settle_new_tab(route, tab)
    policy_error = _policy_error(landed)
    if policy_error is None:
        if landed:
            _withheld.discard(key, tab)
            if action == "browser_tab_create" and not _landed_url(action, payload):
                return dump({**payload, "tab": {**payload["tab"], "url": landed}})
        return result

    blocked_host = _blocked_host(landed, policy_error)
    _withheld.add(key, tab, blocked_host)
    moved_back = False
    if route is not None and action != "browser_tab_create" and _moves_the_page(action, args, payload):
        moved_back = _move_back(route, tab)
        if moved_back:
            _withheld.discard(key, tab)
    blocked = redirect_blocked_result(action, payload, landed, policy_error, moved_back=moved_back)
    logger.warning("%s landed on a blocked address (host %s); page withheld", action, blocked["blocked_host"])
    return dump(blocked)


def extension_withheld_guard(action: str, args: dict[str, Any], route: Optional[RouteContext] = None) -> Optional[str]:
    """Router guard for the extension lane: a refusal (JSON) for a call that would use a tab still showing a
    page the landed-URL policy withheld, else ``None``. Calls that leave the page or the tab (navigate, back,
    open, list, close, switch, group) go through, and so does a batch that starts by leaving the page, whose
    own result is judged again."""
    if route is None or not _withheld.any(route.key) or action in _LEAVING_ACTIONS:
        return None
    if action == "browser_batch":
        steps = args.get("actions") if isinstance(args.get("actions"), list) else []
        if steps and _args.step_action(steps[0]) in _LEAVING_STEPS:
            return None
    host = _withheld.host(route.key, _acted_on_tab(args))
    if host is None:
        return None
    _bt = _origin()
    named = f" ({host})" if host else ""
    return _bt._dumps(_bt._err(
        f"Blocked: this tab is still on a page Hermes withheld{named}, so {action} was not sent. Go to another "
        "page with browser_navigate or browser_back, or close the tab with browser_tab_close.",
        code="tab_withheld", blocked_host=host))
