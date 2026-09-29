"""Visible agent cursor overlay — an in-page animation of the agent's pointer.

The agent drives a real browser on the user's desktop, so the user must be able to SEE what
it is doing: where the pointer goes, when it clicks, when it types. The overlay is a single
injected DOM element (fixed, ``z-index`` maxed, ``pointer-events:none``) holding a gradient
arrow with a "Czesiek" name tag, a soft halo at the tip and a short comet trail, that follows the
mouse events the automation dispatches (agent-browser / Browser Use both click with real CDP
input events, so ``mousemove`` / ``mousedown`` / ``keydown`` reach the page), animates the
move with a 240 ms eased transform, pulses a ring on click and shows a discreet caret at the
focused text field.

Two delivery paths, one script:

* built-in browser tools — :func:`after_browser_command` re-installs the overlay after every
  action command through ``agent-browser eval`` (idempotent; a navigation wipes the DOM, the
  next command re-injects);
* ``browser_exec`` (Browser Use CLI) — ``tools.browser_use_cli`` prepends a preamble that
  registers the same script via ``Page.addScriptToEvaluateOnNewDocument`` (survives every
  navigation) and installs it on the current page with ``js()``.

The element is ``aria-hidden`` + ``role="presentation"`` so it never appears in an
accessibility snapshot, and ``pointer-events:none`` so it never eats a click. Off switch:
``browser.cursor_overlay: false`` or ``AGENT_BROWSER_CURSOR_OVERLAY=false``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from tools.browser_tool_origin import origin_module as _origin

logger = logging.getLogger(__name__)

_TRUTHY = ("true", "1", "yes", "on")

#: Commands after which the overlay is (re)installed. Read-only commands (snapshot, get,
#: screenshot, console, ...) cannot have navigated, so they skip the extra subprocess.
OVERLAY_COMMANDS = frozenset({
    "open", "click", "dblclick", "type", "fill", "press", "keyboard", "hover", "focus",
    "check", "uncheck", "select", "drag", "scroll", "scrollintoview", "back", "forward",
    "reload", "mouse", "eval", "wait", "find", "upload", "download", "tab",
})

#: The overlay script. Kept small (it travels as one argv element / one ``source`` string)
#: and ASCII-only. Multi-line here for reviewability; joined to one line at use.
_OVERLAY_JS = """
(() => {
  var W = window, KEY = "__hermesCursorOverlay", ROOT = "__hermes_cursor_root";
  if (W[KEY]) { try { W[KEY].ping(); } catch (e) {} return "hermes-cursor-present"; }
  var BLUE = "rgba(64,128,255,.95)";
  var EASE = "transform 240ms cubic-bezier(.22,.61,.36,1), opacity 200ms ease-out";
  var reduced = !!(W.matchMedia && W.matchMedia("(prefers-reduced-motion: reduce)").matches);
  var root = null, dot = null, ring = null, caret = null, trail = [];
  var x = -60, y = -60, shown = false, raf = 0, caretTimer = 0;
  function css(el, text) { el.setAttribute("style", text); }
  function build() {
    if (root || !document.documentElement || document.getElementById(ROOT)) return false;
    root = document.createElement("div");
    root.id = ROOT;
    root.setAttribute("aria-hidden", "true");
    root.setAttribute("role", "presentation");
    root.setAttribute("data-hermes-overlay", "cursor");
    css(root, "position:fixed;left:0;top:0;width:0;height:0;pointer-events:none;"
             + "z-index:2147483647;contain:layout style;isolation:isolate;");
    var base = "position:absolute;left:0;top:0;border-radius:50%;opacity:0;"
             + "pointer-events:none;will-change:transform,opacity;";
    ring = document.createElement("div");
    css(ring, base + "width:30px;height:30px;margin:-15px 0 0 -15px;border:2px solid " + BLUE
             + ";background:radial-gradient(circle,rgba(64,128,255,.22),rgba(64,128,255,0) 70%);"
             + "box-shadow:0 0 14px rgba(64,128,255,.45);");
    caret = document.createElement("div");
    css(caret, "position:absolute;left:0;top:0;width:14px;height:14px;border-radius:3px;opacity:0;"
              + "border:1.5px solid rgba(64,128,255,.85);box-shadow:0 0 0 1px rgba(255,255,255,.55),"
              + "0 1px 4px rgba(0,0,0,.18);pointer-events:none;will-change:transform,opacity;");
    for (var i = 0; i < 5; i++) {
      var t = document.createElement("div");
      css(t, base + "width:" + (10 - i) + "px;height:" + (10 - i) + "px;margin:" + (-(10 - i) / 2) + "px 0 0 "
            + (-(10 - i) / 2) + "px;background:rgba(80,140,255," + (0.42 - i * 0.07) + ");");
      root.appendChild(t); trail.push(t);
    }
    root.appendChild(ring); root.appendChild(caret);
    dot = document.createElement("div");
    css(dot, "position:absolute;left:0;top:0;opacity:0;pointer-events:none;will-change:transform,opacity;"
            + "filter:drop-shadow(0 2px 5px rgba(20,30,90,.45));");
    var NS = "http://www.w3.org/2000/svg", svg = document.createElementNS(NS, "svg");
    svg.setAttribute("width", "22"); svg.setAttribute("height", "26"); svg.setAttribute("viewBox", "0 0 22 26");
    svg.setAttribute("style", "display:block;margin:-2px 0 0 -2px;overflow:visible;");
    var defs = document.createElementNS(NS, "defs"), grad = document.createElementNS(NS, "linearGradient");
    grad.setAttribute("id", "__hermes_cursor_grad"); grad.setAttribute("x1", "0"); grad.setAttribute("y1", "0");
    grad.setAttribute("x2", "1"); grad.setAttribute("y2", "1");
    var s0 = document.createElementNS(NS, "stop"), s1 = document.createElementNS(NS, "stop");
    s0.setAttribute("offset", "0"); s0.setAttribute("stop-color", "#22b8ff");
    s1.setAttribute("offset", "1"); s1.setAttribute("stop-color", "#8b5cf6");
    grad.appendChild(s0); grad.appendChild(s1); defs.appendChild(grad); svg.appendChild(defs);
    var arrow = document.createElementNS(NS, "path");
    arrow.setAttribute("d", "M2 2 L2 21 L7 16.6 L10.4 24.2 L13.8 22.7 L10.5 15.2 L17.4 14.8 Z");
    arrow.setAttribute("fill", "url(#__hermes_cursor_grad)"); arrow.setAttribute("stroke", "#ffffff");
    arrow.setAttribute("stroke-width", "1.6"); arrow.setAttribute("stroke-linejoin", "round");
    svg.appendChild(arrow); dot.appendChild(svg);
    var tag = document.createElement("div");
    tag.textContent = "Czesiek";
    css(tag, "position:absolute;left:16px;top:20px;padding:2px 9px;border-radius:999px;white-space:nowrap;"
            + "font:600 11px/16px system-ui,-apple-system,Segoe UI,sans-serif;letter-spacing:.02em;color:#fff;"
            + "background:linear-gradient(135deg,#22b8ff,#8b5cf6);box-shadow:0 0 0 1.5px rgba(255,255,255,.9),"
            + "0 4px 12px rgba(40,50,140,.35);pointer-events:none;");
    dot.appendChild(tag);
    root.appendChild(dot);
    (document.body || document.documentElement).appendChild(root);
    return true;
  }
  function place(px, py) { x = px; y = py; }
  function paint() {
    raf = 0;
    var tf = "translate3d(" + Math.round(x) + "px," + Math.round(y) + "px,0)";
    if (dot) { dot.style.transform = tf; }
    if (ring) { ring.style.transform = tf; }
    for (var i = 0; i < trail.length; i++) { trail[i].style.transform = tf; }
  }
  function schedule() { raf = raf || (W.requestAnimationFrame ? W.requestAnimationFrame(paint) : (paint(), 1)); }
  function show() {
    if (!root && !build()) return;
    if (shown) return;
    shown = true;
    paint();
    dot.style.opacity = "1"; ring.style.opacity = ".85";
    if (!reduced) {
      dot.style.transition = EASE; ring.style.transition = EASE; caret.style.transition = EASE;
      for (var i = 0; i < trail.length; i++) {
        trail[i].style.opacity = "1";
        trail[i].style.transition = "transform " + (240 + (i + 1) * 90) + "ms cubic-bezier(.22,.61,.36,1)";
      }
    }
  }
  function move(px, py) { place(px, py); show(); schedule(); }
  function pulse(px, py, strong) {
    if (!root) return;
    var size = strong ? 52 : 34, d = document.createElement("div");
    css(d, "position:absolute;left:0;top:0;width:" + size + "px;height:" + size + "px;margin:"
          + (-size / 2) + "px 0 0 " + (-size / 2) + "px;border-radius:50%;pointer-events:none;"
          + "border:2px solid " + BLUE + ";" + (strong ? "background:rgba(64,128,255,.18);" : ""));
    var tf = "translate3d(" + Math.round(px) + "px," + Math.round(py) + "px,0)";
    d.style.transform = tf;
    root.appendChild(d);
    var frames = [{transform: tf + " scale(.35)", opacity: .9}, {transform: tf + " scale(1)", opacity: 0}];
    if (d.animate) {
      var a = d.animate(frames, {duration: strong ? 520 : 420, easing: "cubic-bezier(.22,.61,.36,1)"});
      if (a && a.onfinish) { a.onfinish = function () { d.remove(); }; return; }
    }
    W.setTimeout(function () { d.remove(); }, 620);
  }
  function typing() {
    if (!root && !build()) return;
    var el = document.activeElement;
    if (!el || !el.getBoundingClientRect) return;
    var r = el.getBoundingClientRect();
    if (!r.width && !r.height) return;
    move(r.left + Math.min(12, r.width / 2), r.top + r.height / 2);
    caret.style.transform = "translate3d(" + Math.round(Math.max(0, r.right - 18)) + "px,"
                          + Math.round(r.top + 4) + "px,0)";
    caret.style.opacity = ".95";
    W.clearTimeout(caretTimer);
    caretTimer = W.setTimeout(function () { caret.style.opacity = "0"; }, 1200);
  }
  function point(e) {
    var px = e && typeof e.clientX === "number" ? e.clientX : null;
    var py = e && typeof e.clientY === "number" ? e.clientY : null;
    if ((px === null || py === null || (px === 0 && py === 0)) && e && e.target && e.target.getBoundingClientRect) {
      var r = e.target.getBoundingClientRect();
      if (r.width || r.height) { px = r.left + r.width / 2; py = r.top + r.height / 2; }
    }
    return (px === null || py === null) ? null : [px, py];
  }
  function onMove(e) { var p = point(e); if (p) { move(p[0], p[1]); } }
  function onDown(e) { var p = point(e); if (p) { move(p[0], p[1]); pulse(p[0], p[1], true); } }
  function onUp(e) { var p = point(e); if (p) { move(p[0], p[1]); pulse(p[0], p[1], false); } }
  function onType(e) {
    var t = e && e.target; if (!t || !t.tagName) { return; }
    var n = t.tagName.toLowerCase();
    if (n === "input" || n === "textarea" || n === "select" || t.isContentEditable) { typing(); }
  }
  function start() {
    if (!build()) { return; }
    W.addEventListener("mousemove", onMove, true);
    W.addEventListener("pointermove", onMove, true);
    W.addEventListener("mousedown", onDown, true);
    W.addEventListener("pointerdown", onDown, true);
    W.addEventListener("mouseup", onUp, true);
    W.addEventListener("keydown", onType, true);
    W.addEventListener("input", onType, true);
    W.addEventListener("focusin", onType, true);
  }
  W[KEY] = {
    version: 1,
    ping: function () { return "hermes-cursor-present"; },
    move: function (px, py) { move(px, py); },
    click: function (px, py) { move(px, py); pulse(px, py, true); },
    type: typing,
    show: show,
    hide: function () { if (root) { root.style.display = "none"; } }
  };
  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", start); }
  start();
  return "hermes-cursor-installed";
})()
"""


def overlay_js_source() -> str:
    """The overlay script as a SINGLE line (safe as one argv element / one ``source`` string)."""
    return " ".join(line.strip() for line in _OVERLAY_JS.splitlines() if line.strip())


def cursor_overlay_enabled() -> bool:
    """``browser.cursor_overlay`` (default True); ``AGENT_BROWSER_CURSOR_OVERLAY`` wins when set."""
    env_raw = os.environ.get("AGENT_BROWSER_CURSOR_OVERLAY", "").strip().lower()
    if env_raw:
        return env_raw in _TRUTHY
    _bt = _origin()
    return bool(_bt._cached_browser_cfg(
        "_cached_cursor_overlay", "_cursor_overlay_resolved", "cursor_overlay", True,
        lambda v: True if v is None else str(v).strip().lower() in _TRUTHY,
        "browser.cursor_overlay from config",
    ))


def _is_visible_session(session_info: Dict[str, Any]) -> bool:
    """True when a real window is on screen: headed, not Camofox, not the Lightpanda engine."""
    from tools import browser_tool_cloud as _cloud
    if not _cloud._is_headed_mode():
        return False
    if _origin()._is_camofox_mode():
        return False
    return True


def should_inject(command: str, session_info: Dict[str, Any]) -> bool:
    """True when this command should (re)install the overlay."""
    if command not in OVERLAY_COMMANDS:
        return False
    if not cursor_overlay_enabled():
        return False
    return _is_visible_session(session_info)


def after_browser_command(task_id: str, command: str,
                          session_info: Dict[str, Any], engine: str = "auto") -> None:
    """Best-effort overlay (re)install after a browser command; never raises, never blocks a result.

    Runs through ``agent-browser eval`` with ``_skip_overlay`` so the injection cannot recurse
    into this hook. Lightpanda has no renderer to animate, so it is skipped.
    """
    try:
        if engine == "lightpanda" or not should_inject(command, session_info):
            return
        session_name = str(session_info.get("session_name") or "")
        if not session_name:
            return
        from tools.browser_tool_session import _run_browser_command
        _run_browser_command(task_id, "eval", [overlay_js_source()], _skip_overlay=True)
    except Exception as exc:  # pragma: no cover - defensive: the overlay must never break a call
        logger.debug("cursor overlay injection failed for task=%s (%s): %s", task_id, command, exc)