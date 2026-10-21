"""Run vault fill/inspection JS in Node so the behavioural tests never skip.

Why this exists
---------------
The vault fill is JavaScript. Testing it properly means executing it against real DOM
behaviour, not asserting on the script's text. Upstream has no Python JS engine and no
Playwright in its unit-test environment, so a Playwright-based test silently SKIPS on CI --
and a skip among passes is the same false-success shape this whole fix is about (a reviewer
read "4 passed, 7 skipped" as a clean run when the 7 that mattered had not run at all).

Node IS a hard dependency of this repo and is already exec'd from pytest elsewhere
(tests/tools/test_mcp_node_abi.py and friends), so driving the JS through Node removes the
skip entirely. Chromium is preferred when present (truest fidelity); Node is the fallback,
and both exercise the same assertions.

The DOM shim is deliberately minimal: enough to resolve selectors, hold option lists, track
`value`, dispatch events and expose the properties the fill script reads. It is not a browser
engine, and it is not asked to render -- the defect is in option matching and write reporting.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

NODE = shutil.which("node")

# --------------------------------------------------------------------------------------------
# A minimal DOM, in JS. `__HTML__` is replaced with the page markup for each case.
# --------------------------------------------------------------------------------------------
DOM_SHIM = r"""
const __HTML__ = __PAGE_HTML__;

function parseAttrs(src) {
  const attrs = {};
  const re = /([a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*=\s*("([^"]*)"|'([^']*)'|([^\s>]+))/g;
  let m;
  while ((m = re.exec(src))) attrs[m[1].toLowerCase()] = m[3] ?? m[4] ?? m[5] ?? "";
  return attrs;
}

function makeEl(tag, attrs) {
  const el = {
    tagName: tag.toUpperCase(),
    attributes: attrs,
    name: attrs.name || attrs.id || "",
    id: attrs.id || "",
    type: attrs.type || (tag === "select" ? "select-one" : "text"),
    disabled: "disabled" in attrs,
    readOnly: "readonly" in attrs,
    _value: attrs.value !== undefined ? attrs.value : "",
    options: [],
    children: [],
    style: {},
    focused: false,
    events: [],
    removeAttribute(n) { delete this.attributes[n]; },
    setAttribute(n, v) { this.attributes[n] = v; },
    getAttribute(n) { return this.attributes[n] !== undefined ? this.attributes[n] : null; },
    focus() { this.focused = true; },
    dispatchEvent(e) { this.events.push(e && e.type); return true; },
    querySelectorAll() { return []; },
    // the inspection treats a zero-length getClientRects as "not rendered" and skips the
    // control, so a visible shim element must report one rect
    getClientRects() { return [{}]; },
    getBoundingClientRect() { return { width: 120, height: 24, top: 0, left: 0 }; },
    get optionList() { return this.options; },
  };
  Object.defineProperty(el, "value", {
    get() { return this._value; },
    set(v) { this._value = String(v); },
  });
  return el;
}

// --- parse the markup: <select ...> <option ...> ... </select> and standalone <input ...> ---
function buildDom(html) {
  const nodes = [];
  const selectRe = /<select([^>]*)>([\s\S]*?)<\/select>/gi;
  let m;
  while ((m = selectRe.exec(html))) {
    const sel = makeEl("select", parseAttrs(m[1]));
    const optRe = /<option([^>]*)>([\s\S]*?)<\/option>/gi;
    let o;
    while ((o = optRe.exec(m[2]))) {
      const oa = parseAttrs(o[1]);
      const text = o[2].replace(/<[^>]*>/g, "").trim();
      const opt = makeEl("option", oa);
      opt.textContent = oa.selected !== undefined ? text : text;
      opt.value = oa.value !== undefined ? oa.value : text;
      opt._value = opt.value;
      sel.options.push(opt);
    }
    // a select with nothing selected defaults to its first option
    const preselected = sel.options.find((x) => "selected" in x.attributes);
    sel._value = preselected ? preselected.value : (sel.options[0] ? sel.options[0].value : "");
    nodes.push(sel);
  }
  const withoutSelects = html.replace(selectRe, "");
  const inputRe = /<input([^>]*?)\/?>/gi;
  while ((m = inputRe.exec(withoutSelects))) nodes.push(makeEl("input", parseAttrs(m[1])));
  return nodes;
}

const ALL = buildDom(__HTML__);

global.document = {
  querySelectorAll(sel) {
    if (sel === "input, select") return ALL;
    if (sel === "[data-hermes-vault-slot]") return ALL.filter((e) => e.attributes["data-hermes-vault-slot"]);
    const attr = /^\[data-hermes-vault-slot="([^"]*)"\]$/.exec(sel);
    if (attr) return ALL.filter((e) => e.attributes["data-hermes-vault-slot"] === attr[1]);
    return [];
  },
  querySelector(sel) {
    const r = this.querySelectorAll(sel);
    return r.length ? r[0] : null;
  },
  forms: [],
  body: { style: {} },
  getElementById(id) { return ALL.find((e) => e.id === id) || null; },
};

global.window = { location: { origin: __ORIGIN__ } };
global.location = global.window.location;
global.getComputedStyle = () => ({ display: "block", visibility: "visible", opacity: "1" });
global.Event = class { constructor(t) { this.type = t; } };
global.InputEvent = class { constructor(t) { this.type = t; } };
global.HTMLInputElement = { prototype: {} };
class FakeEl {}
global.Element = FakeEl;

global.__inspection = __INSPECTION__;
global.__result = (function () {
  const inspected = eval(global.__inspection);
  const fillSrc = __FILL_JS__;
  const out = fillSrc === null ? null : eval(fillSrc);
  return {
    inspection: inspected,
    out,
    held: ALL.filter((e) => e.tagName === "SELECT").map((e) => [e.name || e.id, e.value]),
  };
})();

console.log(JSON.stringify(global.__result));
"""


def run_in_node(html: str, inspection_js: str, fill_js: str | None = None,
                origin: str = "null") -> dict:
    """Execute the inspection (and optionally the fill) against ``html`` in Node.

    Two calls with the same ``html`` rebuild an identical DOM, so the index stamps the
    inspection produced in call one are still valid in call two -- which is how the real
    browser flow works (inspect, then fill the stamped slots).
    """
    if NODE is None:
        raise RuntimeError("node is required (it is a hard dependency of this repo)")
    script = (DOM_SHIM
              .replace("__PAGE_HTML__", json.dumps(html))
              .replace("__ORIGIN__", json.dumps(origin))
              .replace("__INSPECTION__", json.dumps(inspection_js))
              .replace("__FILL_JS__", json.dumps(fill_js)))
    p = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        raise RuntimeError(f"node failed:\n{p.stderr[:1200]}")
    data = json.loads(p.stdout.strip().splitlines()[-1])
    out = data["out"]
    if isinstance(out, str):
        out = json.loads(out)
    return {"result": out, "held": dict(data["held"]), "inspection": data["inspection"]}



def node_available() -> bool:
    return NODE is not None


def has_chromium() -> str | None:
    for exe in ("/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser"):
        if Path(exe).exists():
            return exe
    return None
