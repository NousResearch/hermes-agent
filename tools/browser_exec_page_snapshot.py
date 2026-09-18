"""Atomic page snapshot for ``browser_exec`` workspaces.

``_HELPERS_DIGEST`` pointed browser_exec at the accessibility-tree recipe: one
``Accessibility.getFullAXTree`` call plus one ``DOM.getBoxModel`` per control — 1 + N
protocol round trips for every look at a page. Measured against the harness's own documented
path on identical page state (local Chrome, three alternating pairs, viewport 1400x1857):

    page                        AX-tree path            snapshot()
    Wikipedia article           83 calls / 291.6 ms     1 call / 31.7 ms
    search results (DDG)        83 calls / 225.0 ms     1 call / 11.8 ms
    local fixture, 21 controls  19 calls / 24.4 ms      1 call / 4.3 ms

Coverage is not one number. Wikipedia is the flattering case: 158/160 AX-named controls
matched. DDG is 31/134 by design -- the snapshot reports visible controls (36) while the AX
tree carries everything collapsed behind its menus (150). Geometry: AX box centre inside a
snapshot entry is 74/80 on the article and 80/80 on DDG. The six article misses are
section-edit links whose AX box is 34.7 x 0.0 (zero height), not wrapper nodes. Where both
paths have a real rect, within 2 px for 63/64 on the article (worst: padded search input,
dx 13.1) and 26/27 on DDG (worst: one feedback button, dx 98.8).

Those coverage counts were taken with the cap lifted so every AX-named control could match.
The shipped default is max_elements=120 (in-view first); truncated is true when more exist.
A live browser_exec check against real Chrome returned exactly 120 controls.

This module owns that payload and installs it where browser-harness already looks: the
workspace's ``agent_helpers.py`` is auto-imported on every CLI call and its public names are
injected into the exec namespace, so ``snapshot()`` needs no CLI, fork or vendor change.
``_HELPERS_DIGEST`` keeps the cdp accessibility tree as the named fallback for what the
snapshot does not cover.

Written into the workspace:

* ``hermes_browser_snapshot.py`` — generated, version-marked, rewritten only when it changes
* ``agent_helpers.py`` — created when absent; an agent-written UTF-8 file keeps every line of
  its own content and gains one marker block. A non-UTF-8 file is left untouched.

Best-effort by design: an install failure is logged and never fails the tool call, and the
generated ``agent_helpers.py`` block guards its own import so a broken helper can never take
the harness down with it.

Attribution: an independent implementation, measured against the harness's own AX path; it
follows the shape of ``snapshot.js`` in ``browser-use/jev-ultrafast`` (MIT).

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["SNAPSHOT_HELPERS_VERSION", "HELPER_MODULE_NAME", "AGENT_HELPERS_NAME",
           "helper_source", "ensure_workspace_helpers"]

SNAPSHOT_HELPERS_VERSION = 2

_SNAPSHOT_JS = r"""
(() => {
  const MAX = __MAX__;
  const CUSTOM = __CUSTOM__;
  const D = document, W = window, t0 = performance.now();
  const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
  const norm = s => String(s == null ? '' : s).replace(/\s+/g, ' ').trim();
  const cap = (s, n) => { const v = norm(s); return v.length > n ? v.slice(0, n - 1) + '\u2026' : v; };

  // Freshness: one MutationObserver per document, idempotent. The counter lets a
  // later observation tell whether the DOM moved since this one — animation alone
  // no longer has to force a re-read.
  const S = (W.__hermesSnap = W.__hermesSnap || { mut: 0, last: 0, obs: false, seq: 0 });
  if (!S.obs) {
    try {
      const ob = new MutationObserver(ms => { S.mut += ms.length; S.last = Date.now(); });
      ob.observe(D.documentElement, { subtree: true, childList: true, attributes: true, characterData: true });
      S.obs = true;
    } catch (e) { S.obs = false; }
  }
  S.seq += 1;

  const CONTROL = new Set(['button','link','textbox','searchbox','combobox','listbox',
    'option','checkbox','radio','switch','menuitem','menuitemcheckbox','menuitemradio',
    'tab','slider','spinbutton','treeitem','gridcell','columnheader','rowheader']);
  const INPUT_ROLE = { text:'textbox', search:'searchbox', email:'textbox', url:'textbox',
    tel:'textbox', password:'textbox', number:'spinbutton', range:'slider', date:'textbox',
    'datetime-local':'textbox', month:'textbox', time:'textbox', week:'textbox',
    color:'button', file:'button', checkbox:'checkbox', radio:'radio', submit:'button',
    reset:'button', button:'button', image:'button' };

  const roleOf = el => {
    const explicit = norm(el.getAttribute('role')).toLowerCase();
    if (CONTROL.has(explicit)) return explicit;
    const tag = el.tagName.toLowerCase();
    if (tag === 'input') return INPUT_ROLE[(el.type || 'text').toLowerCase()] || 'textbox';
    if (tag === 'a') return el.hasAttribute('href') ? 'link' : '';
    if (tag === 'button') return 'button';
    if (tag === 'select') return 'combobox';
    if (tag === 'textarea') return 'textbox';
    if (tag === 'summary') return 'button';
    if (tag === 'option') return 'option';
    if (el.isContentEditable) return 'textbox';
    return '';
  };

  // --- candidates: native + explicit ARIA first, then bounded custom clickables ---
  const cands = [], seen = new Set(), forced = new WeakMap();
  const push = el => { if (!seen.has(el)) { seen.add(el); cands.push(el); } };
  D.querySelectorAll('a[href],button,input,textarea,select,summary,[contenteditable=""],[contenteditable="true"],[role]').forEach(push);

  let probed = 0, limited = false;
  if (CUSTOM) {
    const CUSTOM_LIMIT = 5000;
    const nodes = D.querySelectorAll('div,span,li,td,th,p,h1,h2,h3,h4,h5,h6,label,section,article,img,i,svg,ul,ol');
    for (const el of nodes) {
      if (el.disabled || el.hidden) continue;
      const clickish = el.onclick !== null || el.hasAttribute('onclick') ||
        el.hasAttribute('jsaction') || el.hasAttribute('data-action') || el.hasAttribute('aria-haspopup');
      let isCand = clickish;
      if (!isCand) {
        if (probed >= CUSTOM_LIMIT) { limited = true; continue; }
        probed++;
        try {
          const cs = getComputedStyle(el);
          isCand = cs.cursor === 'pointer' && !!(el.innerText || '').trim();
        } catch (e) { isCand = false; }
      }
      // A custom candidate is a control even though its tag carries no role — without this
      // the role filter in the main loop silently drops every div/li click target.
      if (isCand && !seen.has(el)) { push(el); forced.set(el, 'button'); }
    }
  }
  // Collapse nesting: keep the outermost candidate of a custom cluster.
  const kept = cands.filter(el => {
    let p = el.parentElement, depth = 0;
    while (p && depth < 12) {
      if (seen.has(p) && p !== el) {
        const nativeParent = p.hasAttribute('href') || ['BUTTON','INPUT','SELECT','TEXTAREA','SUMMARY','OPTION'].includes(p.tagName);
        if (nativeParent || p.onclick !== null || p.hasAttribute('onclick') || p.hasAttribute('jsaction')) return false;
      }
      p = p.parentElement; depth++;
    }
    return true;
  });

  const textOf = el => { let t = norm(el.innerText); if (!t) t = norm(el.textContent); return t; };
  const nameOf = el => {
    const by = el.getAttribute('aria-labelledby');
    if (by) {
      const parts = by.split(/\s+/).map(id => { const r = D.getElementById(id); return r ? norm(r.innerText || r.textContent) : ''; }).filter(Boolean);
      if (parts.length) return { name: cap(parts.join(' '), 120), src: 'aria-labelledby' };
    }
    const al = norm(el.getAttribute('aria-label'));
    if (al) return { name: cap(al, 120), src: 'aria-label' };
    const tag = el.tagName.toLowerCase();
    if (tag === 'input' || tag === 'select' || tag === 'textarea') {
      if (el.id) {
        try {
          const lab = D.querySelector('label[for="' + CSS.escape(el.id) + '"]');
          if (lab) { const t = norm(lab.innerText || lab.textContent); if (t) return { name: cap(t, 120), src: 'label[for]' }; }
        } catch (e) { /* invalid id selector */ }
      }
      const wrap = el.closest('label');
      if (wrap) { const t = norm(wrap.innerText || wrap.textContent); if (t) return { name: cap(t, 120), src: 'label-wrapping' }; }
      const alt = norm(el.getAttribute('alt'));
      if (alt) return { name: cap(alt, 120), src: 'alt' };
      const ty = (el.type || '').toLowerCase();
      if ((ty === 'submit' || ty === 'reset' || ty === 'button') && norm(el.value)) return { name: cap(el.value, 120), src: 'value' };
      const ph = norm(el.getAttribute('placeholder'));
      if (ph) return { name: cap(ph, 120), src: 'placeholder' };
      const ti = norm(el.getAttribute('title'));
      if (ti) return { name: cap(ti, 120), src: 'title' };
      const nm = norm(el.getAttribute('name'));
      if (nm) return { name: cap(nm, 120), src: 'name-attr' };
      return { name: '', src: '' };
    }
    const t = textOf(el);
    if (t) return { name: cap(t, 120), src: 'text' };
    // Accessible name often lives in a descendant for image/icon controls — Wikipedia's
    // wordmark link is <a><span class="mw-logo-wordmark"> with the name in the <img alt>.
    let inner = null;
    try { inner = el.querySelector('[aria-label], img[alt], [alt], [title]'); } catch (e) { inner = null; }
    if (inner) {
      const it = norm(inner.getAttribute('aria-label') || inner.getAttribute('alt') || inner.getAttribute('title'));
      if (it) return { name: cap(it, 120), src: 'descendant' };
    }
    const ti2 = norm(el.getAttribute('title'));
    if (ti2) return { name: cap(ti2, 120), src: 'title' };
    const al2 = norm(el.getAttribute('alt'));
    if (al2) return { name: cap(al2, 120), src: 'alt' };
    const aria = norm(el.getAttribute('value'));
    if (aria) return { name: cap(aria, 120), src: 'value' };
    return { name: '', src: '' };
  };

  const valueOf = el => {
    const out = {};
    const tag = el.tagName.toLowerCase(), ty = (el.type || '').toLowerCase();
    if (tag === 'select') {
      out.value = cap(el.value, 120);
      const opts = Array.from(el.options || []);
      out.option_count = opts.length;
      out.options = opts.slice(0, 25).map((o, i) => ({ i, label: cap(o.text, 60), value: cap(o.value, 60), selected: !!o.selected, disabled: !!o.disabled }));
      if (opts.length > 25) out.options_truncated = true;
      out.selected_label = cap((opts.find(o => o.selected) || {}).text || '', 80);
    } else if (tag === 'input' || tag === 'textarea') {
      if (ty === 'password') { out.value = '<password:redacted>'; out.value_len = String(el.value || '').length; }
      else if (ty === 'checkbox' || ty === 'radio') { out.checked = !!el.checked; }
      else out.value = cap(el.value, 200);
      const ph = norm(el.getAttribute('placeholder'));
      if (ph) out.placeholder = cap(ph, 80);
      if (el.maxLength > 0) out.maxlength = el.maxLength;
    } else if (el.isContentEditable) {
      out.value = cap(el.textContent, 200);
    } else if (tag === 'a') {
      out.href = cap(el.href, 200);
    }
    return out;
  };

  const cw = D.documentElement.clientWidth, ch = D.documentElement.clientHeight;
  const entries = [];
  for (const el of kept) {
    const role = roleOf(el) || forced.get(el) || '';
    if (!role) continue;
    let r; try { r = el.getBoundingClientRect(); } catch (e) { continue; }
    if (!(r.width >= 1 && r.height >= 1)) continue;
    let cs = null; try { cs = getComputedStyle(el); } catch (e) { cs = null; }
    if (cs && (cs.display === 'none' || cs.visibility === 'hidden' || cs.visibility === 'collapse')) continue;
    if (el.hidden) continue;
    let inert = null; try { inert = el.closest('[inert]'); } catch (e) { inert = null; }
    if (inert) continue;

    const nm = nameOf(el);
    const inView = r.bottom > 0 && r.top < ch && r.right > 0 && r.left < cw;
    const e = {
      i: 0, tag: el.tagName.toLowerCase(), role,
      origin: forced.has(el) ? 'custom' : (norm(el.getAttribute('role')) ? 'aria' : 'native'),
      name: nm.name, name_src: nm.src, id: cap(el.id, 60),
      type: norm(el.getAttribute('type')).toLowerCase(),
      disabled: !!(el.disabled || el.getAttribute('aria-disabled') === 'true'),
      readonly: el.readOnly === true, required: el.required === true,
      // opacity:0 controls are still hit-testable in Chrome (icon buttons behind a drawn
      // sibling), so they are reported with a flag rather than dropped. Measured: Wikipedia's
      // Vector header buttons are input[type=button] at opacity 0.
      transparent: cs ? parseFloat(cs.opacity || '1') === 0 : undefined,
      clicks_through: cs ? cs.pointerEvents === 'none' : undefined,
      in_view: inView,
      x: Math.round(r.left), y: Math.round(r.top), w: Math.round(r.width), h: Math.round(r.height),
      cx: Math.round(r.left + r.width / 2), cy: Math.round(r.top + r.height / 2),
    };
    Object.assign(e, valueOf(el));
    if (inView) {
      let hit = null;
      try { hit = D.elementFromPoint(clamp(r.left + r.width / 2, 1, cw - 1), clamp(r.top + r.height / 2, 1, ch - 1)); } catch (err) { hit = null; }
      if (hit && hit !== el && !el.contains(hit) && !hit.contains(el)) {
        const cls = (typeof hit.className === 'string' && hit.className) ? '.' + cap(hit.className.split(/\s+/).slice(0, 2).join('.'), 30) : '';
        e.covered_by = hit.tagName.toLowerCase() + (hit.id ? '#' + cap(hit.id, 30) : '') + cls + (norm(hit.innerText) ? ' \u2014 ' + cap(hit.innerText, 40) : '');
      }
    }
    e.sig = [e.tag, e.role, e.name.slice(0, 48), e.id, e.w + 'x' + e.h].join('|');
    entries.push(e);
  }

  const total = entries.length;
  let out = entries;
  if (total > MAX) {
    out = entries.filter(e => e.in_view).concat(entries.filter(e => !e.in_view)).slice(0, MAX);
  }
  out.forEach((e, i) => { e.i = i + 1; });

  return JSON.stringify({
    ok: true, url: location.href, title: cap(D.title, 160), readyState: D.readyState,
    vw: cw, vh: ch, dpr: W.devicePixelRatio || 1,
    sx: Math.round(W.scrollX), sy: Math.round(W.scrollY),
    pw: D.documentElement.scrollWidth, ph: D.documentElement.scrollHeight,
    mut: S.mut, seq: S.seq, page_ms: Math.round(performance.now() - t0),
    count: out.length, total, truncated: total > out.length,
    custom_probed: probed, custom_limited: limited,
    elements: out,
  });
})()
"""

_STATE_JS = r"""
(() => {
  const S = window.__hermesSnap || { mut: 0, seq: 0 };
  return JSON.stringify({ url: location.href, title: document.title, readyState: document.readyState,
    mut: S.mut, seq: S.seq, sy: Math.round(window.scrollY),
    interactive: document.querySelectorAll('a[href],button,input,textarea,select,summary,[role],[contenteditable=""],[contenteditable="true"]').length });
})()
"""

HELPER_MODULE_NAME = "hermes_browser_snapshot.py"
AGENT_HELPERS_NAME = "agent_helpers.py"
QUOTE = "'" * 3
NL = "\n"  # assembled: a literal triple-quote would close the raw block above

_MARKER_BEGIN = "# >>> hermes-browser-snapshot >>>"
_MARKER_END = "# <<< hermes-browser-snapshot <<<"


_HELPER_HEAD = r'''"""Atomic page snapshot — generated by Hermes (tools/browser_exec_page_snapshot.py).

Do not edit this file: Hermes rewrites it when SNAPSHOT_HELPERS_VERSION changes. Put your own
browser helpers in agent_helpers.py next to it — that file is yours and only gains or refreshes
the marker block that imports this module.

`snapshot()` reads every visible interactive control in ONE browser round trip and returns an
indexed table: role, name, value/state, geometry in viewport CSS px, plus the three flags that
matter when acting on it — `covered_by` is occlusion (whatever sits on the control's centre;
absent when the centre belongs to the control or a descendant), `clicks_through` is CSS
`pointer-events: none`, and `transparent` marks `opacity: 0` (Chrome still hit-tests those).
Default cap is 120 (`truncated` is true when more exist).

    snap = snapshot()                  # exactly one protocol round trip
    print(snapshot_table(snap))
    hits = find_entry(snap, "sign in")
    if hits:
        ensure_real_tab()              # clicks miss if this tab is not the real one
        click_at_xy(*point(hits[0]))

Limits (the same ones the upstream MVP declares): common HTML/ARIA controls only. Open shadow
roots, iframes, canvas, new tabs and nested scrolling are out of scope. File inputs still
appear as buttons; choosing a file is not covered. `covered_by` is a hint, not a click
guarantee.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["snapshot", "snapshot_table", "find_entry", "point", "page_changed", "SNAPSHOT_JS",
           "STATE_JS", "_MAX_DEFAULT"]

_MAX_DEFAULT = 120

SNAPSHOT_JS = r@OPEN@
'''

_HELPER_MID = r'''
@CLOSE@

STATE_JS = r@OPEN@
'''

_HELPER_TAIL = r'''
@CLOSE@


def _js(expression: str) -> Any:
    """Deferred import: browser_harness.helpers loads this module while it imports."""
    from browser_harness.helpers import js as _h_js
    return _h_js(expression)


def _loads(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, dict):
        return raw
    raise RuntimeError(f"snapshot: unexpected JS return type {type(raw).__name__}")


def snapshot(max_elements: int = _MAX_DEFAULT, custom: bool = True) -> Dict[str, Any]:
    """Observe the page in ONE round trip. Returns the indexed element table."""
    expr = (SNAPSHOT_JS
            .replace("__MAX__", str(int(max_elements)))
            .replace("__CUSTOM__", "true" if custom else "false"))
    snap = _loads(_js(expr))
    snap["elements"] = [e for e in snap.get("elements", []) if isinstance(e, dict)]
    return snap


def snapshot_table(snap: Dict[str, Any], limit: Optional[int] = None, brief: bool = False) -> str:
    """Render the observed table as text: [i] role name flags = value -> href."""
    els: List[Dict[str, Any]] = snap.get("elements", [])
    head = (f"page: {str(snap.get('title', ''))[:80]} \u2014 {snap.get('url', '')}\n"
            f"      {snap.get('count')}/{snap.get('total')} controls"
            f"{' (truncated)' if snap.get('truncated') else ''}, mut={snap.get('mut')} "
            f"seq={snap.get('seq')}, viewport {snap.get('vw')}x{snap.get('vh')}@{snap.get('dpr')} "
            f"scroll={snap.get('sx')},{snap.get('sy')} page_ms={snap.get('page_ms')}")
    rows = els if limit is None else els[:limit]
    lines = []
    for e in rows:
        val = e.get("value")
        if val in (None, "") and e.get("checked") is not None:
            val = f"checked={e['checked']}"
        if val is None and e.get("selected_label"):
            val = e["selected_label"]
        tail = ""
        if e.get("disabled"):
            tail += " [disabled]"
        if not e.get("in_view"):
            tail += " [off-screen]"
        if e.get("covered_by"):
            tail += f" [covered: {str(e['covered_by'])[:48]}]"
        if e.get("origin") == "custom":
            tail += " [custom]"
        if e.get("transparent"):
            tail += " [invisible-opacity0]"
        if e.get("clicks_through"):
            tail += " [pointer-events:none]"
        if not brief and val not in (None, ""):
            tail += f"  = {str(val)[:60]}"
        elif not brief and e.get("href"):
            tail += f"  -> {str(e['href'])[:60]}"
        lines.append(f"[{e.get('i'):>3}] {str(e.get('role', '')):<9} "
                     f"{str(e.get('name', ''))[:44]:<44}{tail}")
    if limit is not None and len(els) > limit:
        lines.append(f"      \u2026 {len(els) - limit} more")
    return "\n".join([head] + lines)


def find_entry(snap: Dict[str, Any], needle: str, role: Optional[str] = None,
               limit: int = 10) -> List[Dict[str, Any]]:
    """Substring search over the observed table — pure Python, no extra round trip."""
    needle = (needle or "").lower()
    hits = []
    for e in snap.get("elements", []):
        if role and e.get("role") != role:
            continue
        if needle and needle not in str(e.get("name", "")).lower():
            continue
        hits.append(e)
        if len(hits) >= limit:
            break
    return hits


def point(entry: Dict[str, Any]) -> Tuple[int, int]:
    """Click coordinates for an observed entry — viewport px, as click_at_xy expects."""
    return int(entry["cx"]), int(entry["cy"])


def page_changed(snap: Dict[str, Any]) -> Dict[str, Any]:
    """ONE extra round trip: has the DOM moved since `snap` was taken?"""
    st = _loads(_js(STATE_JS))
    st["changed"] = int(st.get("mut", 0)) != int(snap.get("mut", 0))
    st["mut_delta"] = int(st.get("mut", 0)) - int(snap.get("mut", 0))
    st["url_changed"] = st.get("url") != snap.get("url")
    st["scrolled"] = st.get("sy") != snap.get("sy")
    return st
'''

def _sub(text: str) -> str:
    """@OPEN@/@CLOSE@ -> triple quotes: the helper file is generated, not vendored."""
    return text.replace("@OPEN@", QUOTE).replace("@CLOSE@", QUOTE)


def helper_source() -> str:
    """Full text of the generated workspace helper (JS payload + Python API)."""
    return _sub(_HELPER_HEAD) + _SNAPSHOT_JS + _sub(_HELPER_MID) + _STATE_JS + _sub(_HELPER_TAIL)


def _agent_helpers_block() -> str:
    """The marker block appended to (or refreshed inside) the workspace's agent_helpers.py.

    The import is guarded: a helper problem must never take every browser_exec call with it.
    The sys.path insert makes the module importable when the harness loads agent_helpers.py
    by file path from a directory that is not on sys.path.
    """
    return (
        f"{_MARKER_BEGIN}\n"
        f"# Hermes browser helpers v{SNAPSHOT_HELPERS_VERSION} — your own helpers above and\n"
        "# below this block are preserved; only the lines between the markers are refreshed.\n"
        "import os as _hermes_helpers_os\n"
        "import sys as _hermes_helpers_sys\n"
        "_hermes_helpers_dir = _hermes_helpers_os.path.dirname(\n"
        "    _hermes_helpers_os.path.abspath(__file__))\n"
        "if _hermes_helpers_dir not in _hermes_helpers_sys.path:\n"
        "    _hermes_helpers_sys.path.insert(0, _hermes_helpers_dir)\n"
        "try:\n"
        "    from hermes_browser_snapshot import *  # noqa: F401,F403\n"
        "except Exception:  # a broken helper must not break the harness\n"
        "    pass\n"
        f"{_MARKER_END}\n"
    )



def _replace_block(text: str, block: str) -> str:
    """Swap the marker block in `text` for `block`, leaving everything else untouched.

    A stable marker (no version inside it) is what makes an upgrade repair in place rather than
    append a second block; a half-deleted block is repaired by re-appending at its start.
    """
    begin = text.find(_MARKER_BEGIN)
    if begin == -1:
        return text.rstrip(NL) + NL + NL + block
    end = text.find(_MARKER_END, begin)
    tail = "" if end == -1 else text[end + len(_MARKER_END):].lstrip(NL)
    return text[:begin].rstrip(NL) + NL + NL + block + tail


def _write(path, text: str) -> None:
    """Symlink-safe write (tools/spill_safety): never follows a planted link.

    Imported here like every other spill caller, so a reader of this module pulls in nothing.
    """
    from tools.spill_safety import write_text_exclusive

    write_text_exclusive(path, text, private=False, overwrite=True, encoding="utf-8")


def ensure_workspace_helpers(workspace: Optional[str]) -> Optional[str]:
    """Install/refresh the snapshot helpers in `workspace`; return the helper path or None.

    Idempotent, symlink-safe and best-effort — this never raises and never fails a tool call.
    """
    if not workspace:
        return None
    try:
        ws = Path(workspace)
        ws.mkdir(parents=True, exist_ok=True)
        helper = ws / HELPER_MODULE_NAME
        source = helper_source()
        try:
            current = helper.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            current = None
        if current != source:
            _write(helper, source)

        agent = ws / AGENT_HELPERS_NAME
        try:
            agent_text = agent.read_text(encoding="utf-8")
        except FileNotFoundError:
            agent_text = None
        except UnicodeDecodeError:
            logger.debug("browser_exec: %s is not UTF-8; leaving it in place", agent)
            return str(helper)
        except OSError:
            agent_text = None
        block = _agent_helpers_block()
        if agent_text is None:
            _write(agent, block)
        else:
            refreshed = _replace_block(agent_text, block)
            if refreshed != agent_text:
                _write(agent, refreshed)
        return str(helper)
    except Exception as exc:  # pragma: no cover - best effort, never fatal
        logger.debug("browser_exec: snapshot helper install skipped: %s", exc)
        return None
