"""Framework-aware input sync for browser_type.

agent-browser's ``fill`` command uses CDP ``Input.insertText``, which updates
the DOM ``value`` but does not notify React/Vue/Angular controlled components.
After fill succeeds, we focus the target ref and re-apply the value with the
native prototype setter plus bubbling ``input``/``change`` events — the same
pattern Playwright uses for ``fill()`` on controlled inputs.
"""

from __future__ import annotations

import json
from typing import Any


def build_controlled_input_sync_expression(text: str) -> str:
    """Return a self-invoking JS expression for the focused input element."""
    text_json = json.dumps(text)
    return f"""(() => {{
  const text = {text_json};
  const el = document.activeElement;
  if (!el) return {{ok: false, reason: "no active element"}};
  const tag = (el.tagName || "").toUpperCase();
  if (tag !== "INPUT" && tag !== "TEXTAREA") {{
    return {{ok: false, reason: "active element is not a text input", tag: tag}};
  }}
  const proto = tag === "TEXTAREA" ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  const setter = Object.getOwnPropertyDescriptor(proto, "value")?.set;
  if (setter) setter.call(el, text);
  else el.value = text;
  try {{
    el.dispatchEvent(new InputEvent("input", {{
      bubbles: true,
      cancelable: true,
      inputType: "insertText",
      data: text,
    }}));
  }} catch (_err) {{
    el.dispatchEvent(new Event("input", {{ bubbles: true }}));
  }}
  el.dispatchEvent(new Event("change", {{ bubbles: true }}));
  return {{ok: true, value: el.value}};
}})()"""


def sync_controlled_input_value(ref: str, text: str, task_id: str) -> dict[str, Any]:
    """Focus *ref* and dispatch framework-friendly input events for *text*."""
    from tools.browser_tool import _browser_eval, _run_browser_command

    focus = _run_browser_command(task_id, "focus", [ref])
    if not focus.get("success"):
        return {
            "ok": False,
            "error": focus.get("error") or "focus failed before framework input sync",
        }

    eval_raw = _browser_eval(build_controlled_input_sync_expression(text), task_id)
    try:
        eval_payload = json.loads(eval_raw)
    except json.JSONDecodeError:
        return {"ok": False, "error": "framework input sync returned invalid JSON"}

    if not eval_payload.get("success"):
        return {
            "ok": False,
            "error": eval_payload.get("error") or "framework input sync eval failed",
        }

    result = eval_payload.get("result")
    if isinstance(result, dict) and result.get("ok"):
        return {"ok": True}

    if isinstance(result, dict):
        return {
            "ok": False,
            "error": result.get("reason") or "framework input sync rejected active element",
        }

    return {"ok": False, "error": "framework input sync returned unexpected result"}
