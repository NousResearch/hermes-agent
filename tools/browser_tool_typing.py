"""Native editor input for the pinned agent-browser fill implementation."""

import json

from tools import browser_tool_session as _session
from tools.browser_tool_origin import origin_module as _origin

_EDITABLE_STATE = (
    "JSON.stringify({editable:document.activeElement.isContentEditable,"
    "select:document.activeElement.tagName==='SELECT',platform:navigator.platform})"
)

# agent-browser `fill` on a native <select> reports success and changes nothing, and its
# `select` command matches option values only (also succeeding on a miss). Pick by value,
# then by the rendered label (label attribute, else text), and fail with the choices.
# Assign by index: option values need not be unique (an empty placeholder and an empty "None").
_SELECT_OPTION = """((wanted) => {
  const el = document.activeElement;
  const options = Array.from(el.options);
  const shown = (o) => (o.getAttribute('label') || o.text).trim();
  const norm = (s) => String(s).trim().replace(/\\s+/g, ' ').toLowerCase();
  const option = options.find((o) => o.value === wanted) || options.find((o) => norm(shown(o)) === norm(wanted));
  if (!option) return JSON.stringify({error: 'no option matches ' + JSON.stringify(wanted) + '; options: ' + options.map(shown).slice(0, 30).join(', ')});
  el.selectedIndex = options.indexOf(option);
  el.dispatchEvent(new Event('input', {bubbles: true}));
  el.dispatchEvent(new Event('change', {bubbles: true}));
  return JSON.stringify({ok: true});
})(%s)"""


def _select_option(task_id: str, text: str, probe: dict) -> dict:
    result = _session._run_browser_command(task_id, "eval", [_SELECT_OPTION % json.dumps(text)])
    _origin()._merge_fallback_warning(result, probe)
    if not result.get("success"):
        return result
    outcome = json.loads((result.get("data") or {}).get("result") or "{}")
    if outcome.get("error"):
        return {**result, "success": False, "error": outcome["error"]}
    return result


def fill_text(task_id: str, ref: str, text: str) -> dict:
    result = _session._run_browser_command(task_id, "focus", [ref])
    if not result.get("success"):
        return result
    probe = _session._run_browser_command(task_id, "eval", [_EDITABLE_STATE])
    _origin()._merge_fallback_warning(probe, result)
    if not probe.get("success"):
        return probe
    state = json.loads((probe.get("data") or {}).get("result") or "{}")
    if state.get("select"):
        return _select_option(task_id, text, probe)
    commands = [("fill", [ref, text])]
    if state.get("editable"):
        # Driver fill clears .value, which does not clear contentEditable. Native
        # selection/deletion also updates the editor's own document model.
        key = "Meta+a" if str(state.get("platform", "")).startswith("Mac") else "Control+a"
        commands = [("click", [ref]), ("press", [key]), ("press", ["Backspace"])]
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        for i, line in enumerate(lines):
            if i:
                # Enter creates a paragraph in framework-managed editors; a
                # single insertText containing newlines may flatten their state.
                commands.append(("press", ["Enter"]))
            if line:
                commands.append(("keyboard", ["inserttext", line]))
    result = probe
    for command, args in commands:
        step = _session._run_browser_command(task_id, command, args)
        _origin()._merge_fallback_warning(step, result)
        result = step
        if not result.get("success"):
            break
    return result
