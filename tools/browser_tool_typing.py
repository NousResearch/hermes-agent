"""Native editor input for the pinned agent-browser fill implementation."""

import json

from tools import browser_tool_session as _session
from tools.browser_tool_origin import origin_module as _origin

_EDITABLE_STATE = "JSON.stringify({editable:document.activeElement.isContentEditable,platform:navigator.platform})"


def fill_text(task_id: str, ref: str, text: str) -> dict:
    result = _session._run_browser_command(task_id, "focus", [ref])
    if not result.get("success"):
        return result
    probe = _session._run_browser_command(task_id, "eval", [_EDITABLE_STATE])
    _origin()._merge_fallback_warning(probe, result)
    if not probe.get("success"):
        return probe
    state = json.loads((probe.get("data") or {}).get("result") or "{}")
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
