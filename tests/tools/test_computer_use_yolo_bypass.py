"""``--yolo`` / ``approvals.mode: off`` must bypass the computer_use CLI approval prompt.

The terminal tool consults ``tools.approval.is_approval_bypass_active_for_session``
before ever calling the CLI approval callback. ``computer_use`` escalated the
cua-driver daemon to ``unrestricted`` under yolo but still routed every
destructive action (click / key / type / focus) through the CLI prompt, so a
headless ``hermes chat --yolo -q ...`` run blocked on a prompt nobody could
answer and timed out after ``approvals.timeout`` seconds per action.
"""

import json


def _install_backend(cu_tool):
    class _RecordingBackend:
        def __init__(self):
            self.calls = []

        def start(self):
            pass

        def stop(self):
            pass

        def is_available(self):
            return True

        def click(self, **kw):
            self.calls.append(("click", kw))
            from tools.computer_use.backend import ActionResult

            return ActionResult(ok=True, action="click")

    backend = _RecordingBackend()
    cu_tool.reset_backend_for_tests()
    cu_tool._backend = backend
    return backend


def test_yolo_bypasses_computer_use_approval_prompt(monkeypatch):
    from tools import approval
    from tools.computer_use import tool as cu_tool

    prompts = []

    def prompting_callback(action, args, summary):
        prompts.append(action)
        return "timeout"  # what an unattended CLI prompt returns

    cu_tool.set_approval_callback(prompting_callback)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)
    backend = _install_backend(cu_tool)

    result = cu_tool.handle_computer_use({"action": "click", "element": 3})

    assert prompts == [], "yolo run must not raise the CLI approval prompt"
    assert [c[0] for c in backend.calls] == ["click"]
    payload = json.loads(result) if isinstance(result, str) else result
    assert not (isinstance(payload, dict) and payload.get("error"))


def test_prompt_still_runs_without_bypass(monkeypatch):
    from tools import approval
    from tools.computer_use import tool as cu_tool

    prompts = []

    def denying_callback(action, args, summary):
        prompts.append(action)
        return "deny"

    cu_tool.set_approval_callback(denying_callback)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    backend = _install_backend(cu_tool)

    result = cu_tool.handle_computer_use({"action": "click", "element": 3})

    assert prompts == ["click"]
    assert backend.calls == []
    payload = json.loads(result)
    assert payload["error"] == "denied by user"
