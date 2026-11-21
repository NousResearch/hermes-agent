"""T2.3: ESC in the approval modal must deny, not silently fall through.

RED premise: ``_tui_handle_escape_modal`` handles secret/sudo/connection/slash-confirm
but not ``_approval_state``, and the ``escape`` keybind filter ``_modal_prompt_active``
excludes approvals — so ESC during an approval prompt did nothing (the modal sat until
the 300s timeout denied anyway).
"""
import inspect
import queue

from hermes_cli.cli_tui_mixin import CLITuiMixin


class _FakeBuf:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


class _FakeApp:
    def __init__(self):
        self.current_buffer = _FakeBuf()
        self.invalidated = 0

    def invalidate(self):
        self.invalidated += 1


class _FakeEvent:
    def __init__(self):
        self.app = _FakeApp()


class _Tui(CLITuiMixin):
    def __init__(self):
        self._connection_state = None
        self._secret_state = None
        self._sudo_state: dict | None = None
        self._slash_confirm_state = None
        self._approval_state = {
            "command": "rm -rf build",
            "description": "dangerous",
            "choices": ["once", "session", "always", "deny"],
            "selected": 0,
            "response_queue": queue.Queue(),
        }
        self.selection_calls = []

    def _invalidate(self):
        pass

    def _handle_approval_selection(self):
        state = self._approval_state
        if not state:
            return
        chosen = state["choices"][state["selected"]]
        self.selection_calls.append(chosen)
        state["response_queue"].put(chosen)
        self._approval_state = None


def test_escape_denies_pending_approval():
    tui = _Tui()
    event = _FakeEvent()
    tui._tui_handle_escape_modal(event)
    assert tui.selection_calls == ["deny"], "ESC must send deny"
    assert tui._approval_state is None
    assert event.app.current_buffer.reset_count == 1
    assert event.app.invalidated == 1


def test_escape_keybind_filter_covers_approval_modal():
    """The escape keybind only fires under _modal_prompt_active; approvals must be in it."""
    src = inspect.getsource(CLITuiMixin)
    cond = src.split("_modal_prompt_active = Condition")[1][:200]
    assert "_approval_state" in cond


def test_other_modals_keep_their_escape_semantics():
    tui = _Tui()
    tui._approval_state = None
    sudo: dict = {"response_queue": queue.Queue()}
    tui._sudo_state = sudo
    event = _FakeEvent()
    tui._tui_handle_escape_modal(event)
    assert tui._sudo_state is None
    assert sudo["response_queue"].get_nowait() == ""
