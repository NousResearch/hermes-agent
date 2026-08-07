"""Tests for refactored claude_session modules: task_context, idle, session."""

import pytest
from unittest.mock import MagicMock, patch
from tools.claude_session.task_context import TaskContext, FileContext
from tools.claude_session.idle import (
    SessionState, clean_lines, detect_state, detect_activity,
    detect_startup_scene, strip_ansi, is_permission_in_text,
)
from tools.claude_session.session import ClaudeSession, _StateView
from tools.claude_session.errors import SessionNotActiveError


# ---------------------------------------------------------------------------
# TaskContext tests
# ---------------------------------------------------------------------------

class TestTaskContext:
    def test_basic_prompt(self):
        tc = TaskContext(task_description="Fix login bug")
        prompt = tc.to_prompt()
        assert "Fix login bug" in prompt
        assert "## Task" in prompt

    def test_full_prompt(self):
        tc = TaskContext(
            task_description="Implement auth",
            file_contexts=[
                FileContext(path="auth.py", content="def login(): pass", description="Auth module"),
            ],
            constraints=["Must not break existing tests"],
            acceptance_criteria=["Login works with valid credentials"],
            project_conventions="Use snake_case for functions",
        )
        prompt = tc.to_prompt()
        assert "## Task" in prompt
        assert "## Relevant Files" in prompt
        assert "auth.py" in prompt
        assert "def login(): pass" in prompt
        assert "## Constraints" in prompt
        assert "## Acceptance Criteria" in prompt
        assert "## Project Conventions" in prompt

    def test_empty_optional_fields(self):
        tc = TaskContext(task_description="Simple task")
        prompt = tc.to_prompt()
        assert "Relevant Files" not in prompt
        assert "Constraints" not in prompt
        assert "Acceptance Criteria" not in prompt
        assert "Project Conventions" not in prompt


# ---------------------------------------------------------------------------
# idle.py tests
# ---------------------------------------------------------------------------

class TestIdleDetection:
    def test_empty_input(self):
        result = detect_state([])
        assert result.state == SessionState.THINKING

    def test_idle_with_welcome_screen(self):
        lines = ["Welcome to Claude", "Tips for getting started", "❯"]
        result = detect_state(lines)
        assert result.state == SessionState.IDLE

    def test_idle_with_done_marker(self):
        lines = ["✻ Churned for 2m 57s", "─────────", "❯", "─────────", "⏵⏵"]
        result = detect_state(lines)
        assert result.state == SessionState.IDLE

    def test_tool_call(self):
        lines = ["● Read file.py"]
        result = detect_state(lines)
        assert result.state == SessionState.TOOL_CALL
        assert result.tool_name == "Read"
        assert result.tool_target == "file.py"

    def test_tool_call_paren(self):
        lines = ["● Bash(npm test)"]
        result = detect_state(lines)
        assert result.state == SessionState.TOOL_CALL
        assert result.tool_name == "Bash"

    def test_permission(self):
        lines = ["Allow Bash command?", "❯ 1. Yes", "  2. No"]
        result = detect_state(lines)
        assert result.state == SessionState.PERMISSION

    def test_shell_prompt_is_exited(self):
        lines = ["some output", "❯"]
        result = detect_state(lines)
        assert result.state == SessionState.EXITED

    def test_status_bar_prompt_is_idle(self):
        """v2.1.126 renders ❯ + ── + status bar at IDLE — not phantom."""
        lines = ["─────────", "❯", "─────────", "⏵⏵ bypass permissions on"]
        result = detect_state(lines)
        assert result.state == SessionState.IDLE

    def test_phantom_prompt_without_status_bar(self):
        """Separator + prompt + separator without status bar = phantom (THINKING)."""
        lines = ["─────────", "❯", "─────────"]
        result = detect_state(lines)
        assert result.state == SessionState.THINKING

    def test_compacting(self):
        lines = ["Compacting conversation..."]
        result = detect_state(lines)
        assert result.state == SessionState.THINKING
        assert result.is_compacting

    def test_stale_tool_marker(self):
        lines = ["● Read old.py", "✻ Churned for 1m", "Welcome to Claude", "❯"]
        result = detect_state(lines)
        assert result.state == SessionState.IDLE


class TestActivityDetection:
    def test_reading(self):
        result = detect_activity(["● Read file.py"])
        assert result["activity"] == "reading"

    def test_writing(self):
        result = detect_activity(["● Edit file.py"])
        assert result["activity"] == "writing"

    def test_executing(self):
        result = detect_activity(["● Bash(npm test)"])
        assert result["activity"] == "executing"

    def test_searching(self):
        result = detect_activity(["● Grep pattern"])
        assert result["activity"] == "searching"

    def test_idle_no_markers(self):
        result = detect_activity(["some text", "more text"])
        assert result["activity"] == "idle"

    def test_empty(self):
        result = detect_activity([])
        assert result["activity"] == "idle"


class TestCleanLines:
    def test_strips_ansi(self):
        raw = "\x1b[32mgreen text\x1b[0m\nnormal"
        lines = clean_lines(raw)
        assert lines == ["green text", "normal"]

    def test_removes_empty(self):
        lines = clean_lines("hello\n\nworld\n")
        assert lines == ["hello", "world"]


class TestPermissionDetection:
    def test_real_permission(self):
        assert is_permission_in_text("Allow Bash command?\n❯ 1. Yes")

    def test_status_bar_not_permission(self):
        assert not is_permission_in_text("⏵⏵ bypass permissions on\nshift+tab to cycle")


class TestStartupScene:
    def test_workspace_trust(self):
        lines = ["Quick safety check", "Enter to confirm"]
        scene = detect_startup_scene(lines)
        assert scene is not None
        assert scene.scene_type == "workspace_trust"

    def test_no_scene(self):
        assert detect_startup_scene(["normal output"]) is None


# ---------------------------------------------------------------------------
# ClaudeSession tests (unit tests, no tmux required)
# ---------------------------------------------------------------------------

class TestClaudeSessionBasic:
    def test_initial_state(self):
        s = ClaudeSession()
        assert s._state == SessionState.DISCONNECTED
        assert s._session_active is False

    def test_sm_compat(self):
        s = ClaudeSession()
        sv = s._sm
        assert sv.current_state == SessionState.DISCONNECTED
        assert sv.state_duration() > 0

    def test_backward_compat_alias(self):
        from tools.claude_session import ClaudeSessionManager
        assert ClaudeSessionManager is ClaudeSession

    def test_status_no_session(self):
        s = ClaudeSession()
        result = s.status()
        assert result["state"] == SessionState.DISCONNECTED

    def test_send_no_session(self):
        s = ClaudeSession()
        with pytest.raises(SessionNotActiveError):
            s.send("hello")

    def test_send_task_context_no_session(self):
        s = ClaudeSession()
        tc = TaskContext(task_description="test")
        with pytest.raises(SessionNotActiveError):
            s.send(tc)

    def test_stop_no_session(self):
        s = ClaudeSession()
        with pytest.raises(SessionNotActiveError):
            s.stop()

    def test_output_no_session(self):
        s = ClaudeSession()
        result = s.output()
        assert result["lines"] == []

    def test_history_simplified(self):
        s = ClaudeSession()
        result = s.history()
        assert result["total_turns"] == 0


# ---------------------------------------------------------------------------
# Interview state detection tests
# ---------------------------------------------------------------------------

class TestInterviewDetection:
    """Tests for INTERVIEW state detection in idle.py."""

    def test_interview_with_navigation_hints(self):
        """Interview mode with navigation hints + numbered options."""
        lines = [
            "←  ☐ 运行环境  ☐ M1 范围  ✔ Submit  →",
            "❯ 1. WSL 内运行",
            "2. 混合环境",
            "3. 都用 Windows",
            "Enter to select · Tab/Arrow keys to navigate · Esc to cancel",
        ]
        result = detect_state(lines)
        assert result.state == SessionState.INTERVIEW

    def test_interview_with_sections_only(self):
        """Interview mode with checkbox sections and numbered options."""
        lines = [
            "←  ☐ 运行环境  ☐ M1 范围  ✔ Submit  →",
            "Ready to submit your answers?",
            "❯ 1. Submit answers",
            "2. Cancel ⋯",
        ]
        result = detect_state(lines)
        assert result.state == SessionState.INTERVIEW

    def test_interview_with_numbered_options_and_nav(self):
        """Interview mode: numbered options + nav hints with ❯ cursor."""
        lines = [
            "❯ 1. WSL 内运行",
            "2. 混合环境",
            "3. 都用 Windows",
            "4. Type something.",
            "Enter to select · Tab/Arrow keys to navigate · Esc to cancel",
        ]
        result = detect_state(lines)
        assert result.state == SessionState.INTERVIEW

    def test_interview_not_confused_with_permission(self):
        """Permission selector should not be detected as INTERVIEW."""
        lines = [
            "Allow Bash command?",
            "❯ 1. Yes",
            "  2. No",
        ]
        result = detect_state(lines)
        assert result.state == SessionState.PERMISSION

    def test_interview_not_confused_with_idle(self):
        """Interview mode should NOT return IDLE."""
        lines = [
            "❯ 1. WSL 内运行",
            "2. 混合环境",
            "Enter to select · Tab/Arrow keys to navigate · Esc to cancel",
        ]
        result = detect_state(lines)
        assert result.state == SessionState.INTERVIEW
        assert result.state != SessionState.IDLE

    def test_interview_not_triggered_without_nav_or_sections(self):
        """Numbered options alone without nav hints or sections should not trigger INTERVIEW."""
        lines = [
            "Some output",
            "❯ 1. First option",
            "2. Second option",
        ]
        result = detect_state(lines)
        # Should not be INTERVIEW (no nav hints, no section markers)
        assert result.state != SessionState.INTERVIEW

    def test_bare_type_something_not_false_positive(self):
        """Bare 'Type something.' in prose should not trigger INTERVIEW."""
        lines = [
            "To fix this, type something. Then press Enter.",
            "❯ 1. Option A",
        ]
        result = detect_state(lines)
        assert result.state != SessionState.INTERVIEW

    def test_numbered_explanations_not_false_positive(self):
        """Claude's numbered explanations without ❯ cursor or nav hints should not trigger."""
        lines = [
            "Here are the steps:",
            "1. First step",
            "2. Second step",
            "3. Third step",
            "Enter to select · Tab/Arrow keys to navigate",
        ]
        result = detect_state(lines)
        assert result.state != SessionState.INTERVIEW

    def test_interview_with_ctrl_o_hint(self):
        """Interview mode with ctrl+o hint."""
        lines = [
            "❯ 1. Option A",
            "2. Option B",
            "ctrl+o to expand",
        ]
        result = detect_state(lines)
        assert result.state == SessionState.INTERVIEW

    def test_interview_with_type_something_hint(self):
        """Interview mode with numbered 'Type something.' hint."""
        lines = [
            "❯ 1. Chat about this",
            "2. Skip interview and plan immediately",
            "4. Type something.",
        ]
        result = detect_state(lines)
        assert result.state == SessionState.INTERVIEW

    def test_interview_has_higher_priority_than_tool_call(self):
        """INTERVIEW should be detected before TOOL_CALL when both patterns exist."""
        lines = [
            "● Read some_file.py",
            "←  ☐ Question  ✔ Submit  →",
            "❯ 1. Option A",
            "2. Option B",
        ]
        result = detect_state(lines)
        assert result.state == SessionState.INTERVIEW

    def test_permission_has_higher_priority_than_interview(self):
        """PERMISSION should be detected before INTERVIEW."""
        lines = [
            "Allow Bash command?",
            "❯ 1. Yes",
            "  2. No",
            "Enter to select · Tab/Arrow keys to navigate",
        ]
        result = detect_state(lines)
        assert result.state == SessionState.PERMISSION


# ---------------------------------------------------------------------------
# Interview context building tests
# ---------------------------------------------------------------------------

class TestInterviewContext:
    """Tests for _build_interview_context in session.py."""

    def test_build_context_extracts_options(self):
        s = ClaudeSession()
        lines = [
            "←  ☐ 运行环境  ☐ M1 范围  ✔ Submit  →",
            "❯ 1. WSL 内运行",
            "2. 混合环境",
            "3. 都用 Windows",
            "Enter to select · Tab/Arrow keys to navigate · Esc to cancel",
        ]
        ctx = s._build_interview_context(lines)
        assert ctx["state"] == SessionState.INTERVIEW
        assert ctx["needs_hermes_decision"] is True
        assert len(ctx["options"]) == 3
        assert ctx["options"][0]["number"] == 1
        assert ctx["options"][0]["text"] == "WSL 内运行"
        assert ctx["options"][0]["selected"] is True
        assert ctx["options"][1]["selected"] is False

    def test_build_context_extracts_sections(self):
        s = ClaudeSession()
        lines = [
            "←  ☐ 运行环境  ☐ M1 范围  ✔ Submit  →",
            "❯ 1. Submit answers",
            "2. Cancel ⋯",
        ]
        ctx = s._build_interview_context(lines)
        assert len(ctx["sections"]) >= 3  # at least 运行环境, M1 范围, Submit
        checked = [sec for sec in ctx["sections"] if sec["checked"]]
        unchecked = [sec for sec in ctx["sections"] if not sec["checked"]]
        assert len(checked) >= 1  # Submit is ✔
        assert len(unchecked) >= 1  # ☐ sections
        # Verify multi-word labels are captured fully (not truncated)
        labels = [sec["label"] for sec in ctx["sections"]]
        assert any("M1 范围" in label for label in labels), f"Expected 'M1 范围' in {labels}"

    def test_build_context_extracts_question(self):
        s = ClaudeSession()
        lines = [
            "What is your environment?",
            "❯ 1. WSL 内运行",
            "2. 混合环境",
            "Enter to select",
        ]
        ctx = s._build_interview_context(lines)
        assert "What is your environment?" in ctx["question"]

    def test_build_context_question_not_confused_with_numbered_option(self):
        """Numbered options containing '?' should not be picked as the question."""
        s = ClaudeSession()
        lines = [
            "What is your environment?",
            "❯ 1. WSL 内运行",
            "2. Use Windows?",
            "Enter to select",
        ]
        ctx = s._build_interview_context(lines)
        assert "What is your environment?" in ctx["question"]
        assert "Use Windows?" not in ctx["question"]

    def test_build_context_empty_lines(self):
        s = ClaudeSession()
        ctx = s._build_interview_context(None)
        assert ctx["state"] == SessionState.INTERVIEW
        assert ctx["options"] == []
        assert ctx["sections"] == []

    def test_build_context_no_question(self):
        s = ClaudeSession()
        lines = [
            "←  ☐ Config  ✔ Submit  →",
            "❯ 1. Submit",
        ]
        ctx = s._build_interview_context(lines)
        assert ctx["question"] == ""


# ---------------------------------------------------------------------------
# Interview response tests
# ---------------------------------------------------------------------------

class TestRespondInterview:
    """Tests for respond_interview in session.py."""

    def test_respond_no_session_raises(self):
        s = ClaudeSession()
        with pytest.raises(SessionNotActiveError):
            s.respond_interview("1")

    def test_respond_wrong_state_raises(self):
        """respond_interview should reject when not in INTERVIEW state."""
        s = ClaudeSession()
        s._session_active = True
        s._state = SessionState.THINKING
        with pytest.raises(SessionNotActiveError, match="Not in INTERVIEW state"):
            s.respond_interview("1")

    def test_respond_enter(self):
        s = ClaudeSession()
        s._session_active = True
        s._tmux = MagicMock()
        s._state = SessionState.INTERVIEW
        with patch.object(s, '_refresh_state'):
            result = s.respond_interview("enter")
        s._tmux.send_special_key.assert_called_once_with("Enter")
        assert result["responded"] is True

    def test_respond_escape(self):
        s = ClaudeSession()
        s._session_active = True
        s._tmux = MagicMock()
        s._state = SessionState.INTERVIEW
        with patch.object(s, '_refresh_state'):
            result = s.respond_interview("escape")
        s._tmux.send_special_key.assert_called_once_with("Escape")
        assert result["responded"] is True

    def test_respond_number_direct_typing(self):
        """Typing a number navigates via arrow keys to that option."""
        s = ClaudeSession()
        s._session_active = True
        s._tmux = MagicMock()
        s._state = SessionState.INTERVIEW

        with patch.object(s, '_refresh_state'):
            result = s.respond_interview("3")

        # Should send Up 20 times (reset to top), then Down 2 times (to option 3), then Enter
        assert s._tmux.send_special_key.call_count == 23  # 20 Up + 2 Down + 1 Enter
        s._tmux.send_special_key.assert_called_with("Enter")
        assert result["responded"] is True

    def test_respond_option_zero_rejected(self):
        """Option '0' should fall through to custom text path (not isdigit()>0)."""
        s = ClaudeSession()
        s._session_active = True
        s._tmux = MagicMock()
        s._state = SessionState.INTERVIEW
        with patch.object(s, '_refresh_state'):
            result = s.respond_interview("0")
        # "0" is a digit but int("0") > 0 is False, so it falls to custom text
        s._tmux.send_keys.assert_called_once_with("0", enter=True)
        assert result["responded"] is True

    def test_respond_custom_text(self):
        s = ClaudeSession()
        s._session_active = True
        s._tmux = MagicMock()
        s._state = SessionState.INTERVIEW
        with patch.object(s, '_refresh_state'):
            result = s.respond_interview("custom text input")
        s._tmux.send_keys.assert_called_once_with("custom text input", enter=True)
        assert result["responded"] is True
