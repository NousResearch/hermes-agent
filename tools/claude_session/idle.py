"""idle.py — Minimal state detection for Claude Code TUI output.

Extracted from the former output_parser.py. Only retains what's needed
to answer "what state is Claude in?" and "is Claude done?".

No state machine, no turn tracking, no user-prompt parsing.
Just pure functions that inspect tmux output.
"""

import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------

class SessionState:
    IDLE = "IDLE"
    THINKING = "THINKING"
    TOOL_CALL = "TOOL_CALL"
    PERMISSION = "PERMISSION"
    INTERVIEW = "INTERVIEW"
    ERROR = "ERROR"
    DISCONNECTED = "DISCONNECTED"
    EXITED = "EXITED"


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\].*?\x07|\x1b\[.*?m")

_TOOL_CALL_RE = re.compile(r"^●\s+(\w+)(?:\s+(.+))?$")
_TOOL_CALL_PAREN_RE = re.compile(r"^●\s+(\w+)\((.+)\)$")
_PROMPT_RE = re.compile(r"^❯")
_PASTED_TEXT_RE = re.compile(r"^\❯\s*\[Pasted text")

_PERMISSION_RE = re.compile(
    r"(Allow\s+.*\?"
    r"|.*permission\s+to.*"
    r"|❯\s*(Allow|Yes)\b"
    r"|❯\s*\d+\.\s*(Yes|Allow|Deny|No)\b"
    r"|Do you want to proceed\?"
    r"|.*Yes.*No\b)",
    re.IGNORECASE,
)

_STATUS_BAR_RE = re.compile(
    r"(bypass permissions (on|off)|shift\+tab to cycle|esc to interrupt|"
    r"⏵⏵|/model|/mcp|/ide for Visual Studio Code|"
    r"\d+\s+MCP\s+servers?\s+failed|"
    r"tmux focus-events|"
    r"[─━]{5,})",
    re.IGNORECASE,
)
# Status bar without decoration lines — used to distinguish real IDLE from phantom prompt.
# Decoration lines (───) are ambiguous: they appear in both phantom prompts and real IDLE.
_STATUS_BAR_CONTENT_RE = re.compile(
    r"(bypass permissions (on|off)|shift\+tab to cycle|esc to interrupt|"
    r"⏵⏵|/model|/mcp|/ide for Visual Studio Code|"
    r"\d+\s+MCP\s+servers?\s+failed|"
    r"tmux focus-events)",
    re.IGNORECASE,
)
_DECORATION_RE = re.compile(r"^[─━]{5,}$")
_ERROR_RE = re.compile(r"(Error:.*|Failed:.*|error:.*)", re.IGNORECASE)
_DONE_TIME_RE = re.compile(r"^✻\s+\S+.*\bfor\s+\d+[hms]", re.IGNORECASE)
_COMPACT_RE = re.compile(
    r"(Compacting|compressing\s+conversation|context\s+compression|"
    r"condensing|summarizing\s+conversation|✓.*compact|"
    r"concise.*summary|compact.*history)",
    re.IGNORECASE,
)
# Spinner animation patterns — indicate Claude is actively thinking/working
# These appear as animated characters at the start of lines during processing
# NOTE: "Brewed for Xm" is a DONE marker, excluded here (handled by _DONE_TIME_RE)
# Match ANY text after spinner char — Claude Code verbs are constantly changing
_SPINNER_CHAR_RE = re.compile(
    r"^[" + re.escape("✶✽✻✢·*") + r"]\s+\S",
)
# Legacy: specific verb matching (kept for backward compat with tests)
_SPINNER_RE = re.compile(
    r"^[" + re.escape("✶✽✻✢·*") + r"]\s+"
    r"(Nebulizing|Zigzagging|Tomfoolering|thinking|"
    r"still thinking|thinking more|thinking some more|almost done thinking|"
    r"Deliberating|Meditating|Pondering|Reasoning|Analyzing|Processing|"
    r"Photosynthesizing|Marinating|Percolating|"
    r"Cogitating|Ruminating|Speculating|Computing|Evaluating)",
    re.IGNORECASE,
)
_WELCOME_SCREEN_RE = re.compile(
    r"(welcome to claude|welcome back|tips for getting started|recent activity)",
    re.IGNORECASE,
)
_CLAUDE_TUI_SIGNATURE_RE = re.compile(
    r"(claude|thinking|compacting|bypass permissions|"
    r"shift\+tab to cycle|esc to interrupt|/model|/mcp|"
    r"⏵⏵|welcome to claude)",
    re.IGNORECASE,
)
_TRUST_WORKSPACE_RE = re.compile(r"quick\s+safety\s+check", re.IGNORECASE)
_ENTER_TO_CONFIRM_RE = re.compile(r"enter\s+to\s+confirm", re.IGNORECASE)

# Interview/Selector patterns — Claude Code interactive selection menus
_INTERVIEW_NAV_RE = re.compile(
    r"(Enter to select|Tab.*Arrow keys to navigate|Esc to cancel|"
    r"ctrl\+o to expand|\d+\.\s+Type something\.)",
    re.IGNORECASE,
)
_INTERVIEW_OPTION_RE = re.compile(r"^❯\s*\d+\.\s+\S", re.MULTILINE)
_INTERVIEW_NUMBERED_RE = re.compile(r"^\s*\d+\.\s+\S", re.MULTILINE)
_INTERVIEW_SECTION_RE = re.compile(r"[☐✔]\s+\S+", re.UNICODE)

# Tool name → activity classification (for observer)
_TOOL_ACTIVITY_MAP = {
    "Read": "reading",
    "Write": "writing",
    "Edit": "writing",
    "MultiEdit": "writing",
    "Bash": "executing",
    "Grep": "searching",
    "Glob": "searching",
    "Search": "searching",
    "WebSearch": "searching",
    "WebFetch": "searching",
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class StateResult:
    """Result of detecting Claude Code state from tmux output."""
    state: str
    tool_name: Optional[str] = None
    tool_target: Optional[str] = None
    is_compacting: bool = False


@dataclass
class StartupScene:
    """Detected startup scene requiring special handling."""
    scene_type: str
    description: str
    action: str  # "press_enter" | "press_down_enter"


# ---------------------------------------------------------------------------
# Core detection functions
# ---------------------------------------------------------------------------

def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences."""
    return _ANSI_RE.sub("", text)


def clean_lines(raw_output: str) -> list:
    """Split raw tmux output into cleaned, non-empty lines."""
    text = strip_ansi(raw_output)
    return [line for line in text.splitlines() if line.strip()]


def detect_state(lines: list) -> StateResult:
    """Detect current Claude Code state from cleaned output lines.

    Priority: ERROR > PERMISSION > INTERVIEW > TOOL_CALL > IDLE > THINKING
    """
    if not lines:
        return StateResult(state=SessionState.THINKING)

    last_lines = lines[-15:] if len(lines) >= 15 else lines
    all_text = "\n".join(last_lines[-5:])
    recent_lines = lines[-10:] if len(lines) >= 10 else lines

    # ERROR (suppressed when tool markers are active)
    has_active_tool = any(_parse_tool_line(l) for l in recent_lines)
    error_match = _ERROR_RE.search(all_text)
    if error_match and not has_active_tool:
        return StateResult(state=SessionState.ERROR)

    # PERMISSION (exclude status bar noise)
    non_status = [l for l in last_lines if not _STATUS_BAR_RE.search(l)]
    if non_status:
        if _PERMISSION_RE.search("\n".join(non_status)):
            return StateResult(state=SessionState.PERMISSION)

    # INTERVIEW — interactive selector menu (not permission, not regular prompt)
    # Requires both: (1) nav/section indicators AND (2) ❯ cursor on a numbered option.
    # The ❯ cursor is mandatory to distinguish from Claude's numbered explanations.
    # Uses full line list (not windowed) because the ❯ cursor can be many lines
    # above the nav hints when option descriptions span multiple lines.
    full_text = "\n".join(lines)
    if _INTERVIEW_NAV_RE.search(all_text) or _INTERVIEW_SECTION_RE.search(all_text):
        if _INTERVIEW_OPTION_RE.search(full_text):
            return StateResult(state=SessionState.INTERVIEW)

    # TOOL_CALL — skip stale markers (❯ appears below ●)
    rev = list(reversed(recent_lines))
    for i, line in enumerate(rev):
        tool_info = _parse_tool_line(line)
        if tool_info:
            has_prompt_below = any(_PROMPT_RE.search(l) for l in rev[:i])
            if has_prompt_below:
                break
            return StateResult(
                state=SessionState.TOOL_CALL,
                tool_name=tool_info[0],
                tool_target=tool_info[1],
            )

    # IDLE — bare ❯ prompt (not permission selector, not phantom)
    # BUT: if spinner is active, Claude is actually THINKING, not idle
    # Per-line exclusion: detect spinner on any line that is NOT a done marker
    # (a done marker on line A should not suppress spinner on line B)
    has_spinner = any(
        _SPINNER_CHAR_RE.search(l) and not _DONE_TIME_RE.search(l)
        for l in recent_lines
    )
    if has_spinner:
        return StateResult(state=SessionState.THINKING)

    idle_lines = [l for l in last_lines if not _STATUS_BAR_RE.search(l)]
    for line in reversed(idle_lines):
        stripped = line.strip()
        if _PROMPT_RE.search(line):
            if re.match(r"^❯\s*(Allow|Yes|Deny|No|\d+\.)", stripped, re.IGNORECASE):
                continue
            if _PASTED_TEXT_RE.search(stripped):
                return StateResult(state=SessionState.THINKING)
            # Phantom prompt check
            if _is_phantom_prompt(lines, last_lines):
                continue
            # Shell prompt check
            if _is_shell_prompt(lines):
                return StateResult(state=SessionState.EXITED)
            return StateResult(state=SessionState.IDLE)

    # COMPACT
    if _COMPACT_RE.search(all_text):
        return StateResult(state=SessionState.THINKING, is_compacting=True)

    return StateResult(state=SessionState.THINKING)


def detect_activity(lines: list) -> dict:
    """Detect Claude's current activity from output lines.

    Returns dict with: activity, detail, raw
    """
    if not lines:
        return {"activity": "idle", "detail": "", "raw": ""}

    recent = lines[-20:] if len(lines) >= 20 else lines
    rev = list(reversed(recent))

    for i, line in enumerate(rev):
        stripped = line.strip()
        m = _TOOL_CALL_PAREN_RE.match(stripped)
        if m:
            tool_name, target = m.group(1), m.group(2)
        else:
            m = _TOOL_CALL_RE.match(stripped)
            if not m:
                continue
            tool_name, target = m.group(1), m.group(2) or ""

        # Stale marker check: if ✻ completion or ❯ prompt appears AFTER
        # this ● marker in the original output (= before it in rev), the marker is stale.
        # rev[:i] = lines more recent than the marker (below on screen).
        # When i=0, marker is the last line — nothing comes after it, so it can't be stale.
        if i > 0:
            after_in_original = rev[:i]
            if any("✻" in l for l in after_in_original) or any("❯" in l for l in after_in_original):
                continue

        activity = _TOOL_ACTIVITY_MAP.get(tool_name, "tool_call")
        return {"activity": activity, "detail": target, "raw": stripped}

    # Thinking fallback
    for line in reversed(recent):
        for pattern in ("Thinking", "Processing", "Shenaniganing", "Churned"):
            if pattern in line:
                return {"activity": "thinking", "detail": line.strip(), "raw": line.strip()}

    return {"activity": "idle", "detail": "", "raw": ""}


def detect_startup_scene(lines: list) -> Optional[StartupScene]:
    """Detect startup scenes requiring special handling."""
    if not lines:
        return None
    all_text = "\n".join(lines)
    if _TRUST_WORKSPACE_RE.search(all_text) and _ENTER_TO_CONFIRM_RE.search(all_text):
        return StartupScene(
            scene_type="workspace_trust",
            description="Workspace trust confirmation prompt",
            action="press_enter",
        )
    return None


def is_permission_in_text(text: str) -> bool:
    """Check if text contains a real permission prompt (not status bar)."""
    lines = clean_lines(text)
    last = lines[-5:] if len(lines) >= 5 else lines
    non_status = [l for l in last if not _STATUS_BAR_RE.search(l)]
    return bool(_PERMISSION_RE.search("\n".join(non_status)))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_tool_line(line: str) -> Optional[tuple]:
    """Parse '● ToolName target'. Returns (name, target) or None."""
    stripped = line.strip()
    m = _TOOL_CALL_PAREN_RE.match(stripped)
    if m:
        return (m.group(1), m.group(2))
    m = _TOOL_CALL_RE.match(stripped)
    if m:
        return (m.group(1), m.group(2) or "")
    return None


def _is_phantom_prompt(lines: list, last_lines: list) -> bool:
    """Check if ❯ is a phantom prompt (TUI renders it while working)."""
    for i, raw_line in enumerate(last_lines):
        if _PROMPT_RE.search(raw_line):
            nearby_separators = 0
            for j in range(max(0, i - 2), min(len(last_lines), i + 3)):
                if j == i:
                    continue
                if _DECORATION_RE.search(last_lines[j].strip()):
                    nearby_separators += 1
            if nearby_separators >= 2:
                has_welcome = any(_WELCOME_SCREEN_RE.search(l) for l in lines)
                global_idx = len(lines) - len(last_lines) + i
                has_done = any(_DONE_TIME_RE.search(l) for l in lines[:global_idx])
                if has_welcome or has_done:
                    break
                # Status bar content near the prompt means real IDLE (not phantom).
                # Claude Code v2.1.126 renders: ❯ + ── + "bypass permissions on"
                # This looks like phantom (2 separators) but is actually IDLE.
                # Use _STATUS_BAR_CONTENT_RE (excludes decoration lines) to avoid
                # false positives when the only "status bar" lines are ─── separators.
                status_bar_window = last_lines[-3:] if len(last_lines) >= 3 else last_lines
                if any(_STATUS_BAR_CONTENT_RE.search(l) for l in status_bar_window):
                    break
                return True
            # Fewer than 2 separators — normally means real prompt (not phantom).
            # But during animation, a phantom ❯ may appear without the typical
            # 2-separator layout. Only treat as phantom if the broader output
            # contains Claude TUI elements (otherwise it's a bare shell prompt).
            has_claude_tui = any(_CLAUDE_TUI_SIGNATURE_RE.search(l) for l in lines)
            if has_claude_tui:
                status_bar_window = last_lines[-3:] if len(last_lines) >= 3 else last_lines
                if (not any(_STATUS_BAR_CONTENT_RE.search(l) for l in status_bar_window)
                        and not any(_CLAUDE_TUI_SIGNATURE_RE.search(l) for l in status_bar_window)):
                    return True
            break
    return False


def _is_shell_prompt(lines: list) -> bool:
    """Check if output is a bare shell prompt (Claude has exited)."""
    window = lines[-5:] if len(lines) >= 5 else lines
    if not any("❯" in l for l in window):
        return False
    window_text = "\n".join(window)
    if (any(_DECORATION_RE.search(l.strip()) for l in window)
            or any(_STATUS_BAR_RE.search(l) for l in window)
            or any("●" in l for l in window)
            or any(_DONE_TIME_RE.search(l) for l in window)
            or _CLAUDE_TUI_SIGNATURE_RE.search(window_text)
            or _WELCOME_SCREEN_RE.search(window_text)):
        return False
    return True
