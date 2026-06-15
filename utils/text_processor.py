"""
Text Processing Utilities — Pure, stateless text manipulation functions.

Migrated from agent/conversation_loop.py.
Responsibility: Prompt assembly, regex cleaning, markdown formatting, token estimation.
Does NOT handle I/O, network, or state — only input strings -> output strings.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


# ─── Think Block Stripping ────────────────────────────────────────────────────

THINK_TAG_PATTERN = re.compile(r'</?(?:REASONING_SCRATCHPAD|think|reasoning)>', re.IGNORECASE)


def strip_think_blocks(text: str) -> str:
    """Remove reasoning XML tags from text.

    Corresponds to the re.sub pattern in conversation_loop.py L3365-L3367.
    """
    if not text:
        return ""
    return THINK_TAG_PATTERN.sub('', text).strip()


def extract_first_line(text: str, max_length: int = 80) -> str:
    """Extract the first line of text, truncated to max_length."""
    if not text:
        return ""
    first = text.split('\n', 1)[0]
    return first[:max_length]


# ─── Markdown / Code Block Cleaning ───────────────────────────────────────────

def normalize_code_blocks(text: str) -> str:
    """Fix markdown code block fences that may be malformed."""
    if not text:
        return ""
    # Ensure consistent triple-backtick fences
    text = re.sub(r'^```\s*$', '```', text, flags=re.MULTILINE)
    return text


def strip_ansi_escape(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    if not text:
        return ""
    ansi_escape = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
    return ansi_escape.sub('', text)


# ─── Prompt Assembly Helpers ───────────────────────────────────────────────────

def build_system_prompt(system_message: Optional[str], continuation: Optional[str] = None) -> str:
    """Assemble the system prompt with optional continuation."""
    parts = []
    if system_message:
        parts.append(system_message)
    if continuation:
        parts.append(continuation)
    return "\n\n".join(parts)


def join_messages(messages: list, sep: str = "\n") -> str:
    """Join a list of message strings with separator."""
    return sep.join(str(m) for m in messages if m)


# ─── String Sanitisation ──────────────────────────────────────────────────────

def sanitize_api_response_text(text: str) -> str:
    """Sanitise raw API response text before parsing."""
    if not text:
        return ""
    # Strip null bytes that crash downstream .strip() calls
    text = text.replace('\x00', '')
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    return text


def strip_whitespace_edges(text: str) -> str:
    """Strip leading/trailing whitespace from text."""
    if not text:
        return ""
    return text.strip()


def collapse_empty_lines(text: str, max_consecutive: int = 2) -> str:
    """Collapse more than max_consecutive empty lines into max_consecutive."""
    if not text:
        return ""
    pattern = r'\n{' + str(max_consecutive + 1) + r',}'
    return re.sub(pattern, '\n' * max_consecutive, text)


# ─── Token Estimation (rough) ──────────────────────────────────────────────────

def estimate_tokens_rough(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English, 2 for Chinese.

    This is a rough heuristic. For accurate counts, use tiktoken or equivalent.
    """
    if not text:
        return 0
    # Chinese characters: ~1.5-2 tokens each
    # Latin characters: ~4 per token
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    other_chars = len(text) - chinese_chars
    return int(chinese_chars * 1.5 + other_chars * 0.25)


# ─── Response Classification Helpers ─────────────────────────────────────────

def classify_truncated_response(response_text: str) -> bool:
    """Detect if a response appears to be truncated (no proper termination)."""
    if not response_text:
        return False
    stripped = response_text.strip()
    # Check for unclosed code blocks
    open_fences = stripped.count('```') - stripped.count('```') // 2 * 2
    if open_fences % 2 == 1:
        return True
    # Check for incomplete tool calls (starts with `{` but doesn't end with `}`)
    if stripped.startswith('{') and not stripped.rstrip().endswith('}'):
        return True
    return False


def extract_error_reason(http_error_body: str) -> Optional[str]:
    """Extract error reason from HTTP error body."""
    if not http_error_body:
        return None
    # Try to find common error message patterns
    patterns = [
        r'"error"\s*:\s*"([^"]+)"',
        r'"message"\s*:\s*"([^"]+)"',
        r'error:\s*(.+)',
    ]
    for pat in patterns:
        m = re.search(pat, http_error_body, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None