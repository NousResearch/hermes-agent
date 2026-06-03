#!/usr/bin/env python3
"""
超长期记忆自动追加模块 — memo.txt 追加写入

用途：新学到的重要事实、经验、决策，在写入 memory DB 的同时追加到 memo.txt，
作为超长期无限容量备份。session 结束后 memory 会被压缩，但 memo.txt 永不过期。

使用方式：
    from tools.memo_append import memo_append, memo_append_skill, memo_append_memory

memo.txt 路径：~/.hermes/memo.txt
"""

import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_MEMO_PATH: Optional[Path] = None
_MEMO_LOCK = threading.Lock()


def _get_memo_path() -> Path:
    """Lazily resolve ~/.hermes/memo.txt (import-safe, no hermes_constants dep)."""
    global _MEMO_PATH
    if _MEMO_PATH is None:
        _MEMO_PATH = Path(os.path.expanduser("~/.hermes/memo.txt"))
    return _MEMO_PATH


def memo_append(
    content: str,
    category: str = "general",
    *,
    separator: str = "\n━━━━━━━━━━━━━━━━━━━━━━━━\n",
) -> bool:
    """
    Append content to memo.txt with timestamp and category.

    Thread-safe. Silently no-op on any error (memo.txt is backup, must not crash agent).

    Args:
        content: The text to append
        category: Category tag shown in memo header (e.g. "memory", "skill", "decision")
        separator: Section separator

    Returns:
        True if append succeeded, False otherwise
    """
    if not content or not content.strip():
        return False

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    header = f"{separator}{category.upper()} — {now}\n━━━━━━━━━━━━━━━━━━━━━━━━"
    entry = f"{header}\n{content.strip()}\n"

    try:
        with _MEMO_LOCK:
            path = _get_memo_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(entry)
        return True
    except Exception:
        return False


def memo_append_memory(action: str, content: str, target: str = "memory") -> bool:
    """
    Append a memory mutation event to memo.txt.

    Args:
        action: 'add', 'replace', 'remove'
        content: The memory content
        target: 'memory' or 'user'
    """
    body = f"[memory:{target}] {action}\n{content.strip()}"
    return memo_append(body, category=f"memory/{target}")


def memo_append_skill(action: str, skill_name: str, description: str = "") -> bool:
    """
    Append a skill creation/update event to memo.txt.

    Args:
        action: 'create', 'patch', 'delete'
        skill_name: Name of the skill
        description: Optional description or summary
    """
    body = f"[skill] {action}: {skill_name}"
    if description:
        body += f"\n{description.strip()}"
    return memo_append(body, category="skill")
