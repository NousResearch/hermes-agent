#!/usr/bin/env python3
"""
unified_reflection.py — 统一反思模块

将 compound-system（任务后反思）和 skill-evolution（技能自进化）整合到一个模块。

职责：
1. 记录失败/成功事件（来自 compound 或 skill）
2. 提取模式（错误模式、使用模式）
3. 检索类似事件的建议
4. 同时写入 .skill-index/ 和 .compound/

数据流：
┌──────────────┐  ┌──────────────┐
│ compound.sh  │  │ skill tools  │
│ task_end     │  │ skill_used   │
└──────┬───────┘  └──────┬───────┘
       │                  │
       ▼                  ▼
┌──────────────────────────────────┐
│     UnifiedReflection            │
│  - record_event()                │
│  - extract_patterns()            │
│  - get_suggestions()             │
└──────────────────────────────────┘
       │                  │
       ▼                  ▼
┌──────────────┐  ┌──────────────┐
│ .skill-index │  │ .compound/   │
│ failure_log  │  │ reflections/ │
└──────────────┘  └──────────────┘
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ============================================================
# 配置
# ============================================================

HOME = Path.home()
SKILL_INDEX_DIR = HOME / ".skill-index"
COMPOUND_DIR = HOME / ".compound"
FAILURE_LOG = SKILL_INDEX_DIR / "failure_log.jsonl"
PATTERNS_FILE = SKILL_INDEX_DIR / "patterns.json"
COMPOUND_REFLECTIONS = COMPOUND_DIR / "reflections"

# 确保目录存在
SKILL_INDEX_DIR.mkdir(exist_ok=True)
COMPOUND_DIR.mkdir(exist_ok=True)
COMPOUND_REFLECTIONS.mkdir(parents=True, exist_ok=True)

# ============================================================
# 事件类型
# ============================================================

class EventType:
    TASK_END = "task_end"           # 任务完成（来自 compound）
    SKILL_USED = "skill_used"       # 技能被使用
    SKILL_FAILED = "skill_failed"   # 技能使用失败
    ERROR_RECOVERED = "error_recovered"  # 错误已解决
    ERROR_UNRESOLVED = "error_unresolved"  # 错误未解决

class Severity:
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    BLOCKING = 4

# ============================================================
# 核心函数
# ============================================================

def record_event(
    event_type: str,
    description: str,
    outcome: str = "success",
    severity: int = Severity.NONE,
    skill_name: Optional[str] = None,
    error_message: Optional[str] = None,
    tool_calls: Optional[list] = None,
    files_modified: Optional[list] = None,
    tags: Optional[list] = None,
    source: str = "unknown",  # "compound" or "skill_evolution"
) -> dict:
    """
    记录事件（来自 compound 或 skill evolution）。
    
    返回记录的事件 dict。
    """
    now = datetime.now(timezone.utc).isoformat()
    
    event = {
        "timestamp": now,
        "event_type": event_type,
        "description": description[:200],
        "outcome": outcome,
        "severity": severity,
        "skill_name": skill_name,
        "error_message": (error_message or "")[:500],
        "tool_calls": (tool_calls or [])[:20],
        "files_modified": (files_modified or [])[:10],
        "tags": tags or [],
        "source": source,
    }
    
    # 写入 skill-index failure_log.jsonl
    _append_jsonl(FAILURE_LOG, event)

    # 同时写入 .compound/reflections/（如果来自 compound）
    if source == "compound":
        _write_compound_reflection(event)

    # Phase 2.2: 追踪 skill 使用结果
    if event.get("skill_name"):
        _update_skill_usage(event)

    # Phase 3.1: 失败驱动的 skill 进化
    if event.get("skill_name") and event.get("outcome") in ("failure", "skill_failed", "error_unresolved"):
        _trigger_skill_evolution(event)

    return event


def extract_patterns(events: Optional[list] = None) -> list:
    """
    从事件列表中提取重复模式。
    
    如果不传 events，自动读取 failure_log.jsonl。
    """
    if events is None:
        events = _read_jsonl(FAILURE_LOG)
    
    if not events:
        return []
    
    # 按错误类型分组
    error_groups = {}
    for event in events:
        if event.get("outcome") in ("failure", "error_unresolved", "error_recovered"):
            # 从 error_message 提取错误类型
            error_type = _classify_error(event.get("error_message", ""))
            if error_type not in error_groups:
                error_groups[error_type] = []
            error_groups[error_type].append(event)
    
    patterns = []
    for error_type, group_events in error_groups.items():
        if len(group_events) >= 2:  # 至少出现 2 次才算模式
            # 提取共同标签
            all_tags = []
            for e in group_events:
                all_tags.extend(e.get("tags", []))
            tag_counts = {}
            for tag in all_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            common_tags = sorted(tag_counts.keys(), key=lambda t: tag_counts[t], reverse=True)[:5]
            
            # 提取共同解决方案
            solutions = []
            for e in group_events:
                sol = e.get("solution", "")
                if sol and sol not in solutions:
                    solutions.append(sol)
            
            patterns.append({
                "pattern_type": "error",
                "error_type": error_type,
                "occurrence_count": len(group_events),
                "common_tags": common_tags,
                "sample_errors": [e.get("error_message", "")[:100] for e in group_events[:3]],
                "solutions": solutions[:3],
                "first_seen": group_events[0].get("timestamp"),
                "last_seen": group_events[-1].get("timestamp"),
            })
    
    # 保存 patterns
    _save_json(PATTERNS_FILE, patterns)
    
    return patterns


def get_suggestions(
    error_message: str = "",
    skill_name: Optional[str] = None,
    tags: Optional[list] = None,
    limit: int = 5,
    use_embedding: bool = True,
) -> list:
    """
    检索类似事件的建议。

    优先级：
    1. Embedding 语义检索（相关 skills）
    2. 精确匹配 skill_name + error_message
    3. 标签匹配
    4. 模糊匹配 error_message
    """
    events = _read_jsonl(FAILURE_LOG)
    patterns = _load_json(PATTERNS_FILE) or []

    suggestions = []

    # 0. Embedding 语义检索（调用已有的 skill_index.semantic_search）
    if use_embedding and error_message:
        try:
            from tools.skill_index import semantic_search
            emb_results = semantic_search(error_message, top_k=limit)
            for r in emb_results:
                suggestions.append({
                    "type": "skill",
                    "score": r.get("score", 0) * 10,  # Scale to match keyword scores
                    "skill": r.get("name", ""),
                    "description": r.get("description", ""),
                })
        except Exception:
            pass  # 降级到关键词匹配
    
    # 1. 从 patterns 中找匹配
    for pattern in patterns:
        score = 0
        if pattern.get("error_type") and error_message:
            # 错误类型匹配
            if pattern["error_type"].lower() in error_message.lower():
                score += 5
        if tags and pattern.get("common_tags"):
            # 标签匹配
            common = set(tags) & set(pattern["common_tags"])
            score += len(common) * 2
        
        if score > 0:
            suggestions.append({
                "type": "pattern",
                "score": score,
                "error_type": pattern.get("error_type"),
                "occurrence_count": pattern.get("occurrence_count", 0),
                "solutions": pattern.get("solutions", []),
                "common_tags": pattern.get("common_tags", []),
            })
    
    # 2. 从 events 中找直接匹配
    for event in reversed(events):  # 最新的优先
        score = 0
        if skill_name and event.get("skill_name") == skill_name:
            score += 3
        if error_message and event.get("error_message"):
            # 简单模糊匹配
            error_words = set(error_message.lower().split())
            event_words = set(event.get("error_message", "").lower().split())
            overlap = error_words & event_words
            if len(overlap) >= 2:
                score += len(overlap)
        # 标签匹配
        if tags and event.get("tags"):
            common = set(tags) & set(event.get("tags", []))
            score += len(common) * 2
        
        if score > 0:
            suggestions.append({
                "type": "event",
                "score": score,
                "description": event.get("description", ""),
                "error_message": event.get("error_message", "")[:200],
                "solution": event.get("solution", ""),
                "timestamp": event.get("timestamp"),
                "skill_name": event.get("skill_name"),
                "tags": event.get("tags", []),
            })
    
    # 按 score 排序
    suggestions.sort(key=lambda s: s.get("score", 0), reverse=True)
    
    return suggestions[:limit]


# ============================================================
# 内部辅助函数
# ============================================================

def _append_jsonl(path: Path, data: dict):
    """追加一行 JSON 到 JSONL 文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list:
    """读取 JSONL 文件"""
    if not path.exists():
        return []
    
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def _save_json(path: Path, data):
    """保存 JSON 文件"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_json(path: Path):
    """加载 JSON 文件"""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_compound_reflection(event: dict):
    """写入 .compound/reflections/ 目录"""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H%M%S")
    
    filename = f"{date_str}_{time_str}.json"
    filepath = COMPOUND_REFLECTIONS / filename
    
    _save_json(filepath, event)


def _classify_error(error_message: str) -> str:
    """从错误信息分类错误类型"""
    if not error_message:
        return "unknown"
    
    error_lower = error_message.lower()
    
    patterns = {
        "api_error": ["api", "401", "403", "404", "500", "502", "503", "timeout", "rate limit"],
        "config_error": ["config", "yaml", "json", "toml", "env", "variable"],
        "network_error": ["network", "connection", "dns", "socket", "ssh", "tunnel"],
        "tool_error": ["tool", "command", "not found", "permission denied"],
        "code_error": ["syntax", "import", "module", "type", "attribute", "name"],
        "environment_error": ["disk", "memory", "space", "oom", "killed"],
    }
    
    for error_type, keywords in patterns.items():
        for keyword in keywords:
            if keyword in error_lower:
                return error_type
    
    return "other"


# ============================================================
# Skill Usage Tracking + Evolution (Phase 2.2 + Phase 3)
# ============================================================

USAGE_FILE = Path.home() / ".hermes" / "skills" / ".usage.json"


def _load_usage() -> dict:
    """Load .usage.json."""
    if not USAGE_FILE.exists():
        return {}
    try:
        return json.loads(USAGE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_usage(usage: dict) -> None:
    """Save .usage.json."""
    USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    USAGE_FILE.write_text(json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8")


def _update_skill_usage(event: dict) -> None:
    """Update .usage.json with skill outcome.

    Updates success_rate and total_outcomes for the skill.
    """
    skill_name = event.get("skill_name", "")
    if not skill_name:
        return

    usage = _load_usage()
    if skill_name not in usage:
        usage[skill_name] = {
            "use_count": 0,
            "total_outcomes": 0,
            "success_rate": 0.5,
        }

    entry = usage[skill_name]
    entry["total_outcomes"] = entry.get("total_outcomes", 0) + 1
    entry["use_count"] = entry.get("use_count", 0) + 1

    # Update success rate (incremental average)
    is_success = event.get("outcome") == "success"
    total = entry["total_outcomes"]
    old_rate = entry.get("success_rate", 0.5)
    entry["success_rate"] = round(
        old_rate + (1.0 if is_success else 0.0 - old_rate) / total, 4
    )

    _save_usage(usage)


def _trigger_skill_evolution(event: dict) -> None:
    """Failure-driven skill evolution.

    Logic (inspired by SkillRL recursive evolution):
    1. success_rate < 0.3 and uses > 5 → flag for review
    2. "not found" errors → log discovery trigger

    Note: usage is already updated by record_event() before this is called.
    """
    skill_name = event.get("skill_name", "")
    if not skill_name:
        return

    # 1. Check if skill needs review
    usage = _load_usage()
    entry = usage.get(skill_name, {})
    success_rate = entry.get("success_rate", 0.5)
    total = entry.get("total_outcomes", 0)

    if success_rate < 0.3 and total > 5:
        _flag_for_review(skill_name, entry)

    # 2. "not found" → discovery trigger
    error_msg = event.get("error_message", "").lower()
    if "not found" in error_msg or "no such" in error_msg:
        _log_discovery_trigger(event)


def _flag_for_review(skill_name: str, usage_data: dict) -> None:
    """Flag a low-quality skill for human review."""
    review_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "skill_review_needed",
        "skill_name": skill_name,
        "success_rate": usage_data.get("success_rate", 0),
        "total_outcomes": usage_data.get("total_outcomes", 0),
        "description": f"Skill '{skill_name}' has low success rate ({usage_data.get('success_rate', 0):.1%}) after {usage_data.get('total_outcomes', 0)} uses",
    }
    _append_jsonl(FAILURE_LOG, review_entry)


def _log_discovery_trigger(event: dict) -> None:
    """Log that a skill discovery should be triggered."""
    trigger_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "discovery_trigger",
        "description": f"Missing skill detected: {event.get('error_message', '')[:100]}",
        "error_message": event.get("error_message", ""),
        "tags": event.get("tags", []),
    }
    _append_jsonl(FAILURE_LOG, trigger_entry)


# ============================================================
# CLI 接口（供 compound.sh 调用）
# ============================================================

def cli_record(args: list):
    """CLI: record <type> <description> [outcome] [severity] [error_msg]"""
    if len(args) < 2:
        print("Usage: unified_reflection.py record <type> <description> [outcome] [severity] [error_msg]")
        return
    
    event_type = args[0]
    description = args[1]
    outcome = args[2] if len(args) > 2 else "success"
    severity = int(args[3]) if len(args) > 3 else 0
    error_msg = args[4] if len(args) > 4 else None
    
    event = record_event(
        event_type=event_type,
        description=description,
        outcome=outcome,
        severity=severity,
        error_message=error_msg,
        source="compound",
    )
    
    print(json.dumps(event, ensure_ascii=False, indent=2))


def cli_suggestions(args: list):
    """CLI: suggestions <error_message> [skill_name] [limit]"""
    if len(args) < 1:
        print("Usage: unified_reflection.py suggestions <error_message> [skill_name] [limit]")
        return
    
    error_msg = args[0]
    skill_name = args[1] if len(args) > 1 else None
    limit = int(args[2]) if len(args) > 2 else 5
    
    suggestions = get_suggestions(
        error_message=error_msg,
        skill_name=skill_name,
        limit=limit,
    )
    
    print(json.dumps(suggestions, ensure_ascii=False, indent=2))


def cli_patterns(args: list):
    """CLI: patterns — 提取并显示模式"""
    patterns = extract_patterns()
    print(json.dumps(patterns, ensure_ascii=False, indent=2))


def cli_evolve(args: list):
    """CLI: evolve <skill_name> <outcome>

    Record a skill outcome and trigger evolution if needed.
    outcome: success or failure
    """
    if len(args) < 2:
        print("Usage: unified_reflection.py evolve <skill_name> <outcome>")
        print("  outcome: success | failure")
        return

    skill_name = args[0]
    outcome = args[1]

    event = record_event(
        event_type=EventType.SKILL_USED if outcome == "success" else EventType.SKILL_FAILED,
        description=f"Skill {skill_name} {'succeeded' if outcome == 'success' else 'failed'}",
        outcome=outcome,
        skill_name=skill_name,
        source="compound",
    )

    # Show updated stats
    usage = _load_usage()
    entry = usage.get(skill_name, {})
    result = {
        "event": event,
        "stats": {
            "skill": skill_name,
            "success_rate": entry.get("success_rate", 0.5),
            "total_outcomes": entry.get("total_outcomes", 0),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: unified_reflection.py <command> [args...]")
        print("Commands:")
        print("  record <type> <description> [outcome] [severity] [error_msg]")
        print("  suggestions <error_message> [skill_name] [limit]")
        print("  patterns")
        print("  evolve <skill_name> <outcome>")
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    if command == "record":
        cli_record(args)
    elif command == "suggestions":
        cli_suggestions(args)
    elif command == "patterns":
        cli_patterns(args)
    elif command == "evolve":
        cli_evolve(args)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
