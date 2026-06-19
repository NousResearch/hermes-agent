"""Skill Discovery — 自动技能发现系统。

从成功的任务轨迹中自动提取可复用的 Skill。

组件:
  TrajectoryAnalyzer — 轻量轨迹记录与提取
  PatternDetector    — N-gram 重复模式发现
  SkillGenerator     — 从模式生成 SKILL.md
  SkillValidator     — 验证生成的 skill 质量
  SkillDiscovery     — 主协调器

数据:
  .skill-index/trajectories.jsonl   — 工具调用日志
  .skill-index/discovered_patterns.json — 发现的模式
  skills/learned/                   — 生成的 skills
"""

from __future__ import annotations

import json
import re
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SKILL_INDEX_DIR = Path.home() / ".hermes" / "hermes-agent" / ".skill-index"
TRAJECTORY_FILE = SKILL_INDEX_DIR / "trajectories.jsonl"
PATTERNS_FILE = SKILL_INDEX_DIR / "discovered_patterns.json"
LEARNED_SKILLS_DIR = Path.home() / ".hermes" / "skills" / "learned"


# ===========================================================================
# 1. TrajectoryAnalyzer — 轻量轨迹记录与提取
# ===========================================================================

class TrajectoryAnalyzer:
    """轻量轨迹记录与分析。

    记录工具调用到 JSONL，按 task_id 分组提取序列。
    """

    def __init__(self, trajectory_file: Optional[Path] = None):
        self.trajectory_file = trajectory_file or TRAJECTORY_FILE
        self.trajectory_file.parent.mkdir(parents=True, exist_ok=True)

    def record_tool_call(
        self,
        task_id: str,
        tool_name: str,
        success: bool = True,
        task: str = "",
    ) -> None:
        """记录一次工具调用。"""
        entry = {
            "task_id": task_id or "default",
            "tool": tool_name,
            "success": success,
            "ts": datetime.now(timezone.utc).isoformat(),
            "task": task,
        }
        with open(self.trajectory_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def extract_tool_sequences(
        self,
        min_calls: int = 5,
        max_age_days: int = 30,
    ) -> List[Dict[str, Any]]:
        """按 task_id 分组，返回成功轨迹的工具序列。

        Args:
            min_calls: 最少工具调用次数才保留
            max_age_days: 只分析最近 N 天的轨迹

        Returns:
            [{"task_id": "...", "sequence": ["terminal", "write_file"],
              "task": "deploy nginx", "success": True, "timestamp": "..."}]
        """
        if not self.trajectory_file.exists():
            return []

        # Read and group by task_id
        tasks: Dict[str, List[Dict]] = defaultdict(list)
        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_days * 86400)

        try:
            with open(self.trajectory_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Filter by age
                    ts_str = entry.get("ts", "")
                    if ts_str:
                        try:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            if ts.timestamp() < cutoff:
                                continue
                        except (ValueError, TypeError):
                            pass

                    task_id = entry.get("task_id", "default")
                    tasks[task_id].append(entry)
        except (OSError, IOError):
            return []

        # Build sequences from successful tasks
        sequences = []
        for task_id, entries in tasks.items():
            # Check if all calls succeeded
            all_success = all(e.get("success", False) for e in entries)
            if not all_success:
                continue

            # Extract tool name sequence (preserve order)
            seen = set()
            sequence = []
            for e in entries:
                tool = e.get("tool", "")
                if tool and tool not in seen:
                    sequence.append(tool)
                    seen.add(tool)

            if len(sequence) < min_calls:
                continue

            # Get task description from first entry
            task_desc = next((e.get("task", "") for e in entries if e.get("task")), "")
            ts = entries[0].get("ts", "")

            sequences.append({
                "task_id": task_id,
                "sequence": sequence,
                "task": task_desc,
                "success": True,
                "timestamp": ts,
            })

        return sequences

    def get_stats(self) -> Dict[str, Any]:
        """返回轨迹统计。"""
        if not self.trajectory_file.exists():
            return {"total_calls": 0, "unique_tasks": 0, "file_exists": False}

        total = 0
        tasks: Set[str] = set()
        tools: Counter = Counter()

        try:
            with open(self.trajectory_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        total += 1
                        tasks.add(entry.get("task_id", "default"))
                        tools[entry.get("tool", "unknown")] += 1
                    except json.JSONDecodeError:
                        continue
        except (OSError, IOError):
            pass

        return {
            "total_calls": total,
            "unique_tasks": len(tasks),
            "top_tools": tools.most_common(10),
            "file_exists": True,
        }


# ===========================================================================
# 2. PatternDetector — N-gram 重复模式发现
# ===========================================================================

class PatternDetector:
    """从工具调用序列中发现重复模式。"""

    def find_repeated_patterns(
        self,
        sequences: List[Dict[str, Any]],
        min_occurrences: int = 3,
        min_length: int = 3,
        max_length: int = 10,
    ) -> List[Dict[str, Any]]:
        """发现出现 ≥min_occurrences 次的子序列。

        使用滑动窗口 + N-gram 频率统计。

        Returns:
            [{"pattern": ["terminal", "write_file", "patch"],
              "count": 5,
              "confidence": 0.85,
              "task_summaries": ["deploy nginx", "setup flask"]}]
        """
        if not sequences:
            return []

        # Count N-grams across all sequences
        ngram_counts: Dict[Tuple, Dict] = {}

        for seq_data in sequences:
            seq = seq_data.get("sequence", [])
            task = seq_data.get("task", "")

            for n in range(min_length, min(max_length + 1, len(seq) + 1)):
                for i in range(len(seq) - n + 1):
                    ngram = tuple(seq[i:i + n])
                    if ngram not in ngram_counts:
                        ngram_counts[ngram] = {"count": 0, "tasks": set()}
                    ngram_counts[ngram]["count"] += 1
                    if task:
                        ngram_counts[ngram]["tasks"].add(task)

        # Filter by min_occurrences
        candidates = []
        for ngram, data in ngram_counts.items():
            if data["count"] >= min_occurrences:
                candidates.append({
                    "pattern": list(ngram),
                    "count": data["count"],
                    "task_summaries": list(data["tasks"]),
                })

        # Merge overlapping patterns (Jaccard > 0.7)
        merged = self._merge_overlapping(candidates)

        # Sort by count (descending)
        merged.sort(key=lambda x: x["count"], reverse=True)

        # Add confidence score
        total_sequences = len(sequences)
        for p in merged:
            p["confidence"] = min(1.0, p["count"] / max(total_sequences, 1))

        return merged

    def _merge_overlapping(
        self, patterns: List[Dict], threshold: float = 0.7
    ) -> List[Dict]:
        """合并高度重叠的模式。"""
        if not patterns:
            return []

        # Sort by count descending (keep larger patterns)
        patterns.sort(key=lambda x: (-x["count"], -len(x["pattern"])))

        merged = []
        used = set()

        for i, p in enumerate(patterns):
            if i in used:
                continue

            current = {
                "pattern": p["pattern"][:],
                "count": p["count"],
                "task_summaries": list(p["task_summaries"]),
            }

            for j, q in enumerate(patterns):
                if j <= i or j in used:
                    continue

                # Check Jaccard similarity
                set_a = set(p["pattern"])
                set_b = set(q["pattern"])
                if not set_a or not set_b:
                    continue

                jaccard = len(set_a & set_b) / len(set_a | set_b)
                if jaccard >= threshold:
                    # Merge: keep the longer pattern, combine counts
                    current["count"] = max(current["count"], q["count"])
                    # Merge task summaries
                    for t in q["task_summaries"]:
                        if t not in current["task_summaries"]:
                            current["task_summaries"].append(t)
                    used.add(j)

            merged.append(current)
            used.add(i)

        return merged


# ===========================================================================
# 3. SkillGenerator — 从模式生成 SKILL.md
# ===========================================================================

class SkillGenerator:
    """从发现的模式生成 SKILL.md 内容。"""

    # Tool → step description mapping
    TOOL_STEP_MAP = {
        "terminal": "Run shell command",
        "write_file": "Create or modify file",
        "patch": "Edit file contents",
        "read_file": "Read file contents",
        "search_files": "Search for files or content",
        "web_search": "Search the web",
        "web_extract": "Extract web page content",
        "delegate_task": "Delegate to subagent",
        "execute_code": "Execute Python code",
        "skill_manage": "Manage skills",
        "memory": "Query or update memory",
        "session_search": "Search past sessions",
        "fact_store": "Query knowledge base",
    }

    def generate(
        self,
        pattern: Dict[str, Any],
        task_summaries: Optional[List[str]] = None,
    ) -> str:
        """生成 SKILL.md 内容。

        Args:
            pattern: {"pattern": ["terminal", "write_file"], "count": 5, ...}
            task_summaries: ["deploy nginx", "setup flask"]

        Returns:
            Complete SKILL.md string
        """
        tasks = task_summaries or pattern.get("task_summaries", [])
        seq = pattern.get("pattern", [])
        count = pattern.get("count", 0)

        # Generate name from task summaries
        name = self._generate_name(tasks, seq)
        description = self._generate_description(tasks, seq)
        triggers = self._generate_triggers(tasks)
        steps = self._generate_steps(seq)

        # Build SKILL.md
        skill_md = f"""---
name: {name}
description: "{description}"
version: 0.1.0
auto_generated: true
source: skill_discovery
confidence: {pattern.get('confidence', 0):.2f}
occurrences: {count}
created_at: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}
---

# {name.replace('-', ' ').title()}

{description}

## When to Use

{triggers}

## Steps

{steps}

## Notes

- This skill was auto-generated from {count} successful task trajectories.
- Task examples: {', '.join(tasks[:3]) if tasks else 'N/A'}
- Review and refine before using in production.
"""
        return skill_md

    def _generate_name(self, tasks: List[str], seq: List[str]) -> str:
        """生成 kebab-case name。"""
        if tasks:
            # Use first task summary, sanitize
            base = tasks[0].lower()
            base = re.sub(r"[^a-z0-9\s-]", "", base)
            base = re.sub(r"\s+", "-", base.strip())
            base = base[:40]
            if base:
                return f"learned-{base}"

        # Fallback: use tool sequence
        if seq:
            return f"learned-{'-'.join(seq[:3])}"

        return "learned-unknown"

    def _generate_description(self, tasks: List[str], seq: List[str]) -> str:
        """生成 description。"""
        if tasks:
            return f"Automated workflow for: {tasks[0]}"
        return f"Automated workflow using: {', '.join(seq[:3])}"

    def _generate_triggers(self, tasks: List[str]) -> str:
        """生成 triggers 部分。"""
        lines = ["Use this skill when:"]
        for task in tasks[:3]:
            lines.append(f"- Task involves: {task}")
        if not tasks:
            lines.append("- Task matches the tool sequence pattern")
        return "\n".join(lines)

    def _generate_steps(self, seq: List[str]) -> str:
        """生成 steps 部分。"""
        lines = []
        for i, tool in enumerate(seq, 1):
            step_desc = self.TOOL_STEP_MAP.get(tool, f"Use {tool}")
            lines.append(f"{i}. **{step_desc}** (`{tool}`)")
        return "\n".join(lines)


# ===========================================================================
# 4. SkillValidator — 验证生成的 skill
# ===========================================================================

class SkillValidator:
    """验证生成的 skill 质量。"""

    REQUIRED_FIELDS = {"name", "description"}
    SENSITIVE_PATTERNS = [
        r"api[_-]?key",
        r"secret",
        r"password",
        r"token",
        r"bearer",
        r"sk-[a-zA-Z0-9]",
    ]

    def validate(self, skill_md: str) -> Dict[str, Any]:
        """验证 skill 内容。

        Returns:
            {"valid": bool, "score": float 0-1, "issues": list, "checks": dict}
        """
        checks = {
            "yaml_frontmatter": False,
            "has_name": False,
            "has_description": False,
            "has_steps": False,
            "min_length": False,
            "no_secrets": False,
        }
        issues = []

        # Check YAML frontmatter
        if skill_md.startswith("---"):
            checks["yaml_frontmatter"] = True
            # Extract frontmatter
            try:
                end = skill_md.index("---", 3)
                frontmatter = skill_md[3:end].strip()
                for field in self.REQUIRED_FIELDS:
                    if f"{field}:" in frontmatter:
                        checks[f"has_{field}"] = True
                    else:
                        issues.append(f"Missing field in frontmatter: {field}")
            except ValueError:
                issues.append("Invalid YAML frontmatter (no closing ---)")
        else:
            issues.append("Missing YAML frontmatter")

        # Check for steps
        if "## Steps" in skill_md or "## steps" in skill_md:
            checks["has_steps"] = True
        else:
            issues.append("Missing ## Steps section")

        # Check minimum length
        if len(skill_md) >= 200:
            checks["min_length"] = True
        else:
            issues.append(f"Too short: {len(skill_md)} chars (min 200)")

        # Check for secrets
        has_secrets = False
        for pattern in self.SENSITIVE_PATTERNS:
            if re.search(pattern, skill_md, re.IGNORECASE):
                has_secrets = True
                issues.append(f"Potential secret detected: {pattern}")
                break
        checks["no_secrets"] = not has_secrets

        # Calculate score
        passed = sum(1 for v in checks.values() if v)
        score = passed / len(checks) if checks else 0.0

        return {
            "valid": all(checks.values()),
            "score": score,
            "issues": issues,
            "checks": checks,
        }


# ===========================================================================
# 5. SkillDiscovery — 主协调器
# ===========================================================================

class SkillDiscovery:
    """自动技能发现 — 主协调器。"""

    def __init__(self):
        self.analyzer = TrajectoryAnalyzer()
        self.detector = PatternDetector()
        self.generator = SkillGenerator()
        self.validator = SkillValidator()

    def analyze(self, min_occurrences: int = 3) -> Dict[str, Any]:
        """分析轨迹，发现重复模式。

        Returns:
            {"patterns_found": int, "patterns": [...],
             "total_trajectories": int, "stats": {...}}
        """
        stats = self.analyzer.get_stats()
        sequences = self.analyzer.extract_tool_sequences()
        patterns = self.detector.find_repeated_patterns(
            sequences, min_occurrences=min_occurrences
        )

        # Save discovered patterns
        if patterns:
            self._save_patterns(patterns)

        return {
            "patterns_found": len(patterns),
            "patterns": patterns[:10],  # Top 10
            "total_trajectories": len(sequences),
            "stats": stats,
        }

    def generate_skill(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """从模式生成 skill 草稿。

        Returns:
            {"skill_md": str, "validation": dict, "suggested_name": str}
        """
        skill_md = self.generator.generate(pattern)
        validation = self.validator.validate(skill_md)

        # Extract name from generated content
        name = "learned-unknown"
        if skill_md.startswith("---"):
            try:
                end = skill_md.index("---", 3)
                for line in skill_md[3:end].split("\n"):
                    if line.startswith("name:"):
                        name = line.split(":", 1)[1].strip()
                        break
            except ValueError:
                pass

        return {
            "skill_md": skill_md,
            "validation": validation,
            "suggested_name": name,
        }

    def approve_and_save(self, name: str, skill_md: str) -> Dict[str, Any]:
        """审批并保存 skill 到 skills/learned/。

        Returns:
            {"saved": bool, "path": str, "name": str}
        """
        # Validate before saving
        validation = self.validator.validate(skill_md)
        if not validation["valid"]:
            return {
                "saved": False,
                "error": f"Validation failed: {validation['issues']}",
                "name": name,
            }

        # Create directory
        skill_dir = LEARNED_SKILLS_DIR / name
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Write SKILL.md
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(skill_md, encoding="utf-8")

        # Write meta.yaml
        meta = {
            "name": name,
            "auto_generated": True,
            "source": "skill_discovery",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "validation_score": validation["score"],
        }
        meta_file = skill_dir / "meta.yaml"
        # Simple YAML without dependency
        meta_lines = [f"{k}: {v}" for k, v in meta.items()]
        meta_file.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

        return {
            "saved": True,
            "path": str(skill_dir),
            "name": name,
        }

    def _save_patterns(self, patterns: List[Dict]) -> None:
        """保存发现的模式到 JSON。"""
        try:
            data = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "pattern_count": len(patterns),
                "patterns": patterns,
            }
            PATTERNS_FILE.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except (OSError, IOError) as e:
            import logging
            logging.getLogger(__name__).warning("Failed to save patterns: %s", e)


# ===========================================================================
# Public API (for tool registration)
# ===========================================================================

def discover_patterns(min_occurrences: int = 3) -> str:
    """分析轨迹，发现重复模式。JSON string result."""
    sd = SkillDiscovery()
    result = sd.analyze(min_occurrences=min_occurrences)
    return json.dumps(result, ensure_ascii=False, indent=2)


def generate_skill_from_pattern(pattern_json: str) -> str:
    """从模式 JSON 生成 SKILL.md 草稿。JSON string result."""
    try:
        pattern = json.loads(pattern_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON pattern"}, ensure_ascii=False)

    sd = SkillDiscovery()
    result = sd.generate_skill(pattern)
    return json.dumps(result, ensure_ascii=False, indent=2)


def approve_skill(name: str, skill_md: str) -> str:
    """审批并保存 skill。JSON string result."""
    sd = SkillDiscovery()
    result = sd.approve_and_save(name, skill_md)
    return json.dumps(result, ensure_ascii=False, indent=2)
