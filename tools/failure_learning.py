#!/usr/bin/env python3
"""
Failure Learning — 失败分类、日志、分析、修补建议。

当 Agent 任务失败时，记录失败上下文、分类错误类型、
分析失败模式、生成 SKILL.md 修补建议。

架构:
  tools/failure_learning.py    ← 本文件（失败学习逻辑）
  tools/skill_evolution.py     ← 工具注册层（skill_failure_report）
  .skill-index/failures.jsonl  ← 失败日志存储
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FAILURES_DIR = Path.home() / ".hermes" / "hermes-agent" / ".skill-index"
_FAILURES_FILE = _FAILURES_DIR / "failures.jsonl"
_MAX_FAILURE_LOG = 200  # Keep at most this many failure records

# Error categories with detection patterns
_ERROR_CATEGORIES: Dict[str, List[str]] = {
    "missing_skill": [
        r"no\s+skill\s+found",
        r"skill\s+not\s+found",
        r"unknown\s+skill",
        r"no\s+matching\s+skill",
    ],
    "wrong_skill": [
        r"wrong\s+skill",
        r"incorrect\s+skill",
        r"not\s+the\s+right",
        r"used\s+wrong",
    ],
    "execution_error": [
        r"permission\s+denied",
        r"access\s+denied",
        r"command\s+not\s+found",
        r"no\s+such\s+file",
        r"syntax\s+error",
        r"import\s+error",
        r"module\s+not\s+found",
        r"connection\s+refused",
        r"connection\s+timed?\s*out",
        r"errno",
        r"traceback",
    ],
    "timeout": [
        r"timed?\s*out",
        r"deadline\s+exceeded",
        r"slow\s+response",
        r"took\s+too\s+long",
    ],
    "insufficient": [
        r"not\s+enough",
        r"missing\s+information",
        r"incomplete",
        r"needs?\s+more",
        r"insufficient",
    ],
    "hallucination": [
        r"hallucinat",
        r"fabricat",
        r"made\s+up",
        r"not\s+real",
        r"doesn'?t\s+exist",
    ],
}


# ---------------------------------------------------------------------------
# Failure Classifier
# ---------------------------------------------------------------------------

class FailureClassifier:
    """Classify task failures into actionable categories."""

    def classify(self, error_text: str = "", context: str = "") -> str:
        """Classify an error based on text and context.

        Returns one of: missing_skill, wrong_skill, execution_error,
        timeout, insufficient, hallucination, unknown.
        """
        combined = f"{error_text} {context}".lower()

        for category, patterns in _ERROR_CATEGORIES.items():
            for pattern in patterns:
                if re.search(pattern, combined, re.IGNORECASE):
                    return category

        return "unknown"

    def classify_batch(self, failures: List[Dict[str, Any]]) -> Dict[str, int]:
        """Classify a batch of failures and return category counts."""
        counts = Counter()
        for f in failures:
            cat = self.classify(
                error_text=f.get("error_text", ""),
                context=f.get("context", ""),
            )
            counts[cat] += 1
        return dict(counts)


# ---------------------------------------------------------------------------
# Failure Logger
# ---------------------------------------------------------------------------

class FailureLogger:
    """Persist task failures to JSONL for pattern analysis."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or _FAILURES_FILE

    def log(
        self,
        skill_name: str,
        error_text: str = "",
        context: str = "",
        task: str = "",
        category: Optional[str] = None,
    ) -> str:
        """Record a failure. Returns the failure_id."""
        # Auto-classify if not provided
        if category is None:
            classifier = FailureClassifier()
            category = classifier.classify(error_text, context)

        failure_id = f"f{int(datetime.now(timezone.utc).timestamp())}"
        record = {
            "id": failure_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "skill": skill_name,
            "category": category,
            "error_text": error_text[:500],
            "context": context[:500],
            "task": task[:200],
            "resolution": None,
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return failure_id

    def load_failures(
        self,
        skill_name: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Load failures with optional filters."""
        if not self.path.exists():
            return []

        results = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if skill_name and record.get("skill") != skill_name:
                    continue
                if category and record.get("category") != category:
                    continue
                results.append(record)
                if len(results) >= limit:
                    break

        return results


# ---------------------------------------------------------------------------
# Failure Analyzer
# ---------------------------------------------------------------------------

class FailureAnalyzer:
    """Analyze failure patterns across skills and categories."""

    def __init__(self, logger_instance: Optional[FailureLogger] = None):
        self.logger = logger_instance or FailureLogger()

    def analyze_skill(self, skill_name: str) -> Dict[str, Any]:
        """Analyze failure patterns for a specific skill."""
        failures = self.logger.load_failures(skill_name=skill_name, limit=100)
        if not failures:
            return {
                "skill": skill_name,
                "total_failures": 0,
                "categories": {},
                "top_errors": [],
                "recommendation": "no_failures",
            }

        categories = Counter(f.get("category", "unknown") for f in failures)
        error_texts = [f.get("error_text", "") for f in failures if f.get("error_text")]
        top_errors = Counter(error_texts).most_common(5)

        # Generate recommendation
        most_common_cat = categories.most_common(1)[0][0]
        recommendations = {
            "missing_skill": "skill_gap — consider creating a new skill for this task",
            "wrong_skill": "retrieval_issue — review skill descriptions for clarity",
            "execution_error": "skill_content — add error handling or edge cases",
            "timeout": "performance — optimize or add timeout handling",
            "insufficient": "skill_content — add more detailed instructions",
            "hallucination": "verification — add validation steps",
            "unknown": "investigate — review failure context manually",
        }

        return {
            "skill": skill_name,
            "total_failures": len(failures),
            "categories": dict(categories),
            "top_errors": [{"error": e, "count": c} for e, c in top_errors],
            "recommendation": recommendations.get(most_common_cat, "unknown"),
        }

    def analyze_global(self, limit: int = 200) -> Dict[str, Any]:
        """Analyze failure patterns across all skills."""
        failures = self.logger.load_failures(limit=limit)
        if not failures:
            return {
                "total_failures": 0,
                "skills_affected": 0,
                "top_skills": [],
                "top_categories": {},
            }

        skill_counts = Counter(f.get("skill", "unknown") for f in failures)
        category_counts = Counter(f.get("category", "unknown") for f in failures)

        return {
            "total_failures": len(failures),
            "skills_affected": len(skill_counts),
            "top_skills": [
                {"skill": s, "failures": c}
                for s, c in skill_counts.most_common(10)
            ],
            "top_categories": dict(category_counts.most_common()),
        }


# ---------------------------------------------------------------------------
# Skill Patcher — suggest SKILL.md patches based on failure patterns
# ---------------------------------------------------------------------------

class SkillPatcher:
    """Generate SKILL.md patch suggestions based on failure analysis."""

    def __init__(self, analyzer: Optional[FailureAnalyzer] = None):
        self.analyzer = analyzer or FailureAnalyzer()

    def suggest_patch(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """Analyze failures for skill_name and suggest a patch.

        Returns:
            {
                "skill": name,
                "failure_summary": {...},
                "patch_suggestion": {
                    "section": "## Error Handling" or "## Pitfalls",
                    "content": "markdown content to add",
                    "reason": "why this patch helps"
                }
            } or None if no action needed.
        """
        analysis = self.analyzer.analyze_skill(skill_name)

        if analysis["total_failures"] == 0:
            return None

        most_common_cat = max(
            analysis["categories"], key=analysis["categories"].get
        )

        # Generate patch content based on failure category
        patch_content = self._generate_patch_content(
            skill_name, most_common_cat, analysis
        )

        if not patch_content:
            return None

        return {
            "skill": skill_name,
            "failure_summary": {
                "total": analysis["total_failures"],
                "top_category": most_common_cat,
                "categories": analysis["categories"],
            },
            "patch_suggestion": patch_content,
        }

    def _generate_patch_content(
        self, skill_name: str, category: str, analysis: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        """Generate patch content based on failure category."""
        top_errors = analysis.get("top_errors", [])
        error_examples = "; ".join(
            e["error"][:80] for e in top_errors[:3]
        ) if top_errors else "N/A"

        patches = {
            "execution_error": {
                "section": "## Error Handling",
                "content": (
                    f"## Error Handling\n\n"
                    f"Common failures observed ({analysis['total_failures']} occurrences):\n\n"
                    f"```\n{error_examples}\n```\n\n"
                    f"**Before executing commands:**\n"
                    f"1. Check file existence before reading/writing\n"
                    f"2. Verify permissions before sudo operations\n"
                    f"3. Test network connectivity before remote calls\n"
                    f"4. Validate input parameters before passing to tools\n"
                ),
                "reason": f"Top error category: execution_error ({analysis['categories'].get('execution_error', 0)} times). "
                         f"Adding error handling patterns reduces repeat failures.",
            },
            "timeout": {
                "section": "## Performance",
                "content": (
                    f"## Performance\n\n"
                    f"Timeout issues observed ({analysis['total_failures']} occurrences).\n\n"
                    f"**Optimization tips:**\n"
                    f"1. Use background mode for long-running commands\n"
                    f"2. Set explicit timeouts (timeout parameter)\n"
                    f"3. Break large tasks into smaller steps\n"
                    f"4. Check for blocking operations\n"
                ),
                "reason": "Timeout errors indicate the skill needs better task decomposition.",
            },
            "insufficient": {
                "section": "## Prerequisites",
                "content": (
                    f"## Prerequisites\n\n"
                    f"Insufficient information/ resources observed ({analysis['total_failures']} occurrences).\n\n"
                    f"**Before starting:**\n"
                    f"1. Verify all required inputs are available\n"
                    f"2. Check that dependencies are installed\n"
                    f"3. Confirm environment variables are set\n"
                    f"4. Validate target system is accessible\n"
                ),
                "reason": "Insufficient errors indicate missing prerequisite checks.",
            },
            "missing_skill": {
                "section": "## Related Skills",
                "content": (
                    f"## Related Skills\n\n"
                    f"This skill was used for tasks it wasn't designed for ({analysis['total_failures']} times).\n\n"
                    f"**Consider loading:**\n"
                    f"- Use `skill_search` to find more relevant skills\n"
                    f"- Check if a different skill better matches the task\n"
                ),
                "reason": "Missing skill errors suggest the skill description may be too broad.",
            },
        }

        return patches.get(category)
