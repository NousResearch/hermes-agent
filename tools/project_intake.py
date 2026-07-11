"""Controller-side project intake cards and split policy.

The helpers in this module are deterministic and intentionally do not dispatch
work.  They translate a request into controller-visible metadata before a
worker route is selected or invoked.
"""

from __future__ import annotations

import re
from typing import Any, Optional


# These labels are part of the controller-facing routing contract.  Keep them
# stable so a caller can present the reason for a mandatory split directly.
_SPLIT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "multi_deliverable",
        (
            r"\b(?:report|briefing|article|blog post|deck|slides|presentation)\b.*\b(?:and|&|, then)\b.*\b(?:report|briefing|article|blog post|deck|slides|presentation)\b",
            r"\b(?:create|produce|deliver|build)\b.*\b(?:and|&|, then)\b.*\b(?:create|produce|deliver|build)\b",
            r"(?:分别)?(?:产出|制作|创建|交付).*(?:报告|简报|文章|演示文稿|幻灯片).*(?:和|及|与|、).*(?:报告|简报|文章|演示文稿|幻灯片)",
        ),
    ),
    (
        "multi_source",
        (
            r"\b(?:two|three|four|multiple|several|many)\s+(?:sources?|documents?|papers?|websites?|urls?)\b",
            r"\b(?:compare|synthesize|cross-check)\b.*\b(?:sources?|documents?|papers?|websites?|urls?)\b",
            r"(?:对比|比较|综合|交叉验证|整理|汇总|梳理).*(?:多个|多种|若干).*(?:来源|文档|论文|网站|网址|材料)",
        ),
    ),
    (
        "research_plus_writing",
        (
            r"\b(?:research|investigate|study)\b.*\b(?:write|draft|author|compose)\b",
            r"\b(?:write|draft|author|compose)\b.*\b(?:research|investigate|study)\b",
            r"(?:调研|研究|调查).*(?:写|撰写|起草|编写).*(?:文章|报告|文案|稿件)",
            r"(?:写|撰写|起草|编写).*(?:文章|报告|文案|稿件).*(?:调研|研究|调查)",
            r"(?:整理|汇总|梳理).*(?:多个|多种|若干).*(?:来源|文档|论文|网站|网址|材料).*(?:写|撰写|起草|编写).*(?:文章|报告|文案|稿件)",
        ),
    ),
    (
        "implementation_plus_verification",
        (
            r"\b(?:implement|build|code|add|fix)\b.*\b(?:verify|validate|test|tests|testing|qa)\b",
            r"\b(?:verify|validate|test|tests|testing|qa)\b.*\b(?:implement|build|code|add|fix)\b",
            r"(?:实现|开发|编写|修复|添加).*(?:验证|测试|校验|检查)",
            r"(?:验证|测试|校验|检查).*(?:实现|开发|编写|修复|添加)",
        ),
    ),
    (
        "high_impact_decision",
        (
            r"\b(?:choose|decide|recommend|select|approve)\b.*\b(?:production|migration|security|architecture|vendor|budget|strategy)\b",
            r"\b(?:production|migration|security|architecture|vendor|budget|strategy)\b.*\b(?:decision|choice|approval)\b",
            r"(?:决定|选择|推荐|批准).*(?:生产环境|迁移|安全|架构|供应商|预算|战略|方案)",
            r"(?:生产环境|迁移|安全|架构|供应商|预算|战略).*(?:决定|选择|推荐|批准)",
        ),
    ),
)


def split_triggers(goal: str, context: Optional[str] = None) -> list[str]:
    """Return mandatory-split reasons detected from the user request."""
    text = f"{goal}\n{context or ''}".lower()
    return [
        name
        for name, patterns in _SPLIT_RULES
        if any(re.search(pattern, text) for pattern in patterns)
    ]


def _candidate_from_shape(goal: str, task_type: Optional[str]) -> str:
    """Provide a safe display candidate; cost_router replaces it with its route."""
    text = f"{task_type or ''}\n{goal}".lower()
    if any(word in text for word in ("final review", "architecture", "conflict", "audit")):
        return "sol"
    if any(word in text for word in ("classify", "dedupe", "tag", "metadata", "filter")):
        return "luna"
    return "terra"


def _deliverable_and_evidence(goal: str, task_type: Optional[str]) -> tuple[str, list[str]]:
    """Describe the outcome without echoing a generic completion promise."""
    text = f"{task_type or ''}\n{goal}".lower()
    if any(word in text for word in ("classify", "分类")):
        return (
            "Topic classifications for the provided URLs.",
            ["Classifications are provided for each URL and identify its topic."],
        )
    if any(word in text for word in ("draft", "write", "article", "撰写", "文章", "报告")):
        return (
            "A draft that addresses the requested subject and constraints.",
            ["The draft covers the requested subject and stated constraints."],
        )
    if any(word in text for word in ("implement", "build", "code", "实现", "开发", "修复")):
        return (
            "The requested implementation change.",
            ["The implementation is present and its requested behavior is demonstrated."],
        )
    return (
        "A response that addresses the requested goal.",
        ["The response addresses the stated goal and requested constraints."],
    )


def build_project_card(
    goal: str,
    *,
    context: Optional[str] = None,
    task_type: Optional[str] = None,
    route_candidate: Optional[str] = None,
) -> dict[str, Any]:
    """Build an additive user-understandable project card for a request.

    Simple one-shot work remains a single slice and includes an explicit reason.
    A positive ``split_required`` only gates controller routing; it does not
    create tasks or change the synchronous cost_router execution model.
    """
    user_goal = str(goal or "").strip()
    triggers = split_triggers(user_goal, context)
    split_required = bool(triggers)
    user_visible_deliverable, acceptance_evidence = _deliverable_and_evidence(user_goal, task_type)
    return {
        "user_goal": user_goal,
        "user_visible_deliverable": user_visible_deliverable,
        "dependencies": [],
        "acceptance_evidence": acceptance_evidence,
        "route_candidate": route_candidate or _candidate_from_shape(user_goal, task_type),
        "split_required": split_required,
        "split_triggers": triggers,
        "no_split_reason": None if split_required else "Single deliverable with no mandatory split trigger.",
    }
