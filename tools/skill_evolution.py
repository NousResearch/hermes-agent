#!/usr/bin/env python3
"""
Skill Evolution Tools — 独立于核心文件的 Skill 自进化工具集。

提供语义搜索、质量评分、技能链发现、下游推荐等功能。
通过 tools/registry 自动发现机制注册，不需要修改任何核心文件。

架构:
  tools/skill_evolution.py  ← 本文件（工具注册层）
  tools/skill_index.py      ← 纯逻辑层（搜索/评分/淘汰）
  .skill-index/             ← 数据层（索引/评分 JSON）
"""

import json
import logging
from typing import Any, Dict

from tools.registry import registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Handler functions
# ---------------------------------------------------------------------------

def _handle_skill_search(args: Dict[str, Any]) -> str:
    """语义搜索 skills，返回按相关性排序的结果。"""
    query = args.get("query", "")
    top_k = args.get("top_k", 10)
    if not query:
        return json.dumps({"error": "query is required"}, ensure_ascii=False)
    try:
        from tools.skill_index import semantic_search
        results = semantic_search(query, top_k=top_k)
        return json.dumps({
            "query": query,
            "count": len(results),
            "results": results,
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _handle_skill_scores(args: Dict[str, Any]) -> str:
    """显示 skills 质量评分。"""
    top_n = args.get("top_n", 10)
    try:
        from tools.skill_index import load_quality_scores
        scores = load_quality_scores()
        if not scores:
            return json.dumps({"error": "No quality scores available. Run index first."}, ensure_ascii=False)
        # Sort by total score descending
        sorted_items = sorted(
            scores.items(),
            key=lambda x: x[1].get("total", 0),
            reverse=True,
        )[:top_n]
        results = []
        for name, data in sorted_items:
            dims = data.get("dimensions", {})
            results.append({
                "name": name,
                "total": round(data.get("total", 0), 4),
                "usage": round(dims.get("use_frequency", 0), 4),
                "completeness": round(dims.get("completeness", 0), 4),
                "trust": round(dims.get("trust", 0), 4),
                "freshness": round(dims.get("freshness", 0), 4),
                "activity": round(dims.get("activity", 0), 4),
                "success_rate": round(dims.get("success_rate", 0.5), 4),
                "total_outcomes": data.get("total_outcomes", 0),
            })
        return json.dumps({
            "count": len(results),
            "results": results,
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _handle_skill_chain(args: Dict[str, Any]) -> str:
    """根据目标发现 skill 链（BFS 搜索）。"""
    goals = args.get("goals", [])
    max_depth = args.get("max_depth", 3)
    if not goals:
        return json.dumps({"error": "goals list is required"}, ensure_ascii=False)
    try:
        from tools.skill_index import discover_chain
        chains = discover_chain(goals, max_depth=max_depth)
        if not chains:
            return json.dumps({
                "goals": goals,
                "chains": [],
                "message": "No skill chain found for these goals.",
            }, ensure_ascii=False)
        # Format each chain as readable steps
        result_chains = []
        for chain in chains[:3]:  # limit to top 3 chains
            steps = []
            for i, skill in enumerate(chain):
                steps.append({
                    "step": i + 1,
                    "skill": skill.get("name", "unknown"),
                    "provides": skill.get("provides", []),
                    "requires": skill.get("requires", []),
                })
            result_chains.append({"length": len(steps), "steps": steps})
        return json.dumps({
            "goals": goals,
            "chain_count": len(result_chains),
            "chains": result_chains,
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _handle_skill_suggest(args: Dict[str, Any]) -> str:
    """推荐当前 skill 的下游 skill。"""
    skill_name = args.get("skill", "")
    if not skill_name:
        return json.dumps({"error": "skill name is required"}, ensure_ascii=False)
    try:
        from tools.skill_index import suggest_next_skills
        suggestions = suggest_next_skills(skill_name)
        if not suggestions:
            return json.dumps({
                "skill": skill_name,
                "suggestions": [],
                "message": "No downstream skills found.",
            }, ensure_ascii=False)
        results = []
        for s in suggestions:
            results.append({
                "name": s["name"],
                "reason": s.get("reason", ""),
                "score": round(s.get("score", 0), 4),
            })
        return json.dumps({
            "skill": skill_name,
            "count": len(results),
            "suggestions": results,
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

SKILL_SEARCH_SCHEMA = {
    "name": "skill_search",
    "description": "Semantic search across skills by query. Returns skills ranked by relevance. Use instead of skills_list when you need to find the most relevant skill for a specific task.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Semantic search query (e.g. 'deploy server', '代码审查', 'database migration')",
            },
            "top_k": {
                "type": "integer",
                "description": "Max results to return (default: 10)",
            },
        },
        "required": ["query"],
    },
}

SKILL_SCORES_SCHEMA = {
    "name": "skill_scores",
    "description": "Show quality scores for skills. Scores are based on usage frequency, documentation completeness, trust, freshness, and activity.",
    "parameters": {
        "type": "object",
        "properties": {
            "top_n": {
                "type": "integer",
                "description": "Number of top skills to show (default: 10)",
            },
        },
        "required": [],
    },
}

SKILL_CHAIN_SCHEMA = {
    "name": "skill_chain",
    "description": "Discover a chain of skills that together achieve a complex goal. Uses BFS to find the shortest path through skill dependencies.",
    "parameters": {
        "type": "object",
        "properties": {
            "goals": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of goal capabilities needed (e.g. ['server-ready', 'api-endpoint'])",
            },
            "max_depth": {
                "type": "integer",
                "description": "Max chain depth (default: 3)",
            },
        },
        "required": ["goals"],
    },
}

SKILL_SUGGEST_SCHEMA = {
    "name": "skill_suggest",
    "description": "Suggest downstream skills that complement a given skill. Useful for finding what to load next in a workflow.",
    "parameters": {
        "type": "object",
        "properties": {
            "skill": {
                "type": "string",
                "description": "Name of the skill to get suggestions for (e.g. 'server-operations')",
            },
        },
        "required": ["skill"],
    },
}

SKILL_FEEDBACK_SCHEMA = {
    "name": "skill_feedback",
    "description": "Report the outcome of a task that used a skill. Call this after completing a task to feed results back into the quality scoring system. Improves future skill ranking.",
    "parameters": {
        "type": "object",
        "properties": {
            "skill": {
                "type": "string",
                "description": "Name of the skill that was used (e.g. 'server-operations')",
            },
            "success": {
                "type": "boolean",
                "description": "Whether the task completed successfully",
            },
            "task": {
                "type": "string",
                "description": "Brief description of what was done (e.g. 'deploy nginx server')",
            },
            "latency_ms": {
                "type": "integer",
                "description": "Approximate task duration in milliseconds (optional)",
            },
            "error_type": {
                "type": "string",
                "description": "Error type if failed: execution_error, timeout, missing_skill, wrong_skill, insufficient, hallucination (optional)",
            },
        },
        "required": ["skill", "success"],
    },
}


# ---------------------------------------------------------------------------
# Handler: skill_feedback
# ---------------------------------------------------------------------------

def _handle_skill_feedback(args: Dict[str, Any]) -> str:
    """Record task outcome for a skill → feeds back into quality scoring."""
    skill_name = args.get("skill", "")
    success = args.get("success", False)
    task = args.get("task", "")
    latency_ms = args.get("latency_ms", 0)
    error_type = args.get("error_type")

    if not skill_name:
        return json.dumps({"error": "skill name is required"}, ensure_ascii=False)

    try:
        from tools.skill_usage import record_outcome, get_success_rate, get_outcome_stats
        record_outcome(
            skill_name=skill_name,
            success=bool(success),
            task=task,
            latency_ms=int(latency_ms) if latency_ms else 0,
            error_type=error_type,
        )
        rate = get_success_rate(skill_name)
        stats = get_outcome_stats(skill_name)
        return json.dumps({
            "ok": True,
            "skill": skill_name,
            "success": success,
            "new_success_rate": rate,
            "total_outcomes": stats["total_outcomes"],
            "total_successes": stats["total_successes"],
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Register tools — top-level calls so Hermes auto-discovery picks them up
# ---------------------------------------------------------------------------

registry.register(
    name="skill_search",
    toolset="skills",
    schema=SKILL_SEARCH_SCHEMA,
    handler=lambda args, **kw: _handle_skill_search(args),
    emoji="🔍",
)

registry.register(
    name="skill_scores",
    toolset="skills",
    schema=SKILL_SCORES_SCHEMA,
    handler=lambda args, **kw: _handle_skill_scores(args),
    emoji="📊",
)

registry.register(
    name="skill_chain",
    toolset="skills",
    schema=SKILL_CHAIN_SCHEMA,
    handler=lambda args, **kw: _handle_skill_chain(args),
    emoji="⛓️",
)

registry.register(
    name="skill_suggest",
    toolset="skills",
    schema=SKILL_SUGGEST_SCHEMA,
    handler=lambda args, **kw: _handle_skill_suggest(args),
    emoji="💡",
)

registry.register(
    name="skill_feedback",
    toolset="skills",
    schema=SKILL_FEEDBACK_SCHEMA,
    handler=lambda args, **kw: _handle_skill_feedback(args),
    emoji="📝",
)


# ---------------------------------------------------------------------------
# Handler: skill_failure_report
# ---------------------------------------------------------------------------

SKILL_FAILURE_REPORT_SCHEMA = {
    "name": "skill_failure_report",
    "description": "Report a task failure for a skill. Classifies the error, logs it, and returns a patch suggestion if patterns are detected. Builds the failure learning database.",
    "parameters": {
        "type": "object",
        "properties": {
            "skill": {
                "type": "string",
                "description": "Name of the skill that was used (e.g. 'server-operations')",
            },
            "error_text": {
                "type": "string",
                "description": "The error message or description (e.g. 'permission denied for /etc/nginx')",
            },
            "task": {
                "type": "string",
                "description": "What the task was trying to do (e.g. 'deploy nginx server')",
            },
            "context": {
                "type": "string",
                "description": "Additional context about the failure (e.g. 'ran after sudo apt install')",
            },
        },
        "required": ["skill", "error_text"],
    },
}


def _handle_skill_failure_report(args: Dict[str, Any]) -> str:
    """Report a failure, classify it, log it, suggest patches."""
    skill_name = args.get("skill", "")
    error_text = args.get("error_text", "")
    task = args.get("task", "")
    context = args.get("context", "")

    if not skill_name:
        return json.dumps({"error": "skill name is required"}, ensure_ascii=False)

    try:
        from tools.failure_learning import FailureLogger, FailureAnalyzer, SkillPatcher

        logger_inst = FailureLogger()
        failure_id = logger_inst.log(
            skill_name=skill_name,
            error_text=error_text,
            context=context,
            task=task,
        )

        # Analyze patterns and suggest patch
        analyzer = FailureAnalyzer(logger_inst)
        patcher = SkillPatcher(analyzer)
        patch = patcher.suggest_patch(skill_name)

        analysis = analyzer.analyze_skill(skill_name)

        result = {
            "ok": True,
            "failure_id": failure_id,
            "skill": skill_name,
            "category": analysis["categories"],
            "total_failures": analysis["total_failures"],
            "recommendation": analysis["recommendation"],
        }
        if patch:
            result["patch_suggestion"] = {
                "section": patch["patch_suggestion"]["section"],
                "reason": patch["patch_suggestion"]["reason"],
                "content_preview": patch["patch_suggestion"]["content"][:200],
            }

        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Handler: skill_failure_analysis
# ---------------------------------------------------------------------------

SKILL_FAILURE_ANALYSIS_SCHEMA = {
    "name": "skill_failure_analysis",
    "description": "Analyze failure patterns for a specific skill or globally. Shows top error categories, affected skills, and recommendations.",
    "parameters": {
        "type": "object",
        "properties": {
            "skill": {
                "type": "string",
                "description": "Skill name to analyze (omit for global analysis)",
            },
        },
        "required": [],
    },
}


def _handle_skill_failure_analysis(args: Dict[str, Any]) -> str:
    """Analyze failure patterns."""
    skill_name = args.get("skill")

    try:
        from tools.failure_learning import FailureAnalyzer
        analyzer = FailureAnalyzer()

        if skill_name:
            analysis = analyzer.analyze_skill(skill_name)
        else:
            analysis = analyzer.analyze_global()

        return json.dumps(analysis, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


registry.register(
    name="skill_failure_report",
    toolset="skills",
    schema=SKILL_FAILURE_REPORT_SCHEMA,
    handler=lambda args, **kw: _handle_skill_failure_report(args),
    emoji="🚨",
)

registry.register(
    name="skill_failure_analysis",
    toolset="skills",
    schema=SKILL_FAILURE_ANALYSIS_SCHEMA,
    handler=lambda args, **kw: _handle_skill_failure_analysis(args),
    emoji="🔍",
)


# ---------------------------------------------------------------------------
# Skill Discovery Tools (Phase 2)
# ---------------------------------------------------------------------------

SKILL_DISCOVER_SCHEMA = {
    "name": "skill_discover",
    "description": "Analyze tool-call trajectories and discover repeated patterns that could become new skills.",
    "parameters": {
        "type": "object",
        "properties": {
            "min_occurrences": {
                "type": "integer",
                "description": "Minimum occurrences to consider a pattern (default: 3)",
                "default": 3,
            },
        },
    },
}

SKILL_GENERATE_SCHEMA = {
    "name": "skill_generate",
    "description": "Generate a SKILL.md draft from a discovered pattern.",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern_json": {
                "type": "string",
                "description": "JSON string of the pattern object from skill_discover",
            },
        },
        "required": ["pattern_json"],
    },
}

SKILL_APPROVE_SCHEMA = {
    "name": "skill_approve",
    "description": "Approve and save a generated skill to skills/learned/.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Skill name (kebab-case, e.g. 'learned-deploy-nginx')",
            },
            "skill_md": {
                "type": "string",
                "description": "The full SKILL.md content to save",
            },
        },
        "required": ["name", "skill_md"],
    },
}


def _handle_skill_discover(args: Dict[str, Any]) -> str:
    """Analyze trajectories and discover patterns."""
    min_occ = args.get("min_occurrences", 3)
    try:
        from tools.skill_discovery import discover_patterns
        return discover_patterns(min_occurrences=min_occ)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _handle_skill_generate(args: Dict[str, Any]) -> str:
    """Generate SKILL.md from a pattern."""
    pattern_json = args.get("pattern_json", "{}")
    try:
        from tools.skill_discovery import generate_skill_from_pattern
        return generate_skill_from_pattern(pattern_json)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _handle_skill_approve(args: Dict[str, Any]) -> str:
    """Approve and save a skill."""
    name = args.get("name", "")
    skill_md = args.get("skill_md", "")
    if not name or not skill_md:
        return json.dumps({"error": "name and skill_md are required"}, ensure_ascii=False)
    try:
        from tools.skill_discovery import approve_skill
        return approve_skill(name, skill_md)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


registry.register(
    name="skill_discover",
    toolset="skills",
    schema=SKILL_DISCOVER_SCHEMA,
    handler=lambda args, **kw: _handle_skill_discover(args),
    emoji="🔬",
)

registry.register(
    name="skill_generate",
    toolset="skills",
    schema=SKILL_GENERATE_SCHEMA,
    handler=lambda args, **kw: _handle_skill_generate(args),
    emoji="📝",
)

registry.register(
    name="skill_approve",
    toolset="skills",
    schema=SKILL_APPROVE_SCHEMA,
    handler=lambda args, **kw: _handle_skill_approve(args),
    emoji="✅",
)


# ---------------------------------------------------------------------------
# Skill Composition Tools (Phase 3)
# ---------------------------------------------------------------------------

SKILL_GRAPH_SCHEMA = {
    "name": "skill_graph",
    "description": "Build a DAG of skills and find weighted shortest paths for task composition.",
    "parameters": {
        "type": "object",
        "properties": {
            "goal_provides": {
                "type": "string",
                "description": "Comma-separated capabilities to achieve (e.g. 'test-passing,server-ready')",
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum chain length (default: 3)",
                "default": 3,
            },
        },
        "required": ["goal_provides"],
    },
}

SKILL_COMPOSE_SCHEMA = {
    "name": "skill_compose",
    "description": "Compose skills: improve (based on failures) or merge (combine overlapping skills).",
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "Operation: 'improve' or 'merge'",
                "enum": ["improve", "merge"],
            },
            "skill_a": {
                "type": "string",
                "description": "First skill name",
            },
            "skill_b": {
                "type": "string",
                "description": "Second skill name (required for merge)",
            },
        },
        "required": ["operation", "skill_a"],
    },
}


def _handle_skill_graph(args: Dict[str, Any]) -> str:
    """Build DAG and find paths."""
    goal = args.get("goal_provides", "")
    max_depth = args.get("max_depth", 3)
    if not goal:
        return json.dumps({"error": "goal_provides is required"}, ensure_ascii=False)
    try:
        from tools.skill_graph import build_graph_and_find_paths
        return build_graph_and_find_paths(goal, max_depth)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _handle_skill_compose(args: Dict[str, Any]) -> str:
    """Compose skills."""
    op = args.get("operation", "")
    skill_a = args.get("skill_a", "")
    skill_b = args.get("skill_b", "")
    if not op or not skill_a:
        return json.dumps({"error": "operation and skill_a are required"}, ensure_ascii=False)
    try:
        from tools.skill_graph import compose_skill
        return compose_skill(op, skill_a, skill_b)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


registry.register(
    name="skill_graph",
    toolset="skills",
    schema=SKILL_GRAPH_SCHEMA,
    handler=lambda args, **kw: _handle_skill_graph(args),
    emoji="🔗",
)

registry.register(
    name="skill_compose",
    toolset="skills",
    schema=SKILL_COMPOSE_SCHEMA,
    handler=lambda args, **kw: _handle_skill_compose(args),
    emoji="🧩",
)


# ---------------------------------------------------------------------------
# Skill Ranking Tools (Phase 4 — Thompson Sampling)
# ---------------------------------------------------------------------------

SKILL_RANK_SCHEMA = {
    "name": "skill_rank",
    "description": "Rank skills using Thompson Sampling (exploration vs exploitation).",
    "parameters": {
        "type": "object",
        "properties": {
            "candidates_json": {
                "type": "string",
                "description": "JSON array of candidate skills [{\"name\": \"...\", \"score\": 0.8}]",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of top results to return (default: 10)",
                "default": 10,
            },
        },
        "required": ["candidates_json"],
    },
}

SKILL_THOMPSON_STATS_SCHEMA = {
    "name": "skill_thompson_stats",
    "description": "Get Thompson Sampling statistics for a skill.",
    "parameters": {
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "Skill name to get stats for",
            },
        },
        "required": ["skill_name"],
    },
}


def _handle_skill_rank(args: Dict[str, Any]) -> str:
    """Rank skills using Thompson Sampling."""
    candidates_json = args.get("candidates_json", "[]")
    top_k = args.get("top_k", 10)
    try:
        from tools.skill_ranking import thompson_rank
        return thompson_rank(candidates_json, top_k)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _handle_skill_thompson_stats(args: Dict[str, Any]) -> str:
    """Get Thompson stats for a skill."""
    skill_name = args.get("skill_name", "")
    if not skill_name:
        return json.dumps({"error": "skill_name is required"}, ensure_ascii=False)
    try:
        from tools.skill_ranking import thompson_stats
        return thompson_stats(skill_name)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


registry.register(
    name="skill_rank",
    toolset="skills",
    schema=SKILL_RANK_SCHEMA,
    handler=lambda args, **kw: _handle_skill_rank(args),
    emoji="🎲",
)

registry.register(
    name="skill_thompson_stats",
    toolset="skills",
    schema=SKILL_THOMPSON_STATS_SCHEMA,
    handler=lambda args, **kw: _handle_skill_thompson_stats(args),
    emoji="📊",
)

# ---------------------------------------------------------------------------
# Unified Reflection — 整合 compound-system 和 skill-evolution
# ---------------------------------------------------------------------------

def _handle_reflection_record(args: Dict[str, Any]) -> str:
    """Record an event to unified reflection system."""
    event_type = args.get("event_type", "task_end")
    description = args.get("description", "")
    outcome = args.get("outcome", "success")
    severity = args.get("severity", 0)
    skill_name = args.get("skill_name")
    error_message = args.get("error_message")
    tags = args.get("tags")
    source = args.get("source", "skill_evolution")
    try:
        from tools.unified_reflection import record_event
        event = record_event(
            event_type=event_type,
            description=description,
            outcome=outcome,
            severity=severity,
            skill_name=skill_name,
            error_message=error_message,
            tags=tags,
            source=source,
        )
        return json.dumps({"recorded": True, "event": event}, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _handle_reflection_suggestions(args: Dict[str, Any]) -> str:
    """Get suggestions from unified reflection system."""
    error_message = args.get("error_message", "")
    skill_name = args.get("skill_name")
    tags = args.get("tags")
    limit = args.get("limit", 5)
    try:
        from tools.unified_reflection import get_suggestions
        suggestions = get_suggestions(
            error_message=error_message,
            skill_name=skill_name,
            tags=tags,
            limit=limit,
        )
        return json.dumps({
            "count": len(suggestions),
            "suggestions": suggestions,
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _handle_reflection_patterns(args: Dict[str, Any]) -> str:
    """Extract and display patterns from reflection events."""
    try:
        from tools.unified_reflection import extract_patterns
        patterns = extract_patterns()
        return json.dumps({
            "count": len(patterns),
            "patterns": patterns,
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


REFLECTION_RECORD_SCHEMA = {
    "name": "reflection_record",
    "description": "Record an event to unified reflection system (compound + skill evolution).",
    "parameters": {
        "type": "object",
        "properties": {
            "event_type": {
                "type": "string",
                "description": "Event type: task_end, skill_used, skill_failed, error_recovered, error_unresolved",
            },
            "description": {
                "type": "string",
                "description": "Event description",
            },
            "outcome": {
                "type": "string",
                "description": "Outcome: success, failure, error_recovered, error_unresolved",
            },
            "severity": {
                "type": "integer",
                "description": "Severity level: 0=none, 1=low, 2=medium, 3=high, 4=blocking",
            },
            "skill_name": {
                "type": "string",
                "description": "Related skill name (optional)",
            },
            "error_message": {
                "type": "string",
                "description": "Error message if any (optional)",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tags for categorization (optional)",
            },
            "source": {
                "type": "string",
                "description": "Source system: compound or skill_evolution",
            },
        },
        "required": ["event_type", "description"],
    },
}

REFLECTION_SUGGESTIONS_SCHEMA = {
    "name": "reflection_suggestions",
    "description": "Get suggestions from unified reflection system based on error message or skill name.",
    "parameters": {
        "type": "object",
        "properties": {
            "error_message": {
                "type": "string",
                "description": "Error message to search for similar issues",
            },
            "skill_name": {
                "type": "string",
                "description": "Skill name to get suggestions for",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tags to match",
            },
            "limit": {
                "type": "integer",
                "description": "Max suggestions to return (default 5)",
            },
        },
        "required": [],
    },
}

REFLECTION_PATTERNS_SCHEMA = {
    "name": "reflection_patterns",
    "description": "Extract and display patterns from reflection events (error patterns, usage patterns).",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


registry.register(
    name="reflection_record",
    toolset="skills",
    schema=REFLECTION_RECORD_SCHEMA,
    handler=lambda args, **kw: _handle_reflection_record(args),
    emoji="📝",
)

registry.register(
    name="reflection_suggestions",
    toolset="skills",
    schema=REFLECTION_SUGGESTIONS_SCHEMA,
    handler=lambda args, **kw: _handle_reflection_suggestions(args),
    emoji="💡",
)

registry.register(
    name="reflection_patterns",
    toolset="skills",
    schema=REFLECTION_PATTERNS_SCHEMA,
    handler=lambda args, **kw: _handle_reflection_patterns(args),
    emoji="🔍",
)
