"""Skill Graph — DAG weighted pathfinding for skill composition.

Provides:
1. DAG construction from skill provides/requires metadata
2. Dijkstra weighted shortest path
3. SkillComposer: improve and merge skills
"""

from __future__ import annotations

import json
import heapq
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

SKILL_INDEX_DIR = Path.home() / ".hermes" / "skills" / ".skill-index"
SKILLS_DIR = Path.home() / ".hermes" / "skills"
LEARNED_SKILLS_DIR = Path.home() / ".hermes" / "skills" / "learned"
FAILURES_FILE = SKILL_INDEX_DIR / "failures.jsonl"


# ===========================================================================
# SkillGraph — DAG + Dijkstra
# ===========================================================================

class SkillGraph:
    """技能有向无环图，支持加权最短路径。"""

    def __init__(self):
        # node -> [(neighbor, weight)]
        self.graph: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        # name -> skill metadata
        self.skill_info: Dict[str, Dict] = {}
        # Implicit requirements that every skill has
        self.implicit_requires: Set[str] = {"ssh-access", "project-structure"}

    def build(self) -> None:
        """从 skill index 构建 DAG。

        图结构: skill_a --[provides: X]--> skill_b
        表示 skill_a 产出 X，skill_b 需要 X，所以 skill_a → skill_b
        边权重 = 1.0 - success_rate (成功率越高权重越低)
        """
        index_file = SKILL_INDEX_DIR / "skill-index.json"
        if not index_file.exists():
            logger.warning("Skill index not found: %s", index_file)
            return

        try:
            index = json.loads(index_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        skills = index.get("skills", [])

        # Store skill info
        for skill in skills:
            name = skill.get("name", "")
            if not name:
                continue
            self.skill_info[name] = {
                "name": name,
                "provides": skill.get("provides", []),
                "requires": skill.get("requires", []),
                "success_rate": skill.get("success_rate", 0.5),
                "description": skill.get("description", ""),
                # SkillBank fields (Phase 1 of ML-skill-upgrade)
                "principle": skill.get("principle", ""),
                "when_to_apply": skill.get("when_to_apply", ""),
                "common_mistakes": skill.get("common_mistakes", []),
            }

        # Build edges: skill_a -> skill_b if skill_a provides what skill_b requires
        for skill_a in skills:
            name_a = skill_a.get("name", "")
            provides_a = set(skill_a.get("provides", []))
            success_a = skill_a.get("success_rate", 0.5)

            for skill_b in skills:
                name_b = skill_b.get("name", "")
                if name_a == name_b:
                    continue

                requires_b = set(skill_b.get("requires", []))

                # If skill_a provides something skill_b requires
                overlap = provides_a & requires_b
                if overlap:
                    weight = 1.0 - success_a  # Higher success = lower weight
                    self.graph[name_a].append((name_b, weight))

    def dijkstra(self, start: str, goal: str) -> Tuple[List[str], float]:
        """Dijkstra 加权最短路径。

        Returns:
            (path, total_weight) or ([], float('inf')) if no path
        """
        if start == goal:
            return [start], 0.0

        dist: Dict[str, float] = {start: 0.0}
        prev: Dict[str, Optional[str]] = {start: None}
        visited: Set[str] = set()
        heap = [(0.0, start)]

        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)

            if u == goal:
                # Reconstruct path
                path = []
                node = u
                while node is not None:
                    path.append(node)
                    node = prev.get(node)
                path.reverse()
                return path, d

            for v, w in self.graph.get(u, []):
                if v not in visited:
                    new_dist = d + w
                    if new_dist < dist.get(v, float("inf")):
                        dist[v] = new_dist
                        prev[v] = u
                        heapq.heappush(heap, (new_dist, v))

        return [], float("inf")

    def find_paths(
        self,
        goal_provides: List[str],
        max_depth: int = 3,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """找到达目标的多条路径。

        找提供 goal_provides 的 skill，然后找哪些 skill 能到达它们。

        Returns:
            [{"path": ["skill-a", "skill-b"], "weight": 0.8,
              "provides": [...], "skills": [...]}]
        """
        if not goal_provides:
            return []

        # Find skills that provide the goal
        target_skills = set()
        for name, info in self.skill_info.items():
            provides = set(info.get("provides", []))
            if provides & set(goal_provides):
                target_skills.add(name)

        if not target_skills:
            return []

        paths = []

        for target in target_skills:
            # Find paths from any skill to this target
            for start_name in self.skill_info:
                if start_name == target:
                    continue
                path, weight = self.dijkstra(start_name, target)
                if path and weight < float("inf") and len(path) <= max_depth + 1:
                    # Collect provides from all skills in path
                    all_provides = set()
                    for p in path:
                        info = self.skill_info.get(p, {})
                        all_provides.update(info.get("provides", []))

                    paths.append({
                        "path": path,
                        "weight": round(weight, 3),
                        "provides": list(all_provides),
                        "skills": [
                            {
                                "name": p,
                                "success_rate": self.skill_info.get(p, {}).get("success_rate", 0.5),
                                "principle": self.skill_info.get(p, {}).get("principle", ""),
                                "common_mistakes": self.skill_info.get(p, {}).get("common_mistakes", []),
                            }
                            for p in path
                        ],
                    })

        # Deduplicate by path tuple
        seen = set()
        unique = []
        for p in paths:
            key = tuple(p["path"])
            if key not in seen:
                seen.add(key)
                unique.append(p)

        # Sort by weight (ascending = better)
        unique.sort(key=lambda x: x["weight"])

        return unique[:max_results]

    def suggest_composition(self, task_keywords: List[str]) -> List[Dict[str, Any]]:
        """根据任务关键词推荐组合方案。
        
        匹配策略:
        1. 关键词匹配 provides (原有逻辑)
        2. 关键词匹配 when_to_apply (SkillBank 增强)
        """
        # Match keywords to provides (original logic)
        matched_provides = []
        for keyword in task_keywords:
            keyword_lower = keyword.lower()
            for provide_name in self.graph:
                if keyword_lower in provide_name.lower():
                    matched_provides.append(provide_name)

        # Match keywords to when_to_apply (SkillBank enhancement)
        matched_when = []
        for keyword in task_keywords:
            keyword_lower = keyword.lower()
            for name, info in self.skill_info.items():
                when = info.get("when_to_apply", "")
                if when and keyword_lower in when.lower():
                    # Find provides for this skill to use in pathfinding
                    for provide in info.get("provides", []):
                        if provide not in matched_provides:
                            matched_provides.append(provide)

        all_goals = list(set(matched_provides))
        if not all_goals:
            return []

        return self.find_paths(all_goals)


# ===========================================================================
# SkillComposer — improve and merge
# ===========================================================================

class SkillComposer:
    """技能组合操作。"""

    def __init__(self):
        self.graph = SkillGraph()
        self.graph.build()

    def improve_skill(self, skill_name: str) -> Dict[str, Any]:
        """基于失败模式生成改进建议。"""
        if not FAILURES_FILE.exists():
            return {
                "skill": skill_name,
                "improvements": [],
                "message": "No failure data available",
            }

        # Read failures for this skill
        failures = []
        try:
            with open(FAILURES_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get("skill") == skill_name:
                            failures.append(entry)
                    except json.JSONDecodeError:
                        continue
        except (OSError, IOError):
            pass

        if not failures:
            return {
                "skill": skill_name,
                "improvements": [],
                "message": f"No failures recorded for {skill_name}",
            }

        # Analyze failure patterns
        categories = defaultdict(int)
        for f in failures:
            categories[f.get("category", "unknown")] += 1

        improvements = []
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            if cat == "execution_error":
                improvements.append({
                    "type": "error_handling",
                    "suggestion": f"Add error handling for {count} execution errors",
                    "priority": "high" if count >= 3 else "medium",
                })
            elif cat == "insufficient":
                improvements.append({
                    "type": "completeness",
                    "suggestion": f"Expand content for {count} insufficient coverage cases",
                    "priority": "high",
                })
            elif cat == "missing_skill":
                improvements.append({
                    "type": "coverage",
                    "suggestion": f"Add missing scenarios ({count} cases)",
                    "priority": "medium",
                })

        return {
            "skill": skill_name,
            "total_failures": len(failures),
            "categories": dict(categories),
            "improvements": improvements,
        }

    def merge_skills(self, skill_a: str, skill_b: str) -> Dict[str, Any]:
        """合并两个重叠的 skill。"""
        info_a = self.graph.skill_info.get(skill_a, {})
        info_b = self.graph.skill_info.get(skill_b, {})

        if not info_a or not info_b:
            return {
                "error": f"Skill not found: {skill_a if not info_a else skill_b}",
                "merged": False,
            }

        # Check overlap
        provides_a = set(info_a.get("provides", []))
        provides_b = set(info_b.get("provides", []))
        overlap = provides_a & provides_b

        if not overlap:
            return {
                "merged": False,
                "reason": "No overlapping provides",
                "skill_a_provides": list(provides_a),
                "skill_b_provides": list(provides_b),
            }

        # Calculate merge score
        union = provides_a | provides_b
        jaccard = len(overlap) / len(union) if union else 0

        # Merge
        merged_provides = list(union)
        merged_requires = list(
            set(info_a.get("requires", []) + info_b.get("requires", []))
        )
        merged_description = f"Merged from {skill_a} and {skill_b}"

        # Generate merged SKILL.md
        skill_md = f"""---
name: merged-{skill_a}-{skill_b}
description: "{merged_description}"
auto_generated: true
source: skill_composer
merged_from: [{skill_a}, {skill_b}]
---

# Merged: {skill_a} + {skill_b}

{merged_description}

## Provides

{chr(10).join(f'- {p}' for p in merged_provides)}

## Requires

{chr(10).join(f'- {r}' for r in merged_requires)}

## Notes

- Auto-merged from two overlapping skills
- Overlap: {', '.join(overlap)}
- Jaccard similarity: {jaccard:.2f}
"""

        return {
            "merged": True,
            "name": f"merged-{skill_a}-{skill_b}",
            "skill_md": skill_md,
            "provides": merged_provides,
            "requires": merged_requires,
            "overlap": list(overlap),
            "jaccard": round(jaccard, 3),
        }


# ===========================================================================
# Public API
# ===========================================================================

def build_graph_and_find_paths(
    goal_provides: str, max_depth: int = 3
) -> str:
    """Build DAG and find paths. JSON string result."""
    graph = SkillGraph()
    graph.build()
    goals = [g.strip() for g in goal_provides.split(",") if g.strip()]
    paths = graph.find_paths(goals, max_depth=max_depth)
    return json.dumps({
        "goal": goals,
        "paths_found": len(paths),
        "paths": paths,
    }, ensure_ascii=False, indent=2)


def compose_skill(operation: str, skill_a: str, skill_b: str = "") -> str:
    """Compose skills (improve/merge). JSON string result."""
    composer = SkillComposer()
    if operation == "improve":
        result = composer.improve_skill(skill_a)
    elif operation == "merge" and skill_b:
        result = composer.merge_skills(skill_a, skill_b)
    else:
        result = {"error": f"Unknown operation: {operation}"}
    return json.dumps(result, ensure_ascii=False, indent=2)
