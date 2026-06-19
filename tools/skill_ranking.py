"""Skill Ranking — Thompson Sampling for skill selection.

Uses Beta distribution sampling to balance exploration vs exploitation
when selecting skills. Replaces static ranking with probabilistic
ranking that improves over time as success/failure data accumulates.

No numpy required — uses Python's built-in `random.betavariate`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ===========================================================================
# ThompsonSampler
# ===========================================================================

class ThompsonSampler:
    """基于 Beta 分布的技能选择。"""

    def __init__(self, skill_name: str = "", alpha: int = 1, beta: int = 1):
        self.skill_name = skill_name
        self.alpha = alpha  # successes + 1
        self.beta = beta    # failures + 1

    def sample(self) -> float:
        """从 Beta(α, β) 采样一个分数 (0-1)。"""
        return random.betavariate(self.alpha, self.beta)

    def expected_value(self) -> float:
        """期望值 = α / (α + β)。"""
        total = self.alpha + self.beta
        return self.alpha / total if total > 0 else 0.5

    def confidence(self) -> float:
        """置信度 = 1 / (α + β)。数据越多越确信。"""
        total = self.alpha + self.beta
        return 1.0 / total if total > 0 else 1.0


# ===========================================================================
# SkillRanker
# ===========================================================================

class SkillRanker:
    """使用 Thompson Sampling 对 skills 排序。"""

    def __init__(self):
        self.usage_file = Path.home() / ".hermes" / "skills" / ".skill-index" / ".usage.json"

    def _load_usage(self) -> Dict[str, Any]:
        """加载使用数据。"""
        if not self.usage_file.exists():
            return {}
        try:
            return json.loads(self.usage_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _build_samplers(self) -> Dict[str, ThompsonSampler]:
        """从使用数据构建 Thompson samplers。"""
        usage = self._load_usage()
        samplers = {}

        for skill_name, data in usage.items():
            if not isinstance(data, dict):
                continue
            success_rate = data.get("success_rate", 0.5)
            total_outcomes = data.get("total_outcomes", 0)

            # Convert success_rate + total_outcomes to alpha/beta
            if total_outcomes > 0:
                successes = int(success_rate * total_outcomes)
                failures = total_outcomes - successes
            else:
                # No data: use uniform prior
                successes = 0
                failures = 0

            samplers[skill_name] = ThompsonSampler(
                skill_name=skill_name,
                alpha=successes + 1,
                beta=failures + 1,
            )

        return samplers

    def rank_skills(
        self,
        candidates: List[Dict[str, Any]],
        similarity_scores: Optional[Dict[str, float]] = None,
        top_k: int = 10,
        n_samples: int = 100,
    ) -> List[Dict[str, Any]]:
        """使用 Thompson Sampling 排序 skills。

        Args:
            candidates: [{"name": "...", "score": 0.8, ...}]
            similarity_scores: {"skill-name": cosine_similarity}
            top_k: 返回前 K 个
            n_samples: 采样次数（越多越稳定）

        Returns:
            [{"name": "...", "rank_score": 0.75, "thompson": 0.8,
              "similarity": 0.6, "confidence": 0.3}]
        """
        samplers = self._build_samplers()

        results = []
        for skill in candidates:
            name = skill.get("name", "")
            sim_score = skill.get("score", 0.5)
            if similarity_scores and name in similarity_scores:
                sim_score = similarity_scores[name]

            sampler = samplers.get(name)
            if sampler:
                # Multiple samples for stability
                thompson_scores = [sampler.sample() for _ in range(n_samples)]
                avg_thompson = sum(thompson_scores) / len(thompson_scores)
                confidence = sampler.confidence()
            else:
                # No data: use uniform prior (0.5)
                avg_thompson = 0.5
                confidence = 1.0  # High uncertainty

            # Combined score: similarity × Thompson sample
            rank_score = sim_score * avg_thompson

            results.append({
                "name": name,
                "rank_score": round(rank_score, 4),
                "thompson": round(avg_thompson, 4),
                "similarity": round(sim_score, 4),
                "confidence": round(confidence, 4),
                "expected_value": round(sampler.expected_value(), 4) if sampler else 0.5,
            })

        # Sort by rank_score descending
        results.sort(key=lambda x: x["rank_score"], reverse=True)

        return results[:top_k]

    def get_skill_stats(self, skill_name: str) -> Dict[str, Any]:
        """获取单个 skill 的 Thompson 统计。"""
        samplers = self._build_samplers()
        sampler = samplers.get(skill_name)
        if not sampler:
            return {
                "skill": skill_name,
                "has_data": False,
                "alpha": 1,
                "beta": 1,
                "expected_value": 0.5,
                "confidence": 1.0,
            }

        return {
            "skill": skill_name,
            "has_data": True,
            "alpha": sampler.alpha,
            "beta": sampler.beta,
            "expected_value": round(sampler.expected_value(), 4),
            "confidence": round(sampler.confidence(), 4),
            "sample": round(sampler.sample(), 4),
        }


# ===========================================================================
# Public API
# ===========================================================================

def thompson_rank(
    candidates_json: str, top_k: int = 10
) -> str:
    """Rank skills using Thompson Sampling. JSON string result."""
    try:
        candidates = json.loads(candidates_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON candidates"}, ensure_ascii=False)

    ranker = SkillRanker()
    results = ranker.rank_skills(candidates, top_k=top_k)
    return json.dumps({
        "ranked": len(results),
        "results": results,
    }, ensure_ascii=False, indent=2)


def thompson_stats(skill_name: str) -> str:
    """Get Thompson stats for a skill. JSON string result."""
    ranker = SkillRanker()
    stats = ranker.get_skill_stats(skill_name)
    return json.dumps(stats, ensure_ascii=False, indent=2)
