"""GRPO-lite — SE15 分层优化工具。

不是完整 GRPO 训练，不做梯度、不更新模型权重。
实现低风险推理期评估层：
1. 轨迹分组评分
2. 组内相对优势 A=(r-mean)/std
3. KL-like 偏离惩罚（基于策略词集 Jaccard 距离）
4. 策略推荐/淘汰建议
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from tools.registry import registry


def _tokens(text: str) -> set[str]:
    """中文单字+双字组+英文词，作为无embedding的轻量策略表示。"""
    text = (text or "").lower()
    chars = re.findall(r'[\u4e00-\u9fff]', text)
    bigrams = {chars[i] + chars[i + 1] for i in range(len(chars) - 1)}
    words = set(re.findall(r'[a-z]+', text))
    return set(chars) | bigrams | words


def jaccard_distance(a: str, b: str) -> float:
    at, bt = _tokens(a), _tokens(b)
    union = at | bt
    if not union:
        return 0.0
    return 1.0 - len(at & bt) / len(union)


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 1.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return math.sqrt(var) or 1.0


def score_trajectory(t: Dict[str, Any]) -> float:
    """轨迹奖励函数 r。

    输入字段（可选）：
    - success: bool
    - completed: int
    - failed: int
    - verified: bool
    - tool_calls: int
    - errors: int
    - evidence_grade: A/B/C/D
    - duration_seconds: float
    """
    success = 1.0 if t.get('success') else 0.0
    completed = float(t.get('completed', 0) or 0)
    failed = float(t.get('failed', 0) or 0)
    verified = 1.0 if t.get('verified') else 0.0
    tool_calls = float(t.get('tool_calls', t.get('tools', 0)) or 0)
    errors = float(t.get('errors', 0) or 0)
    duration = float(t.get('duration_seconds', 0) or 0)
    grade = str(t.get('evidence_grade', '') or '').upper()
    grade_bonus = {'A': 0.25, 'B': 0.15, 'C': 0.05, 'D': -0.1}.get(grade, 0.0)

    done_rate = completed / max(completed + failed, 1.0)
    efficiency_penalty = min(duration / 600.0, 0.2) + min(tool_calls / 100.0, 0.1)
    error_penalty = min(errors * 0.1, 0.4)

    r = 0.35 * success + 0.25 * done_rate + 0.2 * verified + grade_bonus - efficiency_penalty - error_penalty
    return max(-1.0, min(1.0, r))


def grpo_lite_score(group: List[Dict[str, Any]], reference_policy: str = '', beta: float = 0.1) -> Dict[str, Any]:
    """对一组轨迹做 GRPO-lite 评分。"""
    if not group:
        return {'success': False, 'error': 'group is empty'}

    rewards = [score_trajectory(t) for t in group]
    m = _mean(rewards)
    s = _std(rewards)

    rows = []
    for idx, t in enumerate(group):
        policy = str(t.get('policy', t.get('strategy', t.get('name', f'policy_{idx}'))))
        reward = rewards[idx]
        advantage = (reward - m) / s
        kl_like = jaccard_distance(policy, reference_policy) if reference_policy else 0.0
        objective = advantage - beta * kl_like
        rows.append({
            'index': idx,
            'name': t.get('name', f'trajectory_{idx}'),
            'policy': policy[:160],
            'reward': round(reward, 4),
            'advantage': round(advantage, 4),
            'kl_like': round(kl_like, 4),
            'objective': round(objective, 4),
            'recommendation': 'promote' if objective > 0.5 else ('keep' if objective >= -0.3 else 'retire'),
        })

    rows.sort(key=lambda x: -x['objective'])
    return {
        'success': True,
        'mode': 'grpo_lite_score',
        'group_size': len(group),
        'reward_mean': round(m, 4),
        'reward_std': round(s, 4),
        'beta': beta,
        'reference_policy': reference_policy[:160],
        'ranking': rows,
        'best': rows[0] if rows else None,
        'boundary': 'GRPO-lite only: no gradient, no model weight update, no full GRPO training',
    }


def recommend_strategy(candidates: List[Dict[str, Any]], reference_policy: str = '', beta: float = 0.1) -> Dict[str, Any]:
    """从候选策略/轨迹中推荐最佳策略。"""
    scored = grpo_lite_score(candidates, reference_policy=reference_policy, beta=beta)
    if not scored.get('success'):
        return scored
    return {
        'success': True,
        'recommended': scored['best'],
        'ranking': scored['ranking'],
        'summary': f"推荐 {scored['best']['name']} objective={scored['best']['objective']} reward={scored['best']['reward']}",
        'boundary': scored['boundary'],
    }


def grpo_lite_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Hermes 工具入口。

    Args:
        action: 'score' | 'recommend' | 'ping'
        group/candidates: 轨迹列表
        reference_policy: 参考策略文本
        beta: KL-like 惩罚系数，默认0.1
    """
    action = args.get('action', 'ping')
    beta = float(args.get('beta', 0.1))
    ref = str(args.get('reference_policy', '') or '')

    if action == 'ping':
        return {
            'success': True,
            'tool': 'grpo_lite',
            'features': ['group_reward_scoring', 'relative_advantage', 'kl_like_penalty', 'strategy_recommendation'],
            'boundary': 'no training, no gradients, no model weight updates',
        }
    if action == 'score':
        group = args.get('group') or args.get('trajectories') or []
        return grpo_lite_score(group, reference_policy=ref, beta=beta)
    if action == 'recommend':
        candidates = args.get('candidates') or args.get('group') or []
        return recommend_strategy(candidates, reference_policy=ref, beta=beta)

    return {'success': False, 'error': f'unknown action: {action}'}


registry.register(
    name="grpo_lite",
    toolset="skills",
    schema={
        "name": "grpo_lite",
        "description": "GRPO-lite 分层优化：组评分、相对优势、KL-like 惩罚、策略推荐。不训练模型不改权重。",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["ping", "score", "recommend"]},
                "group": {"type": "array", "description": "轨迹列表"},
                "candidates": {"type": "array", "description": "候选策略/轨迹列表"},
                "reference_policy": {"type": "string", "description": "参考策略文本"},
                "beta": {"type": "number", "description": "KL-like 惩罚系数"}
            }
        }
    },
    handler=grpo_lite_handler,
)
