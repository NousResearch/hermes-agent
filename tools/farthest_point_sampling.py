"""最远点采样 — 策略多样性维护模块。

在策略嵌入空间中维护候选池，每次选取与其他候选距离最大的策略。
保证探索多样性，防止策略收敛到同质化。

距离算法：Jaccard 距离（基于词集），无需 embedding 基础设施。
"""
import re
import math
from typing import Any, Dict, List, Set, Tuple
from collections import defaultdict


class FarthestPointSampler:
    """最远点采样器 — 维护策略池并按最大最小距离选点。"""

    def __init__(self, pool: Dict[str, Set[str]] | None = None):
        """
        Args:
            pool: 策略池 {策略名: 特征词集}
        """
        self._pool: Dict[str, Set[str]] = pool or {}
        self._metric_name = "jaccard"

    def _tokenize(self, text: str) -> Set[str]:
        """将文本拆分为特征词集。"""
        text = text.lower()
        # 拆中文：单字 + 双字组
        chars = re.findall(r'[\u4e00-\u9fff]', text)
        bigrams = set()
        for i in range(len(chars) - 1):
            bigrams.add(chars[i] + chars[i+1])
        # 拆英文单词
        words = set(re.findall(r'[a-z]+', text))
        return bigrams | words

    def add_strategy(self, name: str, description: str) -> bool:
        """添加策略到池。"""
        features = self._tokenize(description)
        if name in self._pool:
            return False
        self._pool[name] = features
        return True

    def update_strategy(self, name: str, description: str) -> bool:
        """更新池中已有策略的特征。"""
        if name not in self._pool:
            return self.add_strategy(name, description)
        self._pool[name] = self._tokenize(description)
        return True

    def remove_strategy(self, name: str) -> bool:
        """从池中移除策略。"""
        return self._pool.pop(name, None) is not None

    def jaccard_distance(self, a: Set[str], b: Set[str]) -> float:
        """Jaccard 距离 = 1 - |A∩B|/|A∪B|。"""
        union = a | b
        if not union:
            return 0.0
        return 1.0 - len(a & b) / len(union)

    def min_distance_to_pool(self, candidate: Set[str]) -> float:
        """候选点与池中所有点的最小距离。"""
        if not self._pool:
            return 1.0  # 空池时最大距离
        return min(self.jaccard_distance(candidate, feat) for feat in self._pool.values())

    def select_farthest(self, candidates: List[Tuple[str, str]], top_k: int = 1) -> List[Tuple[str, float]]:
        """从候选列表中选距离池最远的 top_k 个策略。

        Args:
            candidates: [(策略名, 描述), ...]
            top_k: 返回前 k 个最远点

        Returns:
            [(策略名, 距离分数), ...]
        """
        if not candidates:
            return []

        scored = []
        for name, desc in candidates:
            features = self._tokenize(desc)
            dist = self.min_distance_to_pool(features)
            scored.append((name, dist))

        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def pool_stats(self) -> Dict[str, Any]:
        """返回策略池统计信息。"""
        return {
            'pool_size': len(self._pool),
            'metric': self._metric_name,
            'strategies': list(self._pool.keys()),
            'avg_feature_size': round(sum(len(v) for v in self._pool.values()) / max(len(self._pool), 1), 1),
        }


# 全局单例
_sampler: FarthestPointSampler | None = None


def get_sampler() -> FarthestPointSampler:
    global _sampler
    if _sampler is None:
        _sampler = FarthestPointSampler()
    return _sampler


def fps_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """最远点采样工具处理函数。

    Args:
        action: 'add' | 'select' | 'stats' | 'remove' | 'reset'
        name: 策略名
        description: 策略描述
        candidates: select 模式下 [[name, desc], ...]
        top_k: 返回最远 top_k 个（默认 1）
    """
    sampler = get_sampler()
    action = args.get('action', 'stats')

    if action == 'add':
        name = args.get('name', '')
        desc = args.get('description', '')
        if not name or not desc:
            return {'success': False, 'error': '需要 name 和 description 参数'}
        sampler.add_strategy(name, desc)
        return {
            'success': True,
            'action': 'add',
            'name': name,
            'pool_size': len(sampler._pool),
        }

    elif action == 'select':
        candidates_raw = args.get('candidates', [])
        if not candidates_raw:
            return {'success': False, 'error': '需要 candidates 参数: [[name, desc], ...]'}
        top_k = int(args.get('top_k', 1))
        result = sampler.select_farthest(candidates_raw, top_k=top_k)
        return {
            'success': True,
            'action': 'select',
            'selected': [{'name': n, 'distance': round(d, 4)} for n, d in result],
            'top_k': top_k,
            'pool_size': len(sampler._pool),
        }

    elif action == 'remove':
        name = args.get('name', '')
        sampler.remove_strategy(name)
        return {'success': True, 'action': 'remove', 'name': name}

    elif action == 'reset':
        global _sampler
        _sampler = FarthestPointSampler()
        return {'success': True, 'action': 'reset', 'message': '策略池已清空'}

    else:  # stats (default)
        return {
            'success': True,
            'action': 'stats',
            **sampler.pool_stats(),
        }


# Hermes 工具注册
try:
    from tools.registry import registry
    registry.register(
        name="farthest_point_sampling",
        toolset="skills",
        schema={
            "name": "farthest_point_sampling",
            "description": "最远点采样：维护策略多样性池，选择距离已有策略最远的候选策略",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "select", "stats", "remove", "reset"],
                        "description": "add=加入策略, select=选择最远候选, stats=统计, remove=删除, reset=清空"
                    },
                    "name": {"type": "string", "description": "策略名"},
                    "description": {"type": "string", "description": "策略描述"},
                    "candidates": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "string"}},
                        "description": "候选策略 [[name, desc], ...]"
                    },
                    "top_k": {"type": "integer", "description": "返回前 k 个"}
                }
            }
        },
        handler=fps_handler,
    )
except Exception:
    # 工具注册失败不影响模块直接导入/单元测试
    pass