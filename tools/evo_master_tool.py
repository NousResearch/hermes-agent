"""
EvoMaster 工具 - 超级进化9

CLAW 原生进化核心引擎工具接口。

核心公式：
1. 迭代进化总目标：max E[R_exec(τ) + λ·K_claw(τ)]
2. 策略自更新：π^(t+1) = GPT-Stream(τ^(t), K_claw, Constraint)
3. 知识缓存压缩：K_claw = HashPool(Filter(τ_valid))
"""

from tools.registry import registry
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def evo_master_select_strategy_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据任务类型自动选择最优策略和动作序列（跨会话复用）
    
    Args:
        args: {task: 任务描述}
        
    Returns:
        策略选择结果
    """
    task = args.get('task', '')
    if not task:
        return {'success': False, 'error': 'task 参数不能为空'}
    
    try:
        import sys
        sys.path.insert(0, '/Users/appleoppa/.hermes/scripts')
        from hermes_strategy_selector import StrategySelector
        
        selector = StrategySelector()
        result = selector.select_strategy(task)
        
        return {
            'success': True,
            **result
        }
        
    except Exception as e:
        logger.error(f"策略选择失败: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f"❌ 策略选择失败: {e}"
        }


def evo_master_score_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    使用 Rust 评估中心对知识缓存中的轨迹评分
    
    Args:
        args: 参数字典
        
    Returns:
        评分统计
    """
    try:
        from evo_master import get_evo_master
        
        evo = get_evo_master()
        result = evo.score_trajectories_with_eval_center()
        
        return {
            'success': True,
            'scored': result['scored'],
            'skipped': result['skipped'],
            'total': result['total'],
            'message': f"✅ 评分完成: {result['scored']}/{result['total']} 条轨迹（跳过 {result['skipped']}）"
        }
        
    except Exception as e:
        logger.error(f"EvoMaster 评分失败: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f"❌ 评分失败: {e}"
        }


def evo_master_import_handler(
    args: Dict[str, Any]
) -> Dict[str, Any]:
    """
    从 Rust 评估中心导入真实轨迹到 EvoMaster 知识缓存
    
    Args:
        args: 参数字典，包含 limit, min_score, only_active
        
    Returns:
        导入统计
    """
    limit = args.get('limit', 100)
    min_score = args.get('min_score', 0.0)
    only_active = args.get('only_active', False)
    
    try:
        from evo_master import get_evo_master
        
        evo = get_evo_master()
        result = evo.import_from_eval_center(
            limit=limit,
            min_score=min_score,
            only_active=only_active
        )
        
        return {
            'success': True,
            'imported': result['imported'],
            'duplicate': result['duplicate'],
            'skipped': result['skipped'],
            'total': result['total'],
            'message': f"✅ 从评估中心导入 {result['imported']}/{result['total']} 条轨迹 (重复 {result['duplicate']}, 跳过 {result['skipped']})"
        }
        
    except Exception as e:
        logger.error(f"EvoMaster 导入失败: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f"❌ 导入失败: {e}"
        }


def evo_master_evolve_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行策略进化
    
    实现公式：π^(t+1) = GPT-Stream(τ^(t), K_claw, Constraint)
    
    Args:
        args: 参数字典（空）
        
    Returns:
        新策略信息
    """
    try:
        from evo_master import get_evo_master
        
        evo = get_evo_master()
        strategy = evo.evolve_strategy()
        
        return {
            'success': True,
            'strategy_id': strategy.id,
            'version': strategy.version,
            'performance': strategy.performance,
            'policy_type': strategy.policy.get('type'),
            'top_patterns_count': len(strategy.policy.get('top_patterns', [])),
            'message': f"✅ 策略进化完成: v{strategy.version} (性能: {strategy.performance:.2f})"
        }
        
    except Exception as e:
        logger.error(f"EvoMaster 策略进化失败: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f"❌ 策略进化失败: {e}"
        }


def evo_master_recommend_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    基于知识缓存推荐动作序列
    
    Args:
        args: 参数字典，包含 task
        
    Returns:
        推荐的动作序列
    """
    task = args.get('task', '')
    
    try:
        from evo_master import get_evo_master
        
        evo = get_evo_master()
        actions = evo.get_recommended_actions(task)
        
        return {
            'success': True,
            'task': task,
            'recommended_actions': actions,
            'actions_count': len(actions),
            'message': f"✅ 为任务 '{task}' 推荐了 {len(actions)} 个动作"
        }
        
    except Exception as e:
        logger.error(f"EvoMaster 推荐失败: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f"❌ 推荐失败: {e}"
        }


def evo_master_stats_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取 EvoMaster 统计信息
    
    Args:
        args: 参数字典（空）
        
    Returns:
        统计信息
    """
    try:
        from evo_master import get_evo_master
        
        evo = get_evo_master()
        top_trajs = evo.knowledge_cache.get_top_trajectories(limit=5)
        
        strategy_version = evo.current_strategy.version if evo.current_strategy else 0
        strategy_performance = evo.current_strategy.performance if evo.current_strategy else 0.0
        
        return {
            'success': True,
            'current_strategy_version': strategy_version,
            'current_strategy_performance': strategy_performance,
            'lambda_weight': evo.lambda_weight,
            'top_trajectories': [
                {
                    'task': t.task,
                    'total_value': t.total_value,
                    'reward': t.reward,
                    'knowledge_value': t.knowledge_value,
                    'success': t.success
                }
                for t in top_trajs
            ],
            'message': f"✅ 当前策略 v{strategy_version} (性能: {strategy_performance:.2f})"
        }
        
    except Exception as e:
        logger.error(f"EvoMaster 统计失败: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f"❌ 统计失败: {e}"
        }


# 注册工具
registry.register(
    name="evo_master_select_strategy",
    toolset="skills",
    schema={
        "name": "evo_master_select_strategy",
        "description": "根据任务类型自动选择最优策略和推荐动作序列（超级进化9，跨会话复用）。基于知识缓存中的相似任务匹配，返回最佳策略版本和动作。",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "任务描述"
                }
            },
            "required": ["task"]
        }
    },
    handler=evo_master_select_strategy_handler
)

registry.register(
    name="evo_master_score",
    toolset="skills",
    schema={
        "name": "evo_master_score",
        "description": "使用 Rust 评估中心对 EvoMaster 知识缓存中的轨迹评分（超级进化9）。调用 hermes_eval_center.score_trace() 更新 reward 和 total_value。",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    handler=evo_master_score_handler
)

registry.register(
    name="evo_master_import",
    toolset="skills",
    schema={
        "name": "evo_master_import",
        "description": "从 Rust 评估中心导入真实工具调用轨迹到 EvoMaster 知识缓存（超级进化9）。实现公式 K_claw = HashPool(Filter(τ_valid))。",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "最大导入数量（默认 100）"
                },
                "min_score": {
                    "type": "number",
                    "description": "最低评分阈值（默认 0.0）"
                },
                "only_active": {
                    "type": "boolean",
                    "description": "仅导入 state='active' 的轨迹（默认 False）"
                }
            }
        }
    },
    handler=evo_master_import_handler
)

registry.register(
    name="evo_master_evolve",
    toolset="skills",
    schema={
        "name": "evo_master_evolve",
        "description": "执行 EvoMaster 策略进化（超级进化9）。实现公式 π^(t+1) = GPT-Stream(τ^(t), K_claw, Constraint)。基于知识缓存中的最优轨迹生成新策略。",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    handler=evo_master_evolve_handler
)

registry.register(
    name="evo_master_recommend",
    toolset="skills",
    schema={
        "name": "evo_master_recommend",
        "description": "基于 EvoMaster 知识缓存推荐动作序列（超级进化9）。查找相似任务的成功轨迹，返回推荐的动作序列。",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "任务描述"
                }
            },
            "required": ["task"]
        }
    },
    handler=evo_master_recommend_handler
)

registry.register(
    name="evo_master_stats",
    toolset="skills",
    schema={
        "name": "evo_master_stats",
        "description": "获取 EvoMaster 统计信息（超级进化9）。返回当前策略版本、性能、最优轨迹等信息。",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    handler=evo_master_stats_handler
)
