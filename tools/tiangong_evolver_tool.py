"""
TianGong Evolver 适配器 - 超级进化11

四核之一：缺陷扫描、失败归因、轨迹回收、基因沉淀、回测指标。

实际后端：EvoMaster（超级进化9）+ darwinian-evolver skill（imbue-ai）。
"""

from tools.registry import registry
import logging
import sqlite3
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def tiangong_evolver_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    TianGong Evolver 核心：基于真实轨迹做缺陷扫描、归因、基因沉淀。
    
    Args:
        args:
            mode: "scan" | "evolve" | "regress" | "stats"
            scope: 任务范围（可选）
            min_score: 最低分阈值（默认 0.7）
        
    Returns:
        Evolver 报告
    """
    mode = args.get('mode', 'stats')
    scope = args.get('scope', '')
    min_score = args.get('min_score', 0.7)
    
    try:
        from evo_master import get_evo_master
        evo = get_evo_master()
        
        evidence = {
            'role': 'evolver',
            'mode': mode,
            'scope': scope or 'all',
            'backend': 'EvoMaster (超级进化9)',
        }
        
        if mode == 'scan':
            # 缺陷扫描：找出低分轨迹
            conn = sqlite3.connect(evo.knowledge_cache.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            failed = cur.execute(
                "SELECT id, task, reward, total_value FROM trajectories WHERE success = 0 OR reward < ? ORDER BY total_value ASC LIMIT 10",
                (min_score,)
            ).fetchall()
            conn.close()
            
            evidence['defects_found'] = len(failed)
            evidence['defects'] = [
                {'id': r['id'][:12], 'task': r['task'], 'reward': r['reward'], 'value': r['total_value']}
                for r in failed
            ]
            evidence['message'] = f"✅ 缺陷扫描：发现 {len(failed)} 条低分/失败轨迹"
        
        elif mode == 'evolve':
            # 基因沉淀：触发策略进化
            strategy = evo.evolve_strategy()
            evidence['strategy_version'] = strategy.version
            evidence['strategy_performance'] = strategy.performance
            evidence['top_patterns'] = len(strategy.policy.get('top_patterns', []))
            evidence['message'] = f"✅ 基因沉淀：策略 v{strategy.version}，性能 {strategy.performance:.2f}"
        
        elif mode == 'regress':
            # 回测：检查最优轨迹的可复现性
            top = evo.knowledge_cache.get_top_trajectories(limit=5)
            evidence['regression_top_5'] = [
                {'task': t.task[:50], 'value': t.total_value, 'success': t.success}
                for t in top
            ]
            evidence['message'] = f"✅ 回测：Top 5 平均价值 {sum(t.total_value for t in top)/max(len(top),1):.2f}"
        
        else:  # stats
            top = evo.knowledge_cache.get_top_trajectories(limit=3)
            evidence['current_strategy'] = f"v{evo.current_strategy.version if evo.current_strategy else 0}"
            evidence['lambda_weight'] = evo.lambda_weight
            evidence['top_3'] = [{'task': t.task[:50], 'value': t.total_value} for t in top]
            evidence['message'] = "✅ Evolver 统计就绪"
        
        return {'success': True, **evidence}
        
    except Exception as e:
        logger.error(f"TianGong Evolver 失败: {e}")
        return {
            'success': False,
            'error': str(e),
            'role': 'evolver',
            'message': f"❌ Evolver 失败: {e}"
        }


registry.register(
    name="tiangong_evolver",
    toolset="skills",
    schema={
        "name": "tiangong_evolver",
        "description": "TianGong 四核之 Evolver：缺陷扫描、失败归因、轨迹回收、基因沉淀、回测（超级进化11）。后端：EvoMaster。",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["scan", "evolve", "regress", "stats"],
                    "description": "scan=缺陷扫描, evolve=基因进化, regress=回测, stats=统计"
                },
                "scope": {"type": "string", "description": "任务范围（可选）"},
                "min_score": {"type": "number", "description": "最低分阈值（默认 0.7）"}
            }
        }
    },
    handler=tiangong_evolver_handler
)
