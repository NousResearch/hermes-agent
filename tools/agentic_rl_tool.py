"""
Agentic RL 五层能力工具集 - 超级进化12

实现公式：
- RL_base: π(a|s) → R → ∇π
- APEX_ARL: RL ∪ {MetaG, Reflect, LongPlan}
- I_total = M_base × C_think
- C_think = G_set + P_decompose + S_review

五层能力：
1. 自主目标（G_self ≠ G_env）
2. 周期规划（P_n = Split(G_total)）
3. 动态策略（π_t = f(π_{t-1}, ΔE)）
4. 元推理（R_meta = Eval(Logic)）
5. 闭环自省（S_fix = Error → Policy）
"""

from tools.registry import registry
import logging
import sqlite3
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================
# 第 1 层：自主目标（Self Goal）
# ============================================================

def agentic_rl_self_goal_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    自主目标生成：G_self ≠ G_env
    
    根据环境上下文和历史轨迹，生成自主目标（不是被动响应环境）。
    
    Args:
        args:
            env_context: 环境上下文
            mode: "explore" | "exploit" | "balance"
    """
    env_context = args.get('env_context', '')
    mode = args.get('mode', 'balance')
    
    try:
        from evo_master import get_evo_master
        evo = get_evo_master()
        
        # 从知识缓存中分析当前能力空白
        top = evo.knowledge_cache.get_top_trajectories(limit=10)
        known_tasks = set(t.task[:30] for t in top)
        
        # 基于模式生成自主目标
        self_goals = []
        
        if mode == 'explore':
            self_goals = [
                {'goal': '探索新任务类型', 'reason': '突破当前能力边界', 'priority': 'high'},
                {'goal': '尝试低分轨迹的改进', 'reason': '弥补失败模式', 'priority': 'medium'},
                {'goal': '学习新工具组合', 'reason': '扩展工具协同', 'priority': 'medium'},
            ]
        elif mode == 'exploit':
            self_goals = [
                {'goal': f'优化任务: {list(known_tasks)[0] if known_tasks else "通用任务"}', 'reason': '基于 Top 轨迹深化', 'priority': 'high'},
                {'goal': '提升策略性能', 'reason': '当前策略 v2，目标 v3', 'priority': 'high'},
            ]
        else:  # balance
            self_goals = [
                {'goal': '巩固已知任务的成功模式', 'reason': '稳固基础能力', 'priority': 'high'},
                {'goal': '探索 1-2 个新任务领域', 'reason': '保持扩展性', 'priority': 'medium'},
                {'goal': '改进失败模式', 'reason': '降低 G_self 与 G_env 偏差', 'priority': 'medium'},
            ]
        
        return {
            'success': True,
            'role': 'self_goal',
            'mode': mode,
            'env_context': env_context[:200],
            'self_goals': self_goals,
            'goal_count': len(self_goals),
            'known_task_count': len(known_tasks),
            'formula': 'G_self ≠ G_env',
            'message': f"✅ 已生成 {len(self_goals)} 个自主目标（{mode} 模式）"
        }
    except Exception as e:
        return {'success': False, 'role': 'self_goal', 'error': str(e)}


# ============================================================
# 第 2 层：周期规划（Long Plan）
# ============================================================

def agentic_rl_long_plan_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    周期规划：P_n = Split(G_total)
    
    将长周期目标分解为多阶段子目标。
    
    Args:
        args:
            total_goal: 总目标
            horizon: "short" (1天) | "medium" (1周) | "long" (1月)
    """
    total_goal = args.get('total_goal', '')
    if not total_goal:
        return {'success': False, 'error': 'total_goal 不能为空'}
    
    horizon = args.get('horizon', 'medium')
    
    horizons = {
        'short': {'days': 1, 'phases': 3},
        'medium': {'days': 7, 'phases': 5},
        'long': {'days': 30, 'phases': 7},
    }
    h = horizons.get(horizon, horizons['medium'])
    
    # 标准 5 阶段拆解模板
    phase_templates = [
        {'phase': 1, 'name': '调研与定义', 'percent': 15},
        {'phase': 2, 'name': '设计与规划', 'percent': 20},
        {'phase': 3, 'name': '核心实现', 'percent': 35},
        {'phase': 4, 'name': '验证与测试', 'percent': 20},
        {'phase': 5, 'name': '沉淀与交付', 'percent': 10},
    ]
    
    plan = []
    cumulative_days = 0
    for tmpl in phase_templates[:h['phases']]:
        phase_days = max(1, int(h['days'] * tmpl['percent'] / 100))
        plan.append({
            'phase': tmpl['phase'],
            'name': tmpl['name'],
            'duration_days': phase_days,
            'cumulative_day': cumulative_days + phase_days,
            'sub_goal': f"{tmpl['name']}: 围绕「{total_goal[:30]}」",
            'milestone': f"Phase {tmpl['phase']} 完成"
        })
        cumulative_days += phase_days
    
    return {
        'success': True,
        'role': 'long_plan',
        'total_goal': total_goal,
        'horizon': horizon,
        'total_days': h['days'],
        'phase_count': len(plan),
        'plan': plan,
        'formula': 'P_n = Split(G_total)',
        'message': f"✅ 已生成 {len(plan)} 阶段规划（{h['days']} 天）"
    }


# ============================================================
# 第 3 层：动态策略（Dynamic Policy）
# ============================================================

def agentic_rl_dynamic_policy_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    动态策略：π_t = f(π_{t-1}, ΔE)
    
    根据环境变化和历史策略动态调整。直接调用 EvoMaster。
    
    Args:
        args:
            delta_e: 环境变化描述
            trigger_evolve: 是否触发进化
    """
    delta_e = args.get('delta_e', '')
    trigger_evolve = args.get('trigger_evolve', False)
    
    try:
        from evo_master import get_evo_master
        evo = get_evo_master()
        
        prev_strategy = evo.current_strategy
        prev_version = prev_strategy.version if prev_strategy else 0
        prev_performance = prev_strategy.performance if prev_strategy else 0.0
        
        evidence = {
            'role': 'dynamic_policy',
            'delta_e': delta_e[:200],
            'previous_strategy': f"v{prev_version}",
            'previous_performance': prev_performance,
        }
        
        if trigger_evolve:
            new_strategy = evo.evolve_strategy()
            evidence['new_strategy'] = f"v{new_strategy.version}"
            evidence['new_performance'] = new_strategy.performance
            evidence['delta'] = new_strategy.performance - prev_performance
            evidence['evolved'] = True
            evidence['message'] = f"✅ 策略 v{prev_version} → v{new_strategy.version}，性能 {prev_performance:.2f} → {new_strategy.performance:.2f}"
        else:
            evidence['evolved'] = False
            evidence['message'] = f"✅ 当前策略 v{prev_version}，性能 {prev_performance:.2f}（未触发进化）"
        
        evidence['formula'] = 'π_t = f(π_{t-1}, ΔE)'
        evidence['success'] = True
        return evidence
    except Exception as e:
        return {'success': False, 'role': 'dynamic_policy', 'error': str(e)}


# ============================================================
# 第 4 层：元推理（Meta Reasoning）
# ============================================================

def agentic_rl_meta_reason_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    元推理：R_meta = Eval(Logic)
    
    评估自身推理逻辑的正确性，识别推理偏差。
    
    Args:
        args:
            reasoning_chain: 推理链
            check_bias: 是否检查偏差
    """
    reasoning_chain = args.get('reasoning_chain', '')
    check_bias = args.get('check_bias', True)
    
    # 元推理检查清单（基于认知偏差研究）
    meta_checks = [
        {'check': '前提是否成立', 'category': '逻辑基础'},
        {'check': '推理是否包含跳跃', 'category': '逻辑链完整性'},
        {'check': '是否有反例', 'category': '反向验证'},
        {'check': '是否过度概化', 'category': '认知偏差'},
        {'check': '是否锚定偏差', 'category': '认知偏差'},
        {'check': '是否确认偏差', 'category': '认知偏差'},
        {'check': '是否考虑边界条件', 'category': '完备性'},
        {'check': '结论是否可证伪', 'category': '科学性'},
    ]
    
    if not check_bias:
        meta_checks = [c for c in meta_checks if c['category'] != '认知偏差']
    
    return {
        'success': True,
        'role': 'meta_reason',
        'reasoning_chain': reasoning_chain[:300],
        'check_count': len(meta_checks),
        'checks': meta_checks,
        'recommended_skill': 'agent-self-reflection',
        'formula': 'R_meta = Eval(Logic)',
        'message': f"✅ 元推理检查清单已生成（{len(meta_checks)} 项）"
    }


# ============================================================
# 第 5 层：闭环自省（Self Fix）
# ============================================================

def agentic_rl_self_fix_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    闭环自省：S_fix = Error → Policy
    
    将错误转化为策略改进。从 EvoMaster 失败轨迹中提取改进建议。
    
    Args:
        args:
            error_pattern: 错误模式（可选）
    """
    error_pattern = args.get('error_pattern', '')
    
    try:
        from evo_master import get_evo_master
        evo = get_evo_master()
        
        # 从知识缓存找出失败轨迹
        conn = sqlite3.connect(evo.knowledge_cache.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        if error_pattern:
            failed = cur.execute(
                "SELECT id, task, reward FROM trajectories WHERE (success = 0 OR reward < 0.5) AND task LIKE ? ORDER BY reward ASC LIMIT 5",
                (f'%{error_pattern}%',)
            ).fetchall()
        else:
            failed = cur.execute(
                "SELECT id, task, reward FROM trajectories WHERE success = 0 OR reward < 0.5 ORDER BY reward ASC LIMIT 5"
            ).fetchall()
        
        conn.close()
        
        # 生成策略改进建议
        fix_policies = []
        for f in failed:
            fix_policies.append({
                'error_id': f['id'][:12],
                'error_task': f['task'][:50],
                'error_reward': f['reward'],
                'fix_policy': f"针对任务 {f['task'][:30]} 增加前置验证、错误处理、回滚机制"
            })
        
        return {
            'success': True,
            'role': 'self_fix',
            'error_pattern': error_pattern or 'all',
            'failed_count': len(failed),
            'fix_policies': fix_policies,
            'formula': 'S_fix = Error → Policy',
            'message': f"✅ 闭环自省：发现 {len(failed)} 个错误模式，已生成改进策略"
        }
    except Exception as e:
        return {'success': False, 'role': 'self_fix', 'error': str(e)}


# ============================================================
# 主编排器：I_total = M_base × C_think
# ============================================================

def agentic_rl_orchestrator_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agentic RL 五层主编排器。
    
    实现公式：I_total = M_base × C_think
    其中 C_think = G_set + P_decompose + S_review
    
    Args:
        args:
            task: 任务描述
            mode: "full" | "quick"
    """
    task = args.get('task', '')
    if not task:
        return {'success': False, 'error': 'task 不能为空'}
    
    mode = args.get('mode', 'full')
    
    # 五层能力调用序列
    sequence = []
    
    # 1. 自主目标
    sequence.append({
        'layer': 1,
        'name': '自主目标',
        'tool': 'agentic_rl_self_goal',
        'args': {'env_context': task, 'mode': 'balance'},
        'formula': 'G_self ≠ G_env'
    })
    
    # 2. 周期规划（C_think 第一部分: G_set + P_decompose）
    sequence.append({
        'layer': 2,
        'name': '周期规划',
        'tool': 'agentic_rl_long_plan',
        'args': {'total_goal': task, 'horizon': 'medium' if mode == 'full' else 'short'},
        'formula': 'P_n = Split(G_total)'
    })
    
    # 3. 动态策略
    sequence.append({
        'layer': 3,
        'name': '动态策略',
        'tool': 'agentic_rl_dynamic_policy',
        'args': {'delta_e': task, 'trigger_evolve': mode == 'full'},
        'formula': 'π_t = f(π_{t-1}, ΔE)'
    })
    
    # 4. 元推理（C_think 第二部分: S_review）
    sequence.append({
        'layer': 4,
        'name': '元推理',
        'tool': 'agentic_rl_meta_reason',
        'args': {'reasoning_chain': task, 'check_bias': True},
        'formula': 'R_meta = Eval(Logic)'
    })
    
    # 5. 闭环自省
    sequence.append({
        'layer': 5,
        'name': '闭环自省',
        'tool': 'agentic_rl_self_fix',
        'args': {'error_pattern': ''},
        'formula': 'S_fix = Error → Policy'
    })
    
    return {
        'success': True,
        'role': 'agentic_rl_orchestrator',
        'task': task,
        'mode': mode,
        'layer_count': 5,
        'sequence': sequence,
        'main_formula': 'I_total = M_base × C_think',
        'c_think_formula': 'C_think = G_set + P_decompose + S_review',
        'apex_arl_formula': 'APEX_ARL = RL ∪ {MetaG, Reflect, LongPlan}',
        'hierarchy': 'ApexAgent ⊃ AgenticRL ⊃ StandardRL',
        'timestamp': datetime.now().isoformat(),
        'message': f"✅ Agentic RL 五层调用序列已生成（{mode} 模式，5 层）"
    }


# ============================================================
# 注册所有工具
# ============================================================

registry.register(
    name="agentic_rl_self_goal",
    toolset="skills",
    schema={
        "name": "agentic_rl_self_goal",
        "description": "Agentic RL 第1层：自主目标生成（G_self ≠ G_env）。超级进化12。",
        "parameters": {
            "type": "object",
            "properties": {
                "env_context": {"type": "string", "description": "环境上下文"},
                "mode": {
                    "type": "string",
                    "enum": ["explore", "exploit", "balance"],
                    "description": "探索/利用/平衡模式"
                }
            }
        }
    },
    handler=agentic_rl_self_goal_handler
)

registry.register(
    name="agentic_rl_long_plan",
    toolset="skills",
    schema={
        "name": "agentic_rl_long_plan",
        "description": "Agentic RL 第2层：周期规划（P_n = Split(G_total)）。超级进化12。",
        "parameters": {
            "type": "object",
            "properties": {
                "total_goal": {"type": "string", "description": "总目标"},
                "horizon": {
                    "type": "string",
                    "enum": ["short", "medium", "long"],
                    "description": "短期(1天)/中期(1周)/长期(1月)"
                }
            },
            "required": ["total_goal"]
        }
    },
    handler=agentic_rl_long_plan_handler
)

registry.register(
    name="agentic_rl_dynamic_policy",
    toolset="skills",
    schema={
        "name": "agentic_rl_dynamic_policy",
        "description": "Agentic RL 第3层：动态策略（π_t = f(π_{t-1}, ΔE)）。后端：EvoMaster。超级进化12。",
        "parameters": {
            "type": "object",
            "properties": {
                "delta_e": {"type": "string", "description": "环境变化描述"},
                "trigger_evolve": {"type": "boolean", "description": "是否触发策略进化"}
            }
        }
    },
    handler=agentic_rl_dynamic_policy_handler
)

registry.register(
    name="agentic_rl_meta_reason",
    toolset="skills",
    schema={
        "name": "agentic_rl_meta_reason",
        "description": "Agentic RL 第4层：元推理（R_meta = Eval(Logic)）。检查认知偏差。超级进化12。",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning_chain": {"type": "string", "description": "推理链"},
                "check_bias": {"type": "boolean", "description": "是否检查认知偏差"}
            }
        }
    },
    handler=agentic_rl_meta_reason_handler
)

registry.register(
    name="agentic_rl_self_fix",
    toolset="skills",
    schema={
        "name": "agentic_rl_self_fix",
        "description": "Agentic RL 第5层：闭环自省（S_fix = Error → Policy）。从 EvoMaster 失败轨迹生成改进策略。超级进化12。",
        "parameters": {
            "type": "object",
            "properties": {
                "error_pattern": {"type": "string", "description": "错误模式（可选）"}
            }
        }
    },
    handler=agentic_rl_self_fix_handler
)

registry.register(
    name="agentic_rl_orchestrator",
    toolset="skills",
    schema={
        "name": "agentic_rl_orchestrator",
        "description": "Agentic RL 主编排器：五层协同（I_total = M_base × C_think）。超级进化12。",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "任务描述"},
                "mode": {
                    "type": "string",
                    "enum": ["full", "quick"],
                    "description": "full=完整五层, quick=简化"
                }
            },
            "required": ["task"]
        }
    },
    handler=agentic_rl_orchestrator_handler
)
