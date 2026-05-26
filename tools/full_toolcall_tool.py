"""
Full Tool-Call Tool - Hermes 工具集成

将 Full Tool-Call 全局轨迹规划和执行能力集成为 Hermes 工具。
"""

from tools.registry import registry
from full_toolcall_integration import get_full_toolcall
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


def full_toolcall_audit_handler(plan_path: str = '', run_path: str = '') -> dict:
    """审计 Full Tool-Call sidecar 产物，明确其不是 Hermes 主链路核心改造。"""
    try:
        if not run_path:
            latest = Path('/Users/appleoppa/.hermes/workspace/开智/full_toolcall_sidecar/run_latest.json')
            run_path = str(latest) if latest.exists() else ''
        if not plan_path:
            latest = Path('/Users/appleoppa/.hermes/workspace/开智/full_toolcall_sidecar/plan_latest.json')
            plan_path = str(latest) if latest.exists() else ''
        evidence = {}
        if plan_path and Path(plan_path).exists():
            plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
            evidence['plan'] = {
                'path': plan_path,
                'schema': plan.get('schema'),
                'goal': plan.get('goal'),
                'boundary': plan.get('boundary'),
                'steps': len(plan.get('global_trajectory', [])) if isinstance(plan.get('global_trajectory'), list) else None,
            }
        if run_path and Path(run_path).exists():
            run = json.loads(Path(run_path).read_text(encoding='utf-8'))
            summary = run.get('summary', {})
            evidence['run'] = {
                'path': run_path,
                'schema': run.get('schema'),
                'boundary': run.get('boundary'),
                'summary': summary,
                'tasks': len(run.get('tasks', [])) if isinstance(run.get('tasks'), list) else None,
            }
        if not evidence:
            return {'success': False, 'error': '未找到可审计的 plan/run 产物'}
        return {
            'success': True,
            'status': 'sidecar_verified_not_core_mainloop',
            'evidence': evidence,
            'boundary': 'Full Tool-Call 当前为 sidecar/tool 层能力，未修改 run_agent.py 主循环；不能宣称主链路原生完成',
            'promotion_gate': ['设计评审', '备份 run_agent.py', '最小补丁', '全量工具调用回归测试', '可回滚开关'],
        }
    except Exception as e:
        logger.error(f"full_toolcall_audit 执行失败: {e}")
        return {'success': False, 'error': str(e), 'message': f"审计异常: {e}"}


def full_toolcall_plan_handler(input_text: str, goal: str) -> dict:
    """生成全局轨迹规划"""
    try:
        ftc = get_full_toolcall()
        result = ftc.plan(input_text, goal)
        
        if not result['success']:
            return {
                'success': False,
                'error': result.get('error', 'unknown'),
                'message': '规划失败'
            }
        
        plan = result['plan']
        summary = result['summary']
        
        return {
            'success': True,
            'plan_path': result['plan_path'],
            'summary': {
                'steps': summary['steps'],
                'tools': summary['tools'],
                'risks': summary['risks'],
                'parallel_groups': summary['parallel_groups']
            },
            'trajectory': plan['global_trajectory'],
            'parallel_points': plan['parallel_points'],
            'risks': plan['risks'],
            'message': f"✅ 规划完成：{summary['steps']} 步骤，{summary['risks']} 风险"
        }
        
    except Exception as e:
        logger.error(f"full_toolcall_plan 执行失败: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f"规划异常: {e}"
        }


def full_toolcall_execute_handler(plan_path: str, simulate_failure: bool = False) -> dict:
    """执行全局轨迹"""
    try:
        ftc = get_full_toolcall()
        result = ftc.execute(plan_path, simulate_failure=simulate_failure)
        
        if not result.success:
            return {
                'success': False,
                'error': result.error,
                'message': '执行失败'
            }
        
        return {
            'success': True,
            'run_path': result.run_path,
            'summary': {
                'tasks_total': result.tasks_total,
                'completed': result.completed,
                'failed': result.failed,
                'success_rate': result.completed / result.tasks_total if result.tasks_total > 0 else 0,
                'events': result.events,
                'workers_used': result.workers_used,
                'healing_events': result.healing_events
            },
            'message': f"✅ 执行完成：{result.completed}/{result.tasks_total} 任务，{result.healing_events} 次自愈"
        }
        
    except Exception as e:
        logger.error(f"full_toolcall_execute 执行失败: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f"执行异常: {e}"
        }


def full_toolcall_handler(input_text: str, goal: str, simulate_failure: bool = False) -> dict:
    """一站式全局轨迹规划和执行"""
    try:
        ftc = get_full_toolcall()
        result = ftc.plan_and_execute(input_text, goal, simulate_failure=simulate_failure)
        
        if not result['success']:
            return {
                'success': False,
                'stage': result['stage'],
                'error': result.get('error', 'unknown'),
                'message': f"失败于 {result['stage']} 阶段"
            }
        
        summary = result['summary']
        
        return {
            'success': True,
            'stage': result['stage'],
            'plan_path': result['plan']['plan_path'],
            'run_path': result['execution'].run_path,
            'summary': {
                'steps': summary['steps'],
                'completed': summary['completed'],
                'failed': summary['failed'],
                'success_rate': summary['success_rate'],
                'healing_events': summary['healing_events']
            },
            'message': f"✅ 完成：{summary['completed']}/{summary['steps']} 任务，成功率 {summary['success_rate']:.1%}"
        }
        
    except Exception as e:
        logger.error(f"full_toolcall 执行失败: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f"执行异常: {e}"
        }


# 注册工具
registry.register(
    name="full_toolcall_audit",
    toolset="workflow",
    schema={
        "name": "full_toolcall_audit",
        "description": "审计 Full Tool-Call sidecar 产物，明确完成边界和主链路晋升门禁。",
        "parameters": {
            "type": "object",
            "properties": {
                "plan_path": {"type": "string", "description": "可选：规划文件路径"},
                "run_path": {"type": "string", "description": "可选：运行文件路径"}
            }
        }
    },
    handler=full_toolcall_audit_handler
)

registry.register(
    name="full_toolcall_plan",
    toolset="workflow",
    schema={
        "name": "full_toolcall_plan",
        "description": "生成全局轨迹规划（Full Tool-Call）。输入任务描述和目标，输出包含步骤、工具、风险、并行点的全局轨迹。",
        "parameters": {
            "type": "object",
            "properties": {
                "input_text": {
                    "type": "string",
                    "description": "输入材料（任务描述、需求文档等）或文件路径"
                },
                "goal": {
                    "type": "string",
                    "description": "目标描述（简洁明确）"
                }
            },
            "required": ["input_text", "goal"]
        }
    },
    handler=full_toolcall_plan_handler
)

registry.register(
    name="full_toolcall_execute",
    toolset="workflow",
    schema={
        "name": "full_toolcall_execute",
        "description": "执行全局轨迹（Full Tool-Call）。基于规划文件执行任务，支持并行调度、负载均衡和故障自愈。",
        "parameters": {
            "type": "object",
            "properties": {
                "plan_path": {
                    "type": "string",
                    "description": "规划文件路径（由 full_toolcall_plan 生成）"
                },
                "simulate_failure": {
                    "type": "boolean",
                    "description": "是否模拟故障（用于测试，默认 false）"
                }
            },
            "required": ["plan_path"]
        }
    },
    handler=full_toolcall_execute_handler
)

registry.register(
    name="full_toolcall",
    toolset="workflow",
    schema={
        "name": "full_toolcall",
        "description": "一站式全局轨迹规划和执行（Full Tool-Call）。输入任务描述和目标，自动完成规划和执行，返回完整结果。",
        "parameters": {
            "type": "object",
            "properties": {
                "input_text": {
                    "type": "string",
                    "description": "输入材料（任务描述、需求文档等）或文件路径"
                },
                "goal": {
                    "type": "string",
                    "description": "目标描述（简洁明确）"
                },
                "simulate_failure": {
                    "type": "boolean",
                    "description": "是否模拟故障（用于测试，默认 false）"
                }
            },
            "required": ["input_text", "goal"]
        }
    },
    handler=full_toolcall_handler
)
