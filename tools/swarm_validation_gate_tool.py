"""StraTA 蜂群主公式自验证门禁 — SE15 闭环检查工具。

在任务声称完成前，代入主公式逐项检查已落地状态。
支持: 主公式验证、偏离度检查、自愈闭环。
"""
from tools.registry import registry
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# 主公式组件清单
STATA_COMPONENTS = {
    "pi_z_given_s1": {
        "symbol": "π(z|s₁)",
        "name": "GPT全局策略生成",
        "description": "任务前是否生成了全局轨迹/计划/策略",
        "check_questions": ["是否调用了 full_toolcall_plan / qr route / delegate_task？", "策略 z 是否覆盖了任务所有子目标？"],
    },
    "pi_at_given_zs": {
        "symbol": "π(aₜ|z,sₜ)",
        "name": "多Agent并行执行",
        "description": "是否按策略 z 派发了并行子智能体",
        "check_questions": ["是否使用了 delegate_task batch 模式？", "子智能体是否按策略 z 分工？"],
    },
    "grpo": {
        "symbol": "GRPO(z,aₜ)",
        "name": "分层优化",
        "description": "是否对执行结果进行了评分/优化",
        "check_questions": ["是否运行了 eval center / evolver 评分？", "是否识别了需要改进的差异？"],
    },
    "memllm": {
        "symbol": "MemLLM",
        "name": "长期记忆同步",
        "description": "是否将关键经验写入了记忆/基因库",
        "check_questions": ["是否调用了 memory_tool 写 MEMORY/USER？", "是否写入了进化基因？"],
    },
    "farthest_point_sampling": {
        "symbol": "FPS",
        "name": "最远点采样(策略多样性)",
        "description": "执行策略是否覆盖了多种可能路径",
        "check_questions": ["是否使用了不同的子智能体角色/视角？", "是否考虑了备选方案？"],
    },
    "self_validation": {
        "symbol": "自验证",
        "name": "主公式验证闭环",
        "description": "完成前代入公式检查",
        "check_questions": ["是否逐项检查了 π·π·GRPO·MemLLM？", "未达标项是否标记并进入自愈？"],
    },
}


def swarm_verify(args: Dict[str, Any]) -> Dict[str, Any]:
    """StraTA 主公式验证器—在声称完成前逐项检查。

    Args:
        mode: 'quick' (默认) | 'full' | 'gap_only'
        checks: 手动传入的各组件检查结果 dict (optional)
    """
    mode = args.get('mode', 'quick')
    manual_checks = args.get('checks', None)

    if manual_checks:
        # 使用手动传入的检查结果
        results = {}
        passed = []
        failed = []
        for comp_id, comp_data in STATA_COMPONENTS.items():
            status = manual_checks.get(comp_id, 'untested')
            results[comp_id] = {
                'symbol': comp_data['symbol'],
                'name': comp_data['name'],
                'status': status,
            }
            if status in ('pass', 'done', '已完成', 'true', True):
                passed.append(comp_id)
            elif status in ('fail', 'missing', '缺失', 'false', False):
                failed.append(comp_id)
    else:
        # 自动模式：基于当前任务上下文判断
        results = {}
        passed = []
        failed = []
        for comp_id, comp_data in STATA_COMPONENTS.items():
            # 默认标记为 untested，需要人工确认
            results[comp_id] = {
                'symbol': comp_data['symbol'],
                'name': comp_data['name'],
                'status': 'untested',
                'check_questions': comp_data['check_questions'],
            }
            if mode == 'quick':
                failed.append(comp_id)  # 快速模式下所有 untested 视为未通过

    passed_count = len(passed)
    failed_count = len(failed)
    untested_count = len(results) - passed_count - failed_count
    completion = passed_count / len(results) * 100 if len(results) > 0 else 0

    return {
        'success': True,
        'formula': 'ApexStraTA = π(z|s₁) ⊗ π(aₜ|z, sₜ) ⊗ GRPO(z, aₜ) ⊗ MemLLM',
        'mode': mode,
        'components': len(results),
        'components_detail': list(results.values()),
        'summary': {
            'passed': passed_count,
            'failed': failed_count,
            'untested': untested_count,
            'completion_pct': round(completion, 1),
            'needs_self_heal': failed_count > 0 or untested_count > 0,
        },
        'verdict': 'complete' if completion >= 100 else ('partial' if completion >= 60 else 'incomplete'),
        'next_action': '无需自愈' if completion >= 100 else f"建议启动自愈循环修正 {failed_count + untested_count} 个未达标组件",
    }


def swarm_deviation_check(args: Dict[str, Any]) -> Dict[str, Any]:
    """偏离度检查—比较 agent 实际动作与原始策略 z 的匹配度。

    Args:
        expected_steps: 原始策略期望步骤列表
        actual_steps: agent 实际执行的步骤列表
    """
    expected = args.get('expected_steps', []) or []
    actual = args.get('actual_steps', []) or []

    if not expected:
        return {'success': False, 'error': '需要 expected_steps 参数'}

    # 计算覆盖度
    expected_set = set(s.lower().strip() for s in expected if isinstance(s, str))
    actual_set = set(s.lower().strip() for s in actual if isinstance(s, str))

    covered = expected_set & actual_set
    missing = expected_set - actual_set
    extra = actual_set - expected_set

    coverage_pct = len(covered) / len(expected_set) * 100 if expected_set else 0
    deviation_pct = len(missing) / len(expected_set) * 100 if expected_set else 0

    return {
        'success': True,
        'coverage_pct': round(coverage_pct, 1),
        'deviation_pct': round(deviation_pct, 1),
        'covered_steps': len(covered),
        'missing_steps': len(missing),
        'extra_steps': len(extra),
        'covered': sorted(covered),
        'missing': sorted(missing),
        'extra': sorted(extra),
        'needs_correction': deviation_pct > 30,
        'verdict': 'on_track' if coverage_pct >= 70 else 'deviated',
    }


def swarm_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """蜂群主公式控制面板。

    Args:
        action: 'verify' (主公式验证) | 'deviation' (偏离度检查) | 'heal' (自愈)
        mode: verify 用: quick/full/gap_only
        checks: verify 用: 手动检查结果
        expected_steps: deviation 用: 期望步骤列表
        actual_steps: deviation 用: 实际步骤列表
    """
    action = args.get('action', 'verify')

    if action == 'verify':
        return swarm_verify(args)
    elif action == 'deviation':
        return swarm_deviation_check(args)
    elif action == 'heal':
        # 自愈 = verify 发现未达标 → 返回自愈建议
        verify_result = swarm_verify(args)
        if verify_result.get('summary', {}).get('needs_self_heal'):
            failed_components = [
                c for c in (verify_result.get('components_detail') or [])
                if c.get('status') in ('fail', 'missing', 'untested')
            ]
            return {
                'success': True,
                'action': 'heal',
                'failed_components': [c['name'] for c in failed_components],
                'heal_suggestion': f"对 {len(failed_components)} 个未达标组件分别启动 subagent 自愈轮次",
                'max_heal_rounds': 3,
                'healed': False,
            }
        else:
            return {'success': True, 'action': 'heal', 'message': '无需自愈，所有组件已达标'}

    return {'success': False, 'error': f'未知 action: {action}'}


registry.register(
    name="swarm_validation",
    toolset="skills",
    schema={
        "name": "swarm_validation",
        "description": "蜂群主公式自验证门禁——主公式验证/偏离度检查/自愈",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["verify", "deviation", "heal"],
                    "description": "verify=主公式验证, deviation=偏离度检查, heal=自愈"
                },
                "mode": {
                    "type": "string",
                    "enum": ["quick", "full", "gap_only"],
                    "description": "verify 模式: quick≈快速, full=完整, gap_only=仅检查未达标项"
                },
                "checks": {
                    "type": "object",
                    "description": "手动传入的各组件检查结果"
                },
                "expected_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "偏离度检查用: 原始策略期望步骤列表"
                },
                "actual_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "偏离度检查用: 实际执行的步骤列表"
                },
            }
        }
    },
    handler=swarm_handler,
)
