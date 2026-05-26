"""
TianGong Orchestrator - 超级进化11

四核协同主编排器：evolver + autoresearch + openhands + superpowers。

实现 GPT 主导的状态机与门禁流程。
"""

from tools.registry import registry
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


def tiangong_orchestrator_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    TianGong Orchestrator：四核协同编排。
    
    Args:
        args:
            task: 任务描述（必需）
            mode: "plan" | "execute" | "status"
            cores: 启用的核心列表（默认全部）
            
    Returns:
        编排报告
    """
    task = args.get('task', '')
    if not task:
        return {'success': False, 'error': 'task 参数不能为空', 'role': 'orchestrator'}
    
    mode = args.get('mode', 'plan')
    cores = args.get('cores', ['evolver', 'autoresearch', 'openhands', 'superpowers'])
    
    evidence = {
        'role': 'orchestrator',
        'task': task,
        'mode': mode,
        'cores_enabled': cores,
        'timestamp': datetime.now().isoformat(),
    }
    
    if mode == 'plan':
        # 生成四核协同计划
        plan = {
            'state_machine': [
                'received', 'scoped', 'gated', 'routed', 'planned',
                'assigned', 'executing', 'verifying', 'audited', 'completed'
            ],
            'gates': {
                'G0_entry': '目标、范围、副作用、工具需求、验收标准',
                'G1_plan': '主路径、备选路径、验证方法、风险点',
                'G2_exec': '依赖齐全、输入存在、操作安全、可回滚',
                'G3_verify': '测试/读回/检索/构建/状态检查',
                'G4_deliver': '需求覆盖、证据闭环、文件变更透明、风险披露',
            },
            'four_core_assignment': {},
        }
        
        # 根据任务类型分配四核职责
        task_lower = task.lower()
        
        if 'evolver' in cores:
            if any(k in task_lower for k in ['bug', 'defect', 'fail', 'error', '缺陷', '失败', '错误']):
                plan['four_core_assignment']['evolver'] = '缺陷扫描与失败归因'
            else:
                plan['four_core_assignment']['evolver'] = '轨迹回收与基因沉淀'
        
        if 'autoresearch' in cores:
            if any(k in task_lower for k in ['research', 'paper', 'arxiv', 'study', '研究', '论文']):
                plan['four_core_assignment']['autoresearch'] = '多源检索与知识蒸馏'
            else:
                plan['four_core_assignment']['autoresearch'] = '背景知识补足'
        
        if 'openhands' in cores:
            if any(k in task_lower for k in ['file', 'terminal', 'command', '文件', '命令']):
                plan['four_core_assignment']['openhands'] = '文件/终端执行'
            else:
                plan['four_core_assignment']['openhands'] = '工具执行规划'
        
        if 'superpowers' in cores:
            if any(k in task_lower for k in ['design', 'plan', 'review', '设计', '规划', '审查']):
                plan['four_core_assignment']['superpowers'] = '设计/拆解/审查'
            else:
                plan['four_core_assignment']['superpowers'] = '质量门禁与交付'
        
        evidence['plan'] = plan
        evidence['next_state'] = 'scoped'
        evidence['message'] = f"✅ 四核协同计划已生成（{len(cores)} 个核心）"
    
    elif mode == 'execute':
        # 执行模式：返回四核调用序列
        execution_sequence = []
        
        # 1. Superpowers 澄清
        if 'superpowers' in cores:
            execution_sequence.append({
                'step': 1,
                'core': 'superpowers',
                'action': 'clarify',
                'tool': 'tiangong_superpowers',
                'args': {'action': 'clarify', 'context': task},
            })
        
        # 2. AutoResearch 背景调研
        if 'autoresearch' in cores:
            execution_sequence.append({
                'step': 2,
                'core': 'autoresearch',
                'action': 'research',
                'tool': 'tiangong_autoresearch',
                'args': {'query': task, 'sources': ['arxiv'], 'max_results': 3},
            })
        
        # 3. Superpowers 设计
        if 'superpowers' in cores:
            execution_sequence.append({
                'step': 3,
                'core': 'superpowers',
                'action': 'design',
                'tool': 'tiangong_superpowers',
                'args': {'action': 'design', 'context': task},
            })
        
        # 4. OpenHands 执行规划
        if 'openhands' in cores:
            execution_sequence.append({
                'step': 4,
                'core': 'openhands',
                'action': 'plan',
                'tool': 'tiangong_openhands',
                'args': {'action_type': 'plan', 'task': task},
            })
        
        # 5. Superpowers 验证
        if 'superpowers' in cores:
            execution_sequence.append({
                'step': 5,
                'core': 'superpowers',
                'action': 'verify',
                'tool': 'tiangong_superpowers',
                'args': {'action': 'verify', 'context': task},
            })
        
        # 6. Evolver 沉淀
        if 'evolver' in cores:
            execution_sequence.append({
                'step': 6,
                'core': 'evolver',
                'action': 'evolve',
                'tool': 'tiangong_evolver',
                'args': {'mode': 'evolve'},
            })
        
        evidence['execution_sequence'] = execution_sequence
        evidence['total_steps'] = len(execution_sequence)
        evidence['message'] = f"✅ 执行序列已生成（{len(execution_sequence)} 步）"
    
    else:  # status
        evidence['status'] = {
            'cores_available': 4,
            'cores_enabled': len(cores),
            'state_machine_stages': 10,
            'gates': 5,
            'backend': 'Hermes 原生 + EvoMaster + arxiv',
        }
        evidence['message'] = "✅ TianGong Orchestrator 就绪"
    
    return {'success': True, **evidence}


registry.register(
    name="tiangong_orchestrator",
    toolset="skills",
    schema={
        "name": "tiangong_orchestrator",
        "description": "TianGong 主编排器：四核协同（evolver/autoresearch/openhands/superpowers）+ 状态机 + 门禁（超级进化11）。",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "任务描述"},
                "mode": {
                    "type": "string",
                    "enum": ["plan", "execute", "status"],
                    "description": "plan=生成计划, execute=执行序列, status=状态"
                },
                "cores": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["evolver", "autoresearch", "openhands", "superpowers"]},
                    "description": "启用的核心（默认全部）"
                }
            },
            "required": ["task"]
        }
    },
    handler=tiangong_orchestrator_handler
)
