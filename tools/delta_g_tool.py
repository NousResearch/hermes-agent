"""
SE14 ΔG 演化范式工具 - 超级进化14

CLAW 神技能：APEX ΔG 自洽演化体系

核心公式：
- EV = BV + AV, ΣC_all ≤ SV
- HarmRate = 34% 负势湮灭机制
- EV = f_θ(h(x)) 高维隐态映射决策

底层引擎：hermes_apex_delta_g.so (PyO3 嵌入式)
"""

from tools.registry import registry
import logging
import json
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# 优先尝试加载 PyO3 嵌入式核心
try:
    import hermes_apex_delta_g as _delta_g
    _NATIVE_AVAILABLE = True
    logger.info(f"✅ SE14 ΔG PyO3 native core loaded: {_delta_g.version()}")
except ImportError:
    _delta_g = None
    _NATIVE_AVAILABLE = False
    logger.warning("⚠️ SE14 ΔG PyO3 native core not available")


def delta_g_compute_ev_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    计算效用演化值 EV = BV + AV
    
    Args:
        task: 任务上下文 {task_id, task_type, complexity, priority, resources}
        tools: 工具调用列表 [{tool_name, success, latency_ms, cost}, ...]
        harm_threshold: HarmRate 阈值（默认 0.34）
        sv_limit: 系统资源上确界（默认 100.0）
    """
    if not _NATIVE_AVAILABLE:
        return {'success': False, 'error': 'SE14 PyO3 module not available'}
    
    try:
        task = args.get('task', {})
        tools = args.get('tools', [])
        harm_threshold = args.get('harm_threshold', 0.34)
        sv_limit = args.get('sv_limit', 100.0)
        
        # 默认填充
        if 'task_id' not in task:
            task['task_id'] = f"task_{hash(str(task)) % 100000}"
        if 'task_type' not in task:
            task['task_type'] = 'general'
        if 'complexity' not in task:
            task['complexity'] = 0.5
        if 'priority' not in task:
            task['priority'] = 0.5
        if 'resources' not in task:
            task['resources'] = {}
        
        # 工具默认值
        for t in tools:
            t.setdefault('success', True)
            t.setdefault('latency_ms', 100.0)
            t.setdefault('cost', 1.0)
        
        result_json = _delta_g.py_compute_ev(
            json.dumps(task), json.dumps(tools), harm_threshold, sv_limit
        )
        result = json.loads(result_json)
        result['success'] = True
        return result
        
    except Exception as e:
        logger.error(f"SE14 EV 计算失败: {e}")
        return {'success': False, 'error': str(e)}


def delta_g_zerolang_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行 ZeroLang DSL 代码
    
    支持指令：
    - SET x = value
    - ADD x y
    - PRINT x
    """
    if not _NATIVE_AVAILABLE:
        return {'success': False, 'error': 'SE14 PyO3 module not available'}
    
    try:
        code = args.get('code', '')
        if not code:
            return {'success': False, 'error': 'No code provided'}
        
        output = _delta_g.py_zerolang_execute(code)
        return {
            'success': True,
            'output': output,
            'language': 'ZeroLang',
            'engine': 'hermes_apex_delta_g'
        }
    except Exception as e:
        logger.error(f"ZeroLang 执行失败: {e}")
        return {'success': False, 'error': str(e)}


def delta_g_a2a_send_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    创建 A2A (Agent-to-Agent) 消息
    
    Args:
        from_agent: 发送方
        to_agent: 接收方
        message_type: 消息类型 (request/response/broadcast)
        payload: 消息载荷
    """
    if not _NATIVE_AVAILABLE:
        return {'success': False, 'error': 'SE14 PyO3 module not available'}
    
    try:
        from_agent = args.get('from_agent', 'unknown')
        to_agent = args.get('to_agent', 'broadcast')
        msg_type = args.get('message_type', 'request')
        
        msg_json = _delta_g.py_a2a_create(from_agent, to_agent, msg_type)
        return {
            'success': True,
            'message': json.loads(msg_json),
            'message_json': msg_json
        }
    except Exception as e:
        logger.error(f"A2A 消息创建失败: {e}")
        return {'success': False, 'error': str(e)}


def delta_g_info_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """获取 SE14 ΔG 模块信息"""
    if not _NATIVE_AVAILABLE:
        return {'success': False, 'error': 'SE14 PyO3 module not available'}
    
    return {
        'success': True,
        'version': _delta_g.version(),
        'native_available': _NATIVE_AVAILABLE,
        'capabilities': [
            'EV 效用演化计算（EV = BV + AV）',
            '资源约束检查（ΣC_all ≤ SV）',
            'HarmRate 负势湮灭机制',
            'ZeroLang DSL 解释器',
            'A2A (Agent-to-Agent) 协作'
        ],
        'formulas': {
            'EV': 'EV = BV + AV',
            'constraint': 'ΣC_all ≤ SV',
            'harm_threshold': '34%',
            'high_dim': 'EV = f_θ(h(x))'
        }
    }


# 注册工具
registry.register(
    name="delta_g_compute_ev",
    toolset="skills",
    schema={
        "name": "delta_g_compute_ev",
        "description": "SE14 ΔG: 计算效用演化值 EV = BV + AV，自动判断负势湮灭触发",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "object", "description": "任务上下文"},
                "tools": {"type": "array", "description": "工具调用列表"},
                "harm_threshold": {"type": "number", "description": "HarmRate 阈值（默认 0.34）"},
                "sv_limit": {"type": "number", "description": "系统资源上确界（默认 100.0）"}
            },
            "required": ["task"]
        }
    },
    handler=delta_g_compute_ev_handler
)

registry.register(
    name="delta_g_zerolang",
    toolset="skills",
    schema={
        "name": "delta_g_zerolang",
        "description": "SE14 ΔG: 执行 ZeroLang DSL 代码（SET/ADD/PRINT）",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "ZeroLang 代码"}
            },
            "required": ["code"]
        }
    },
    handler=delta_g_zerolang_handler
)

registry.register(
    name="delta_g_a2a_send",
    toolset="skills",
    schema={
        "name": "delta_g_a2a_send",
        "description": "SE14 ΔG: 创建 A2A (Agent-to-Agent) 协作消息",
        "parameters": {
            "type": "object",
            "properties": {
                "from_agent": {"type": "string", "description": "发送方 Agent"},
                "to_agent": {"type": "string", "description": "接收方 Agent"},
                "message_type": {"type": "string", "description": "消息类型 (request/response/broadcast)"}
            },
            "required": ["from_agent", "to_agent"]
        }
    },
    handler=delta_g_a2a_send_handler
)

registry.register(
    name="delta_g_info",
    toolset="skills",
    schema={
        "name": "delta_g_info",
        "description": "SE14 ΔG: 获取 ΔG 演化范式模块信息",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    handler=delta_g_info_handler
)
