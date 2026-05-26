"""
TianGong OpenHands 适配器 - 超级进化11

四核之一：文件、终端、浏览器、沙箱执行。

实际后端：Hermes 原生工具集（terminal/file/browser/computer_use）。
不实际接入 OpenHands docker 沙箱（重量级、与 Hermes 架构重叠）。
"""

from tools.registry import registry
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def tiangong_openhands_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    TianGong OpenHands：执行规划、轨迹记录与真实工具调用。
    
    支持三种模式：
    - plan: 基于关键词推断需要的工具（不执行）
    - map_tools: 返回 OpenHands→Hermes 工具映射
    - stats: 统计可用工具集
    - execute: 真实执行（NEW - 调用 Hermes 原生工具）
    
    Args:
        args:
            action_type: "plan" | "map_tools" | "stats" | "execute"
            task: 任务描述
            tool: execute 模式下，要调用的 Hermes 工具名
            tool_args: execute 模式下，传给工具的参数 dict
            
    Returns:
        执行计划、工具映射或真实执行结果
    """
    action_type = args.get('action_type', 'map_tools')
    task = args.get('task', '')
    
    evidence = {
        'role': 'openhands',
        'backend': 'Hermes 原生工具集',
        'action_type': action_type,
    }
    
    # NEW: 真实执行模式（调用 Hermes 原生工具）
    if action_type == 'execute':
        tool_name = args.get('tool', '')
        tool_args = args.get('tool_args', {})
        
        if not tool_name:
            return {'success': False, 'error': 'execute 模式需要 tool 参数', 'role': 'openhands'}
        
        # 通过 registry 调用 Hermes 原生工具
        try:
            tool_entry = registry._tools.get(tool_name)
            if not tool_entry:
                return {
                    'success': False,
                    'error': f'工具 {tool_name} 未注册',
                    'available_tools': list(registry._tools.keys())[:20],
                    'role': 'openhands'
                }
            
            # 调用工具 handler
            handler = tool_entry.handler if hasattr(tool_entry, 'handler') else tool_entry['handler']
            result = handler(tool_args)
            
            evidence['tool_name'] = tool_name
            evidence['tool_args'] = tool_args
            evidence['tool_result'] = result
            evidence['message'] = f"✅ 执行 {tool_name} 完成"
            return {'success': True, **evidence}
            
        except Exception as e:
            logger.error(f"openhands execute 失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'tool_name': tool_name,
                'role': 'openhands'
            }
    
    if action_type == 'plan':
        if not task:
            return {'success': False, 'error': 'plan 模式需要 task 参数', 'role': 'openhands'}
        
        # 基于关键词推断需要的工具
        task_lower = task.lower()
        recommended = []
        
        if any(k in task_lower for k in ['read', 'cat', 'view', 'show', '读取', '查看']):
            recommended.append({'tool': 'read_file', 'reason': '读取文件内容'})
        if any(k in task_lower for k in ['write', 'create', 'save', '写入', '创建']):
            recommended.append({'tool': 'write_file', 'reason': '写入文件'})
        if any(k in task_lower for k in ['edit', 'modify', 'change', 'patch', '修改', '编辑']):
            recommended.append({'tool': 'patch', 'reason': '编辑文件'})
        if any(k in task_lower for k in ['search', 'find', 'grep', '搜索', '查找']):
            recommended.append({'tool': 'search_files', 'reason': '搜索内容/文件'})
        if any(k in task_lower for k in ['run', 'exec', 'shell', 'command', '执行', '运行']):
            recommended.append({'tool': 'terminal', 'reason': '执行 shell 命令'})
        if any(k in task_lower for k in ['browse', 'click', 'navigate', '浏览', '点击']):
            recommended.append({'tool': 'browser', 'reason': '浏览器自动化'})
        if any(k in task_lower for k in ['screenshot', 'gui', 'ui', '截图', '界面']):
            recommended.append({'tool': 'computer_use', 'reason': 'GUI 自动化'})
        
        if not recommended:
            recommended = [{'tool': 'terminal', 'reason': '默认通用工具'}]
        
        evidence['task'] = task
        evidence['recommended_tools'] = recommended
        evidence['execution_path'] = ' → '.join(r['tool'] for r in recommended)
        evidence['message'] = f"✅ 执行计划：{len(recommended)} 个工具，路径 {evidence['execution_path']}"
    
    elif action_type == 'map_tools':
        # 返回 OpenHands 概念到 Hermes 工具的完整映射
        evidence['mapping'] = {
            'OpenHands.bash': 'terminal',
            'OpenHands.editor': 'patch / write_file',
            'OpenHands.file_read': 'read_file',
            'OpenHands.file_search': 'search_files',
            'OpenHands.browser': 'browser (browser-use toolset)',
            'OpenHands.gui': 'computer_use',
            'OpenHands.python': 'execute_code',
            'OpenHands.delegate': 'delegate_task',
        }
        evidence['note'] = 'OpenHands docker 沙箱不接入；Hermes 原生工具已覆盖等价能力'
        evidence['message'] = "✅ 工具映射就绪（8 个映射）"
    
    else:  # stats
        evidence['available_toolsets'] = [
            'terminal', 'file', 'search', 'web', 'browser', 'computer_use',
            'execute_code', 'delegation', 'cron'
        ]
        evidence['message'] = "✅ Hermes 原生工具集已就绪（9 个 toolset）"
    
    return {'success': True, **evidence}


registry.register(
    name="tiangong_openhands",
    toolset="skills",
    schema={
        "name": "tiangong_openhands",
        "description": "TianGong 四核之 OpenHands：执行规划、工具映射与真实执行（超级进化11）。后端：Hermes 原生工具集。",
        "parameters": {
            "type": "object",
            "properties": {
                "action_type": {
                    "type": "string",
                    "enum": ["plan", "map_tools", "stats", "execute"],
                    "description": "plan=生成执行计划, map_tools=工具映射, stats=统计, execute=真实执行"
                },
                "task": {"type": "string", "description": "任务描述"},
                "tool": {"type": "string", "description": "execute 模式：要调用的 Hermes 工具名"},
                "tool_args": {"type": "object", "description": "execute 模式：传给工具的参数"}
            }
        }
    },
    handler=tiangong_openhands_handler
)
