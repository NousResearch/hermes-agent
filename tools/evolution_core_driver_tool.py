"""SE16.5 Evolution Core Driver — 进化核心驱动状态工具。

将 evomap/evolver/autoresearch/superpowers/openhands/fusion/cli/mcp
收敛为可审计状态面板，不虚构未安装能力。
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from tools.registry import registry

HERMES = Path('/Users/appleoppa/.hermes')
AGENT = HERMES / 'hermes-agent'
TOOLS = AGENT / 'tools'
WORKSPACE = HERMES / 'workspace'

COMPONENTS = {
    'evolver': {
        'desc': '进化驱动核心：缺陷识别、策略演化、基因沉淀',
        'files': [AGENT / 'evo_master.py', TOOLS / 'evo_master_tool.py', TOOLS / 'evolver_auto_learn_tool.py'],
        'commands': [],
    },
    'autoresearch': {
        'desc': '外部知识/开源项目只读研究',
        'files': [HERMES / 'scripts' / 'hermes_open_source_devour_scout.py'],
        'commands': [],
    },
    'superpowers': {
        'desc': '任务拆解/TDD/审查能力',
        'files': [TOOLS / 'tiangong_superpowers_tool.py'],
        'commands': [],
    },
    'openhands': {
        'desc': '工具执行/文件终端调用桥',
        'files': [TOOLS / 'tiangong_openhands_tool.py'],
        'commands': [],
    },
    'fusion_engine': {
        'desc': '多技能/多公式融合治理层',
        'files': [HERMES / 'skills' / 'workflow' / 'apex-hermes-evolution-engine' / 'SKILL.md'],
        'commands': [],
    },
    'cli': {
        'desc': '命令行入口',
        'files': [],
        'commands': ['hermes', 'qr'],
    },
    'mcp': {
        'desc': 'MCP/外部工具接口（如可用）',
        'files': [HERMES / 'mcp.json', HERMES / 'mcp_servers.json'],
        'commands': [],
        'optional': True,
    },
}


def _file_status(path: Path) -> Dict[str, Any]:
    return {
        'path': str(path),
        'exists': path.exists(),
        'size': path.stat().st_size if path.exists() else 0,
    }


def _command_status(cmd: str) -> Dict[str, Any]:
    found = shutil.which(cmd)
    return {'command': cmd, 'exists': bool(found), 'path': found or ''}


def status() -> Dict[str, Any]:
    rows = []
    for name, meta in COMPONENTS.items():
        file_checks = [_file_status(p) for p in meta.get('files', [])]
        cmd_checks = [_command_status(c) for c in meta.get('commands', [])]
        required = file_checks + cmd_checks
        if meta.get('optional') and not required:
            ok = True
        elif not required:
            ok = False
        else:
            ok = all(x.get('exists') for x in required)
        rows.append({
            'component': name,
            'description': meta['desc'],
            'status': 'ok' if ok else ('optional_missing' if meta.get('optional') else 'missing'),
            'files': file_checks,
            'commands': cmd_checks,
            'boundary': 'status_only_no_install_no_core_rewrite',
        })
    return {
        'success': True,
        'schema': 'evolution_core_driver_status_v1',
        'components': rows,
        'ok_count': sum(1 for r in rows if r['status'] == 'ok'),
        'total': len(rows),
        'boundary': '不自动安装外部项目，不自动重构核心；只做状态核验和安全接入建议',
    }


def recommend() -> Dict[str, Any]:
    s = status()
    actions = []
    for r in s['components']:
        if r['status'] == 'missing':
            actions.append({'component': r['component'], 'action': '保守挂起：先确认依赖/安装路径，再做dry-run接入'})
        elif r['status'] == 'optional_missing':
            actions.append({'component': r['component'], 'action': '可选项缺失，不影响主链路；需要时单独启用'})
    if not actions:
        actions.append({'component': 'all', 'action': '所有核心组件存在；可进入ECC十二因子门禁评分'})
    return {'success': True, 'status': s, 'recommendations': actions}


def evolution_core_driver_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    action = args.get('action', 'status')
    if action == 'status':
        return status()
    if action == 'recommend':
        return recommend()
    return {'success': False, 'error': f'unknown action: {action}'}


registry.register(
    name='evolution_core_driver',
    toolset='skills',
    schema={
        'name': 'evolution_core_driver',
        'description': 'SE16.5 进化核心驱动状态工具：检查evolver/autoresearch/superpowers/openhands/fusion/cli/mcp状态',
        'parameters': {
            'type': 'object',
            'properties': {
                'action': {'type': 'string', 'enum': ['status', 'recommend']}
            }
        }
    },
    handler=evolution_core_driver_handler,
)
