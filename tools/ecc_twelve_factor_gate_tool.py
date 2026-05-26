"""SE17 ECC 十二因子治理门禁。

Apex_agent = ΔG ⊙ Π F_i
把 RuntimeOS/Agent Harness 愿景收敛为可评分、可审计、可回滚的十二因子门禁。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from tools.registry import registry

HERMES = Path('/Users/appleoppa/.hermes')
AGENT = HERMES / 'hermes-agent'

FACTORS = [
    ('runtime', 'Runtime', [AGENT / 'run_agent.py', AGENT / 'agent' / 'conversation_loop.py']),
    ('plugin_architecture', 'Plugin Architecture', [AGENT / 'plugins', AGENT / 'tools' / 'registry.py']),
    ('skills', 'Skills', [HERMES / 'skills', HERMES / 'skills' / 'workflow' / 'apex-hermes-evolution-engine' / 'SKILL.md']),
    ('memory', 'Memory', [HERMES / 'memories' / 'MEMORY.md', HERMES / 'memories' / 'USER.md']),
    ('hooks', 'Hooks', [AGENT / 'plugins', AGENT / 'gateway' / 'builtin_hooks']),
    ('rules', 'Rules', [HERMES / 'SOUL.md', HERMES / 'hermes-agent' / 'AGENTS.md']),
    ('multi_agent', 'Multi-agent', [AGENT / 'tools' / 'delegate_tool.py']),
    ('session_state', 'Session State', [AGENT / 'hermes_state.py']),
    ('security', 'Security', [AGENT / 'tools' / 'security.py', HERMES / 'config.yaml']),
    ('observability', 'Observability', [HERMES / 'logs', AGENT / 'hermes_logging.py']),
    ('governance', 'Governance', [HERMES / 'workspace' / '治理' / 'ACTIVE_MANIFEST.md', HERMES / 'workspace' / 'standards']),
    ('learning', 'Learning', [HERMES / 'workspace' / '开智' / '02-进化基因' / 'apex_evolution_genes.sqlite3', AGENT / 'evo_master.py']),
]


def factor_status() -> List[Dict[str, Any]]:
    rows = []
    for key, name, paths in FACTORS:
        checks = []
        for p in paths:
            checks.append({'path': str(p), 'exists': p.exists(), 'size': p.stat().st_size if p.exists() and p.is_file() else 0})
        exists_count = sum(1 for x in checks if x['exists'])
        score = exists_count / len(checks) if checks else 0.0
        rows.append({
            'factor': key,
            'name': name,
            'score': round(score, 3),
            'status': 'pass' if score >= 0.5 else 'fail',
            'evidence': checks,
        })
    return rows


def ecc_score(delta_g: float = 1.0) -> Dict[str, Any]:
    rows = factor_status()
    product = 1.0
    for r in rows:
        product *= max(r['score'], 0.01)  # 避免全域归零但保留强惩罚
    apex_agent = delta_g * product
    return {
        'success': True,
        'formula': 'Apex_agent = ΔG ⊙ Π F_i',
        'delta_g': delta_g,
        'factor_product': round(product, 6),
        'apex_agent_score': round(apex_agent, 6),
        'passed': sum(1 for r in rows if r['status'] == 'pass'),
        'total': len(rows),
        'factors': rows,
        'verdict': 'go' if apex_agent >= 0.2 and sum(1 for r in rows if r['status']=='pass') >= 10 else 'hold',
        'boundary': '治理评分门禁，不自动重构核心，不自动GitHub同步，不夜间无人值守改core',
    }


def ecc_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    action = args.get('action', 'score')
    if action == 'status':
        return {'success': True, 'factors': factor_status()}
    if action == 'score':
        return ecc_score(float(args.get('delta_g', 1.0)))
    return {'success': False, 'error': f'unknown action: {action}'}


registry.register(
    name='ecc_twelve_factor_gate',
    toolset='skills',
    schema={
        'name': 'ecc_twelve_factor_gate',
        'description': 'SE17 ECC十二因子治理门禁：Runtime/Plugin/Skills/Memory/Hooks/Rules/Multi-agent/Session/Security/Observability/Governance/Learning评分',
        'parameters': {
            'type': 'object',
            'properties': {
                'action': {'type': 'string', 'enum': ['status', 'score']},
                'delta_g': {'type': 'number'}
            }
        }
    },
    handler=ecc_handler,
)
