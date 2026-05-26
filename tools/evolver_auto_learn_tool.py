"""
SE11-C: evolver GitHub 自动学习代码补全
==============================================

从 ~/.hermes/workspace/github-evolution/research/*.json 中读取能力清单，
自动生成 Hermes 工具骨架并注册。

工作流：
1. 扫描 research/*.json
2. 提取 dominant_traits（高频能力模式）
3. 为每个未实现的能力模式生成工具骨架
4. 写入 tools/evolver_<trait>_tool.py
5. 调用 registry.register()
"""

from tools.registry import registry
import logging
import json
import os
from pathlib import Path
from typing import Dict, Any, List
import ast
import hashlib
import re
from datetime import datetime

logger = logging.getLogger(__name__)

RESEARCH_DIR = Path.home() / ".hermes" / "workspace" / "github-evolution" / "research"
TOOLS_DIR = Path.home() / ".hermes" / "hermes-agent" / "tools"


def scan_research_jsons() -> List[Dict[str, Any]]:
    """扫描所有 evolver 研究 JSON"""
    if not RESEARCH_DIR.exists():
        return []
    
    results = []
    for json_file in sorted(RESEARCH_DIR.glob("evolver_*.json")):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            results.append({
                'file': str(json_file),
                'topic': data.get('topic', ''),
                'dominant_traits': data.get('dominant_traits', []),
                'trait_count': data.get('trait_count', {}),
                'route': data.get('route', []),
            })
        except Exception as e:
            logger.warning(f"读取 {json_file} 失败: {e}")
    return results


def aggregate_traits(researches: List[Dict[str, Any]]) -> Dict[str, int]:
    """聚合所有研究中的特征"""
    aggregated = {}
    for r in researches:
        for trait_name, count in r.get('trait_count', {}).items():
            aggregated[trait_name] = aggregated.get(trait_name, 0) + count
    return aggregated


def get_existing_tool_names() -> set:
    """获取已存在的工具名"""
    return set(registry._tools.keys())


def generate_tool_skeleton(trait_name: str, total_count: int) -> str:
    """为某个 trait 生成候选工具骨架代码。

    注意：生成物默认是候选骨架，不注册为已完成能力；必须经人工补全业务逻辑、测试和审计后才能晋升。
    """
    safe_name = trait_name.replace('-', '_').replace('.', '_').lower()
    return f'''"""
Auto-generated candidate tool from evolver research
Trait: {trait_name}
Total occurrences across research: {total_count}
Generated: {datetime.utcnow().isoformat()}
Boundary: skeleton_candidate_only_not_real_capability
"""

from tools.registry import registry
import logging

logger = logging.getLogger(__name__)


def evolver_{safe_name}_handler(args):
    """
    {trait_name} - evolver candidate handler.

    该文件只是 GitHub 研究信号生成的候选骨架，尚未实现可交付业务逻辑。
    """
    return {{
        'success': False,
        'status': 'skeleton_candidate_only',
        'trait': '{trait_name}',
        'auto_generated': True,
        'occurrences_in_research': {total_count},
        'message': '候选骨架尚未通过人工实现、测试和审计，不能当作真实能力使用',
        'required_before_promotion': ['fill_real_logic', 'add_tests', 'run_audit', 'manual_approval'],
    }}


registry.register(
    name="evolver_{safe_name}",
    toolset="skills",
    schema={{
        "name": "evolver_{safe_name}",
        "description": "Evolver candidate skeleton for trait '{trait_name}' (not a completed capability)",
        "parameters": {{
            "type": "object",
            "properties": {{
                "context": {{"type": "string", "description": "Task context"}}
            }}
        }}
    }},
    handler=evolver_{safe_name}_handler
)
'''


def audit_generated_tools() -> Dict[str, Any]:
    """审计已生成 evolver 工具，区分真实能力与候选骨架。"""
    files = sorted(TOOLS_DIR.glob('evolver_*_tool.py'))
    rows = []
    for path in files:
        text = path.read_text(encoding='utf-8', errors='replace')
        status = 'implemented_candidate'
        reasons = []
        if 'skeleton_candidate_only' in text or 'Skeleton' in text or '具体实现待补充' in text or 'auto-generated handler' in text:
            status = 'skeleton_only'
            reasons.append('contains_skeleton_marker')
        try:
            ast.parse(text)
            syntax_ok = True
        except SyntaxError as exc:
            syntax_ok = False
            status = 'syntax_error'
            reasons.append(f'syntax_error:{exc.lineno}')
        rows.append({
            'file': str(path),
            'size': path.stat().st_size,
            'sha256': hashlib.sha256(text.encode('utf-8', errors='replace')).hexdigest()[:16],
            'syntax_ok': syntax_ok,
            'status': status,
            'reasons': reasons,
        })
    return {
        'success': True,
        'evolver_tool_files': len(rows),
        'skeleton_only': sum(1 for r in rows if r['status'] == 'skeleton_only'),
        'syntax_errors': sum(1 for r in rows if r['status'] == 'syntax_error'),
        'implemented_candidates': sum(1 for r in rows if r['status'] == 'implemented_candidate'),
        'rows': rows,
        'boundary': 'audit_only_no_registration_no_core_write',
    }


def evolver_auto_learn_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    SE11-C: evolver GitHub 自动学习
    
    Args:
        args:
            mode: 'scan' (扫描研究) | 'generate' (生成工具) | 'stats' (统计) | 'audit' (审计候选骨架)
            min_count: generate 模式下，trait 最小出现次数（默认 5）
            dry_run: True 不实际写入文件
    """
    mode = args.get('mode', 'scan')
    min_count = args.get('min_count', 5)
    dry_run = args.get('dry_run', True)
    
    researches = scan_research_jsons()

    if mode == 'audit':
        return audit_generated_tools()
    
    if mode == 'stats':
        traits = aggregate_traits(researches)
        return {
            'success': True,
            'research_count': len(researches),
            'unique_traits': len(traits),
            'top_traits': sorted(traits.items(), key=lambda x: -x[1])[:10],
            'topics_sample': [r['topic'][:60] for r in researches[:5]],
        }
    
    if mode == 'scan':
        return {
            'success': True,
            'research_count': len(researches),
            'sample': [
                {
                    'file': Path(r['file']).name,
                    'topic': r['topic'][:80],
                    'traits': r['trait_count']
                }
                for r in researches[:5]
            ]
        }
    
    if mode == 'generate':
        traits = aggregate_traits(researches)
        existing_tools = get_existing_tool_names()
        
        # 找出 high-frequency 但还没工具的 traits
        candidates = []
        for trait_name, count in traits.items():
            if count < min_count:
                continue
            safe_name = trait_name.replace('-', '_').replace('.', '_').lower()
            tool_name = f'evolver_{safe_name}'
            if tool_name in existing_tools:
                continue
            candidates.append((trait_name, count, tool_name, safe_name))
        
        generated = []
        for trait_name, count, tool_name, safe_name in candidates:
            tool_file = TOOLS_DIR / f"evolver_{safe_name}_tool.py"
            if not dry_run:
                code = generate_tool_skeleton(trait_name, count)
                tool_file.write_text(code, encoding='utf-8')
            
            generated.append({
                'trait': trait_name,
                'count': count,
                'tool_name': tool_name,
                'tool_file': str(tool_file),
                'written': not dry_run,
            })
        
        return {
            'success': True,
            'mode': 'generate',
            'dry_run': dry_run,
            'min_count': min_count,
            'researches_analyzed': len(researches),
            'unique_traits': len(traits),
            'existing_evolver_tools': sum(1 for t in existing_tools if t.startswith('evolver_')),
            'candidates_count': len(candidates),
            'generated': generated,
        }
    
    return {'success': False, 'error': f'Unknown mode: {mode}'}


registry.register(
    name="evolver_auto_learn",
    toolset="skills",
    schema={
        "name": "evolver_auto_learn",
        "description": "SE11-C: evolver GitHub 自动学习——从 research JSON 自动生成 Hermes 工具",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["scan", "generate", "stats", "audit"],
                    "description": "scan=扫描研究, generate=生成候选工具骨架, stats=统计, audit=审计候选骨架"
                },
                "min_count": {
                    "type": "integer",
                    "description": "generate 模式下 trait 最小出现次数（默认 5）"
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "True 不实际写入文件（默认 True）"
                }
            }
        }
    },
    handler=evolver_auto_learn_handler
)
