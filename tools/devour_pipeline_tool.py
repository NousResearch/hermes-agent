"""
SE12-B: devour 自动集成闭环 (吞噬闭环)
================================================

完整闭环：
1. devour scan 扫描外部项目
2. devour extract 抽取关键能力
3. 生成 Hermes 工具骨架
4. 写入 tools/devour_<name>_tool.py
5. registry.register() 注册

注：实际调用 LLM 生成具体逻辑超出本地实现范围，
本工具生成"骨架 + 引导描述"，由后续 evolve 流程填充实现。
"""

from tools.registry import registry
import logging
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import ast
import hashlib
import re
from datetime import datetime

logger = logging.getLogger(__name__)

DEVOUR_BIN = Path.home() / ".local" / "bin" / "devour"
TOOLS_DIR = Path.home() / ".hermes" / "hermes-agent" / "tools"
DEVOUR_INTAKE_DIR = Path.home() / ".hermes" / "workspace" / "devour-intake"


def run_devour_scan(repo_dir: str, pattern: str = "*.py") -> List[Dict[str, Any]]:
    """运行 devour scan"""
    if not DEVOUR_BIN.exists():
        raise RuntimeError(f"devour 二进制不存在: {DEVOUR_BIN}")
    
    result = subprocess.run(
        [str(DEVOUR_BIN), "scan", "--dir", repo_dir, "--pattern", pattern],
        capture_output=True, text=True, timeout=60
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"devour scan 失败: {result.stderr}")
    
    return json.loads(result.stdout)


def run_devour_extract(file_path: str, min_lines: int = 5, max_lines: int = 200) -> Dict[str, Any]:
    """运行 devour extract（返回值标准化为 dict 含 functions 列表）"""
    result = subprocess.run(
        [str(DEVOUR_BIN), "extract", "--file", file_path,
         "--min-lines", str(min_lines), "--max-lines", str(max_lines)],
        capture_output=True, text=True, timeout=60
    )
    
    if result.returncode != 0:
        return {"error": result.stderr, "functions": []}
    
    try:
        parsed = json.loads(result.stdout)
        # devour extract 返回列表，标准化为 dict
        if isinstance(parsed, list):
            return {"functions": parsed}
        return parsed
    except Exception:
        return {"raw": result.stdout, "functions": []}


def generate_devour_tool_skeleton(name: str, source_repo: str, capabilities: Dict[str, Any]) -> str:
    """为吞噬来的能力生成候选工具骨架。

    生成物默认不是已完成能力；必须经过许可证检查、真实逻辑实现、测试和人工批准后才能晋升。
    """
    safe_name = name.replace('-', '_').replace('.', '_').lower()
    return f'''"""
Auto-generated candidate tool from devour (SE12 吞噬自进化)
Source: {source_repo}
Generated: {datetime.utcnow().isoformat()}
Capabilities: {len(capabilities.get("functions", []))} functions
Boundary: skeleton_candidate_only_not_real_capability
"""

from tools.registry import registry
import logging

logger = logging.getLogger(__name__)


def devour_{safe_name}_handler(args):
    """
    {name} - devour candidate handler.

    源仓库: {source_repo}
    该文件只是 devour scan/extract 生成的候选骨架，尚未实现可交付业务逻辑。
    """
    return {{
        'success': False,
        'status': 'skeleton_candidate_only',
        'tool': 'devour_{safe_name}',
        'source_repo': '{source_repo}',
        'auto_generated': True,
        'capabilities_count': {len(capabilities.get("functions", []))},
        'message': '候选骨架尚未通过许可证审计、真实实现、测试和人工批准，不能当作真实能力使用',
        'required_before_promotion': ['license_audit', 'fill_real_logic', 'add_tests', 'run_security_audit', 'manual_approval'],
    }}


registry.register(
    name="devour_{safe_name}",
    toolset="skills",
    schema={{
        "name": "devour_{safe_name}",
        "description": "Devour candidate skeleton from {source_repo} (not a completed capability)",
        "parameters": {{
            "type": "object",
            "properties": {{
                "context": {{"type": "string", "description": "Task context"}}
            }}
        }}
    }},
    handler=devour_{safe_name}_handler
)
'''


def audit_devour_tools() -> Dict[str, Any]:
    """审计 devour 生成工具，区分候选骨架和真实能力。"""
    files = sorted(TOOLS_DIR.glob('devour_*_tool.py'))
    rows = []
    for path in files:
        text = path.read_text(encoding='utf-8', errors='replace')
        status = 'implemented_candidate'
        reasons = []
        if 'skeleton_candidate_only' in text or 'Skeleton' in text or '候选骨架' in text or 'extend with LLM-generated logic' in text:
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
        'devour_tool_files': len(rows),
        'skeleton_only': sum(1 for r in rows if r['status'] == 'skeleton_only'),
        'syntax_errors': sum(1 for r in rows if r['status'] == 'syntax_error'),
        'implemented_candidates': sum(1 for r in rows if r['status'] == 'implemented_candidate'),
        'rows': rows,
        'boundary': 'audit_only_no_registration_no_core_write',
    }


def devour_pipeline_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    SE12-B: devour 自动集成闭环
    
    Args:
        args:
            mode: 'scan' | 'extract' | 'integrate' | 'stats' | 'audit'
            repo_dir: 要扫描的目录
            pattern: 文件模式（默认 *.py）
            min_files: 最小函数数才生成工具（默认 3）
            dry_run: True 不实际写入文件
    """
    mode = args.get('mode', 'scan')
    repo_dir = args.get('repo_dir', '')
    pattern = args.get('pattern', '*.py')
    min_funcs = args.get('min_funcs', 3)
    dry_run = args.get('dry_run', True)
    
    if mode == 'audit':
        return audit_devour_tools()

    if mode == 'stats':
        # 统计已生成的 devour 工具
        existing = sorted(TOOLS_DIR.glob('devour_*_tool.py'))
        return {
            'success': True,
            'devour_tools_count': len(existing),
            'tools': [t.name for t in existing[:10]],
        }
    
    if not repo_dir:
        return {'success': False, 'error': '需要 repo_dir 参数'}
    
    repo_path = Path(repo_dir).expanduser().resolve()
    if not repo_path.exists():
        return {'success': False, 'error': f'目录不存在: {repo_path}'}
    
    try:
        # 1. scan
        scan_result = run_devour_scan(str(repo_path), pattern)
        
        if mode == 'scan':
            return {
                'success': True,
                'mode': 'scan',
                'repo_dir': str(repo_path),
                'pattern': pattern,
                'files_count': len(scan_result),
                'sample': scan_result[:5],
            }
        
        # 2. extract for each file
        extracted = {}
        if mode in ('extract', 'integrate'):
            for entry in scan_result[:10]:  # 限制前 10 个
                file_path = repo_path / entry['file']
                if not file_path.exists():
                    file_path = next(repo_path.rglob(entry['file']), None)
                    if not file_path:
                        continue
                extract_result = run_devour_extract(str(file_path))
                extracted[entry['file']] = extract_result
        
        if mode == 'extract':
            return {
                'success': True,
                'mode': 'extract',
                'extracted_count': len(extracted),
                'extracted_keys': list(extracted.keys()),
            }
        
        # 3. integrate: 生成工具
        if mode == 'integrate':
            # 收集所有合格能力
            DEVOUR_INTAKE_DIR.mkdir(parents=True, exist_ok=True)
            
            generated = []
            for file_name, extract_data in extracted.items():
                funcs = extract_data.get('functions', [])
                if len(funcs) < min_funcs:
                    continue
                
                # 工具名 = 文件名 stem
                tool_stem = Path(file_name).stem
                safe_name = tool_stem.replace('-', '_').lower()
                tool_file = TOOLS_DIR / f"devour_{safe_name}_tool.py"
                
                if tool_file.exists():
                    continue  # 已存在，跳过
                
                # 生成骨架
                code = generate_devour_tool_skeleton(
                    name=tool_stem,
                    source_repo=str(repo_path),
                    capabilities={'functions': funcs}
                )
                
                # 保存抽取结果到 intake
                intake_file = DEVOUR_INTAKE_DIR / f"{tool_stem}_extract.json"
                intake_file.write_text(
                    json.dumps(extract_data, ensure_ascii=False, indent=2),
                    encoding='utf-8'
                )
                
                if not dry_run:
                    tool_file.write_text(code, encoding='utf-8')
                
                generated.append({
                    'tool_name': f'devour_{safe_name}',
                    'tool_file': str(tool_file),
                    'source_file': file_name,
                    'functions_count': len(funcs),
                    'intake_file': str(intake_file),
                    'written': not dry_run,
                })
            
            return {
                'success': True,
                'mode': 'integrate',
                'dry_run': dry_run,
                'repo_dir': str(repo_path),
                'files_scanned': len(scan_result),
                'files_extracted': len(extracted),
                'tools_generated': len(generated),
                'generated': generated,
            }
        
        return {'success': False, 'error': f'未知 mode: {mode}'}
        
    except Exception as e:
        logger.error(f"devour_pipeline 异常: {e}")
        return {'success': False, 'error': str(e)}


registry.register(
    name="devour_pipeline",
    toolset="skills",
    schema={
        "name": "devour_pipeline",
        "description": "SE12-B: devour 自动集成闭环——scan/extract/integrate/stats",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["scan", "extract", "integrate", "stats", "audit"],
                    "description": "scan=扫描, extract=抽取, integrate=生成候选骨架, stats=统计, audit=审计候选骨架"
                },
                "repo_dir": {"type": "string", "description": "要吞噬的目录"},
                "pattern": {"type": "string", "description": "文件模式（默认 *.py）"},
                "min_funcs": {"type": "integer", "description": "最小函数数（默认 3）"},
                "dry_run": {"type": "boolean", "description": "True 不实际写入"}
            }
        }
    },
    handler=devour_pipeline_handler
)
