"""
APEX ΔE Evolution Engine 工具 - 超级进化13

CLAW 神技能开启 - APEX ΔE 公式原生 Rust 实现接口

核心公式：
APEX_ΔE = α·Ψ + β·Ω + λ·Φ + ∇Θ + Evol_code

五维含义：
- α·Ψ: 意识逻辑基底 (truth gate)
- β·Ω: 代码底层构架 (Rust native core, NOT sidecar)
- λ·Φ: 全网知识溯源 (GitHub + arXiv real-time scout)
- ∇Θ: 认知迭代差值 (eval center trace fullness)
- Evol_code: 原生代码级永续演化 (event-driven, NOT cron)

底层引擎（双模式）：
- 优先：hermes_apex_evolution PyO3 .so 嵌入式核心（Hermes 主进程内调用）
- 降级：apex13 binary（subprocess 调用，CLI 备用）
"""

from tools.registry import registry
import logging
import subprocess
import json
import os
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

APEX13_BIN = os.path.expanduser("~/.local/bin/apex13")
EVOLUTION_DIR = os.path.expanduser("~/.hermes/workspace/evolution/super_evolution13")

# 优先尝试加载 PyO3 嵌入式核心
try:
    import hermes_apex_evolution as _native
    _NATIVE_AVAILABLE = True
    logger.info(f"✅ APEX13 PyO3 native core loaded: {_native.version()}")
except ImportError:
    _native = None
    _NATIVE_AVAILABLE = False
    logger.info("⚠️  APEX13 PyO3 native core not available, using subprocess CLI")


def _run_apex13(args: list, timeout: int = 60) -> Dict[str, Any]:
    """执行 apex13 子命令"""
    if not os.path.exists(APEX13_BIN):
        return {'success': False, 'error': f'apex13 binary not found at {APEX13_BIN}'}
    
    try:
        result = subprocess.run(
            [APEX13_BIN] + args,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'exit_code': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': f'timeout after {timeout}s'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def apex13_scout_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """λ·Φ 全网知识溯源：实时抓取 GitHub + arXiv"""
    topics = args.get('topics', 'agent_framework,agentic_rl,context_learning')
    output = args.get('output', f'{EVOLUTION_DIR}/source_scout.json')
    
    Path(EVOLUTION_DIR).mkdir(parents=True, exist_ok=True)
    
    res = _run_apex13(['scout', '--topics', topics, '--output', output], timeout=120)
    
    if not res['success']:
        return {'success': False, 'error': res.get('error') or res.get('stderr'), 'message': '❌ Scout 失败'}
    
    try:
        with open(output) as f:
            data = json.load(f)
        
        total_repos = sum(len(q['repos']) for q in data['queries'])
        total_papers = sum(len(q['papers']) for q in data['queries'])
        
        return {
            'success': True,
            'topics': topics.split(','),
            'total_repos': total_repos,
            'total_papers': total_papers,
            'output_file': output,
            'message': f"✅ λ·Φ Scout 完成: {len(data['queries'])} 主题, {total_repos} 仓库, {total_papers} 论文"
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def apex13_audit_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """β·Ω 代码底层冗余审计（Rust 原生扫描）"""
    roots = args.get('roots', os.path.expanduser('~/.hermes/workspace'))
    output = args.get('output', f'{EVOLUTION_DIR}/redundancy_audit.json')
    
    Path(EVOLUTION_DIR).mkdir(parents=True, exist_ok=True)
    
    res = _run_apex13(['audit', '--roots', roots, '--output', output], timeout=300)
    
    if not res['success']:
        return {'success': False, 'error': res.get('error') or res.get('stderr'), 'message': '❌ Audit 失败'}
    
    try:
        with open(output) as f:
            data = json.load(f)
        
        return {
            'success': True,
            'total_files': data.get('total_files', 0),
            'duplicate_groups': len(data.get('duplicate_groups', [])),
            'large_files': len(data.get('large_files', [])),
            'redundant_reports': len(data.get('redundant_reports', [])),
            'output_file': output,
            'message': f"✅ β·Ω Audit 完成: {data.get('total_files', 0)} 文件, {len(data.get('duplicate_groups', []))} 重复组"
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def apex13_eval_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """∇Θ 认知迭代评估：实时计算 APEX ΔE 五维分数"""
    workspace = args.get('workspace', os.path.expanduser('~/.hermes/workspace'))
    output = args.get('output', f'{EVOLUTION_DIR}/evaluation_matrix.json')
    
    Path(EVOLUTION_DIR).mkdir(parents=True, exist_ok=True)
    
    res = _run_apex13(['eval', '--workspace', workspace, '--output', output], timeout=60)
    
    if not res['success']:
        return {'success': False, 'error': res.get('error') or res.get('stderr'), 'message': '❌ Eval 失败'}
    
    try:
        with open(output) as f:
            data = json.load(f)
        
        scores = data.get('scores', {})
        return {
            'success': True,
            'apex_delta_e': data.get('apex_delta_e', 0),
            'scores': scores,
            'core_modified': data.get('core_modified', False),
            'output_file': output,
            'message': (
                f"✅ APEX ΔE = {data.get('apex_delta_e', 0):.3f} | "
                f"α·Ψ={scores.get('alpha_psi', 0):.2f} "
                f"β·Ω={scores.get('beta_omega', 0):.2f} "
                f"λ·Φ={scores.get('lambda_phi', 0):.2f} "
                f"∇Θ={scores.get('nabla_theta', 0):.2f} "
                f"Evol={scores.get('evol_code', 0):.2f}"
            )
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def apex13_evol_status_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Evol_code 永续演化引擎状态查询（事件日志统计）"""
    log_path = args.get('log', f'{EVOLUTION_DIR}/evol_events.jsonl')
    
    if not os.path.exists(log_path):
        return {
            'success': True,
            'status': 'not_started',
            'event_count': 0,
            'action_count': 0,
            'message': '⚠️  Evol_code 引擎尚未启动。运行 apex13 evol 启动事件驱动演化。'
        }
    
    event_count = 0
    action_count = 0
    last_action = None
    
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if '_evol_action' in obj:
                        action_count += 1
                        last_action = obj['_evol_action']
                    else:
                        event_count += 1
                except json.JSONDecodeError:
                    continue
        
        return {
            'success': True,
            'status': 'active' if event_count > 0 else 'idle',
            'event_count': event_count,
            'action_count': action_count,
            'last_action': last_action,
            'log_path': log_path,
            'message': f"✅ Evol_code 状态: {event_count} 事件 / {action_count} 演化动作"
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def apex13_full_cycle_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """APEX ΔE 完整闭环：scout → audit → eval 一键运行"""
    results = {}
    
    # λ·Φ Scout
    scout_res = apex13_scout_handler({'topics': args.get('topics', 'agent_framework,agentic_rl,context_learning')})
    results['scout'] = scout_res
    
    # β·Ω Audit
    audit_res = apex13_audit_handler({'roots': args.get('roots', os.path.expanduser('~/.hermes/workspace'))})
    results['audit'] = audit_res
    
    # ∇Θ Eval (这个会读取 scout/audit 结果计算最新分数)
    eval_res = apex13_eval_handler({})
    results['eval'] = eval_res
    
    apex_delta_e = eval_res.get('apex_delta_e', 0) if eval_res.get('success') else 0
    
    return {
        'success': all(r.get('success', False) for r in results.values()),
        'apex_delta_e': apex_delta_e,
        'phases': results,
        'message': f"🌟 APEX ΔE 完整闭环: scout/audit/eval 全部完成, ΔE = {apex_delta_e:.3f}"
    }


def apex13_native_start_evol_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """启动 PyO3 嵌入式 Evol_code watcher（在 Hermes 主进程内运行）"""
    if not _NATIVE_AVAILABLE:
        return {
            'success': False,
            'error': 'PyO3 native core not available',
            'message': '❌ 嵌入式核心未加载，无法启动主进程内 watcher'
        }
    
    watch_dirs = args.get('watch_dirs', [os.path.expanduser('~/.hermes/workspace')])
    if isinstance(watch_dirs, str):
        watch_dirs = watch_dirs.split(',')
    
    log_path = args.get('log', f'{EVOLUTION_DIR}/evol_events.jsonl')
    threshold = args.get('threshold', 10)
    
    Path(EVOLUTION_DIR).mkdir(parents=True, exist_ok=True)
    
    try:
        result = _native.start_evol_watcher(watch_dirs, log_path, threshold)
        return {
            'success': True,
            'native_mode': True,
            'watch_dirs': watch_dirs,
            'log_path': log_path,
            'threshold': threshold,
            'message': result
        }
    except Exception as e:
        return {'success': False, 'error': str(e), 'message': f'❌ Evol watcher 启动失败: {e}'}


def apex13_native_stop_evol_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """停止 PyO3 嵌入式 Evol_code watcher"""
    if not _NATIVE_AVAILABLE:
        return {'success': False, 'error': 'PyO3 native core not available'}
    
    try:
        result = _native.stop_evol_watcher()
        return {'success': True, 'message': result}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def apex13_native_evol_status_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """查询 PyO3 嵌入式 watcher 实时运行状态"""
    if not _NATIVE_AVAILABLE:
        return {'success': False, 'error': 'PyO3 native core not available'}
    
    try:
        status = _native.evol_watcher_status()
        return {
            'success': True,
            'native_mode': True,
            'running': dict(status).get('running', False),
            'log_path': dict(status).get('log_path', ''),
            'message': f"嵌入式 watcher: {'🟢 运行中' if dict(status).get('running') else '⚪ 未启动'}"
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def apex13_native_info_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """查询 PyO3 嵌入式核心信息"""
    if not _NATIVE_AVAILABLE:
        return {
            'success': False,
            'native_available': False,
            'message': '❌ PyO3 嵌入式核心未加载'
        }
    
    try:
        return {
            'success': True,
            'native_available': True,
            'version': _native.version(),
            'so_path': '/Users/appleoppa/.hermes/hermes-agent/venv/lib/python3.11/site-packages/hermes_apex_evolution.so',
            'message': f'✅ {_native.version()}'
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


# 注册工具

registry.register(
    name="apex13_scout",
    toolset="skills",
    schema={
        "name": "apex13_scout",
        "description": "λ·Φ 全网知识溯源（超级进化13）。Rust 原生实时抓取 GitHub 仓库 + arXiv 论文元数据，只读不 clone。",
        "parameters": {
            "type": "object",
            "properties": {
                "topics": {"type": "string", "description": "查询主题（逗号分隔），默认 agent_framework,agentic_rl,context_learning"},
                "output": {"type": "string", "description": "输出 JSON 路径"}
            }
        }
    },
    handler=apex13_scout_handler
)

registry.register(
    name="apex13_audit",
    toolset="skills",
    schema={
        "name": "apex13_audit",
        "description": "β·Ω 代码底层冗余审计（超级进化13）。Rust 原生扫描重复文件、大文件、陈旧 .pyc，只读不删除。",
        "parameters": {
            "type": "object",
            "properties": {
                "roots": {"type": "string", "description": "扫描根目录（逗号分隔），默认 ~/.hermes/workspace"},
                "output": {"type": "string", "description": "输出 JSON 路径"}
            }
        }
    },
    handler=apex13_audit_handler
)

registry.register(
    name="apex13_eval",
    toolset="skills",
    schema={
        "name": "apex13_eval",
        "description": "∇Θ 认知迭代评估（超级进化13）。实时计算 APEX ΔE = α·Ψ + β·Ω + λ·Φ + ∇Θ + Evol_code 五维分数。",
        "parameters": {
            "type": "object",
            "properties": {
                "workspace": {"type": "string", "description": "评估工作区，默认 ~/.hermes/workspace"},
                "output": {"type": "string", "description": "输出 JSON 路径"}
            }
        }
    },
    handler=apex13_eval_handler
)

registry.register(
    name="apex13_evol_status",
    toolset="skills",
    schema={
        "name": "apex13_evol_status",
        "description": "Evol_code 永续演化引擎状态查询（超级进化13）。返回事件计数、演化动作数、最近动作。",
        "parameters": {
            "type": "object",
            "properties": {
                "log": {"type": "string", "description": "事件日志路径"}
            }
        }
    },
    handler=apex13_evol_status_handler
)

registry.register(
    name="apex13_full_cycle",
    toolset="skills",
    schema={
        "name": "apex13_full_cycle",
        "description": "APEX ΔE 完整闭环（超级进化13）。一键运行 scout（λ·Φ）+ audit（β·Ω）+ eval（∇Θ）三阶段，返回综合 ΔE 分数。",
        "parameters": {
            "type": "object",
            "properties": {
                "topics": {"type": "string", "description": "scout 主题"},
                "roots": {"type": "string", "description": "audit 根目录"}
            }
        }
    },
    handler=apex13_full_cycle_handler
)

# === PyO3 嵌入式核心专属工具 ===

registry.register(
    name="apex13_native_start_evol",
    toolset="skills",
    schema={
        "name": "apex13_native_start_evol",
        "description": "启动 Evol_code 永续演化 watcher（PyO3 嵌入式，在 Hermes 主进程内后台线程运行）。监听文件系统事件，达到阈值触发演化动作。",
        "parameters": {
            "type": "object",
            "properties": {
                "watch_dirs": {"type": "string", "description": "监听目录（逗号分隔），默认 ~/.hermes/workspace"},
                "log": {"type": "string", "description": "事件日志路径"},
                "threshold": {"type": "integer", "description": "触发演化的事件阈值，默认 10"}
            }
        }
    },
    handler=apex13_native_start_evol_handler
)

registry.register(
    name="apex13_native_stop_evol",
    toolset="skills",
    schema={
        "name": "apex13_native_stop_evol",
        "description": "停止 PyO3 嵌入式 Evol_code watcher。",
        "parameters": {"type": "object", "properties": {}}
    },
    handler=apex13_native_stop_evol_handler
)

registry.register(
    name="apex13_native_evol_status",
    toolset="skills",
    schema={
        "name": "apex13_native_evol_status",
        "description": "查询 PyO3 嵌入式 Evol_code watcher 实时运行状态。",
        "parameters": {"type": "object", "properties": {}}
    },
    handler=apex13_native_evol_status_handler
)

registry.register(
    name="apex13_native_info",
    toolset="skills",
    schema={
        "name": "apex13_native_info",
        "description": "查询 PyO3 嵌入式核心信息（版本、.so 路径、加载状态）。",
        "parameters": {"type": "object", "properties": {}}
    },
    handler=apex13_native_info_handler
)

logger.info("✅ APEX13 (超级进化13) 工具已注册: scout/audit/eval/evol_status/full_cycle + 4 native tools")
