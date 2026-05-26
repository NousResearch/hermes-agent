"""devour ΔG 真实计算工具

直接调用已安装的 Rust PyO3 核心 hermes_apex_delta_g.so 做真实 ΔG 计算。
从骨架升级为真实能力。
"""
from tools.registry import registry
import logging
import math

logger = logging.getLogger(__name__)

try:
    import hermes_apex_delta_g as _dg
    _HAVE_SO = True
    _ENGINE = getattr(_dg, 'DeltaGEngine', None)
except Exception:
    _HAVE_SO = False
    _ENGINE = None


def devour_delta_g_handler(args: dict) -> dict:
    """
    ΔG 真实效用计算 — 调用已安装的 Rust PyO3 核心。

    Args:
        mode: 'compute' | 'ping'
        complexity: 任务复杂度（0-1，默认 0.5）
        priority: 任务优先级（0-1，默认 0.3）
        latency_ms: 延迟毫秒（默认 500）
        c_all: 总成本（默认 0）
        sv: 系统资源上确界（默认 1）
    """
    mode = args.get('mode', 'compute')

    if mode == 'ping':
        return {
            'success': True,
            'tool': 'devour_delta_g',
            'engine_available': _HAVE_SO and _ENGINE is not None,
            'engine_version': getattr(_dg, '__version__', 'unknown') if _HAVE_SO else 'N/A',
            'message': 'ΔG tool 已从骨架升级为真实能力' if _HAVE_SO else 'ΔG engine .so 不可用',
        }

    # mode == 'compute'
    complexity = float(args.get('complexity', 0.5))
    priority = float(args.get('priority', 0.3))
    latency_ms = float(args.get('latency_ms', 500.0))
    c_all = float(args.get('c_all', 0.0))
    sv = float(args.get('sv', 1.0))

    # 初始化降级默认值
    ev = bv = av = 0.0
    harm_rate = 0.0
    should_terminate = False
    _FALLBACK = False

    if _HAVE_SO and _ENGINE:
        try:
            engine = _ENGINE()
            result = engine.compute_ev(complexity, priority, latency_ms)
            if isinstance(result, dict):
                ev, bv, av = result.get('ev', 0), result.get('bv', 0), result.get('av', 0)
                harm_rate = result.get('harm_rate', 0.0)
                should_terminate = result.get('should_terminate', False)
            else:
                ev, bv, av = float(result), 0.0, 0.0
                harm_rate = 0.0
                should_terminate = False
        except Exception as exc:
            logger.warning(f"Rust ΔG 引擎调用失败，降级: {exc}")
            _FALLBACK = True
    else:
        _FALLBACK = True

    if _FALLBACK:
        # Python 降级计算（与原 Rust 公式一致）
        task_entropy = complexity * (1.0 - priority)
        bv = complexity * priority * (1.0 + task_entropy)
        av = (1.0 - priority) * math.exp(-latency_ms / 2000.0) * complexity
        ev = bv + av
        harm_rate = max(0, 1.0 - ev / (complexity + 0.01))
        should_terminate = ev < 0 or harm_rate > 0.34
        if c_all > 0:
            should_terminate = should_terminate or (c_all > sv)

    return {
        'success': True,
        'tool': 'devour_delta_g',
        'engine': 'rust_pyo3' if _HAVE_SO and _ENGINE else 'python_fallback',
        'ev': round(ev, 6),
        'bv': round(bv, 6),
        'av': round(av, 6),
        'c_all': c_all,
        'sv': sv,
        'harm_rate': round(harm_rate, 4),
        'should_terminate': should_terminate,
        'message': f"EV={ev:.4f} (BV={bv:.4f}+AV={av:.4f}) harm={harm_rate:.2%} {'☠️湮灭' if should_terminate else '✅存活'}",
    }


registry.register(
    name="devour_delta_g",
    toolset="skills",
    schema={
        "name": "devour_delta_g",
        "description": "ΔG 真实效用计算引擎——从骨架升级为真实能力，调用 Rust PyO3 核心或 Python 降级",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["compute", "ping"],
                    "description": "compute=计算 ΔG, ping=检测引擎状态"
                },
                "complexity": {"type": "number", "description": "任务复杂度（0-1）"},
                "priority": {"type": "number", "description": "任务优先级（0-1）"},
                "latency_ms": {"type": "number", "description": "延迟毫秒"},
                "c_all": {"type": "number", "description": "总成本"},
                "sv": {"type": "number", "description": "系统上确界"}
            }
        }
    },
    handler=devour_delta_g_handler
)
