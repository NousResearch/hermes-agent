"""模型故障自动降级——量子路由切换器

当调用模型返回 429/503/超时等错误时，自动调用 qr route
查询可用模型并降级到其他供应商。

用法：
    from tools.model_failover_tool import auto_failover
    model, provider = auto_failover("当前任务描述")
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from typing import Any, Dict, Optional, Tuple

from tools.registry import registry

logger = logging.getLogger(__name__)

_QR_BIN = None


def _resolve_qr() -> str:
    global _QR_BIN
    if _QR_BIN is not None:
        return _QR_BIN
    candidates = [
        "/Users/appleoppa/.cargo/bin/qr",
    ]
    for c in candidates:
        import os
        if os.path.isfile(c) and os.access(c, os.X_OK):
            _QR_BIN = c
            return c
    _QR_BIN = "qr"
    return "qr"


def _qr_route(task: str) -> Dict[str, Any]:
    """调用 qr route，返回路由结果"""
    qr = _resolve_qr()
    cmd = [qr, "route", task]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        logger.warning("qr route failed (exit %d): %s", result.returncode, result.stderr[:200])
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as e:
        logger.warning("qr route exception: %s", e)
    return {}


# 降级优先级：C(Claude) -> B(DeepSeek) -> A(GPT) -> D(astron)
_FALLOVER_TIER_ORDER = ["C", "B", "A", "D"]


def auto_failover(
    task: str = "",
    failed_model: str = "",
    failed_provider: str = "",
) -> Tuple[str, str]:
    """自动故障降级。

    Args:
        task: 当前任务描述（用于路由）
        failed_model: 失败的模型名（可选）
        failed_provider: 失败的供应商名（可选）

    Returns:
        (model, provider) 可用模型，或 ("", "") 表示无可用降级通路
    """
    result = _qr_route(task or "通用任务")

    if not result:
        logger.error("量子路由不可用，无法降级")
        return ("", "")

    # 如果路由已有结果，直接用
    selected = result.get("selected") or result.get("model")
    all_online = result.get("all_online") or []
    health = result.get("health") or []

    if selected and selected != failed_provider:
        # 找到对应 provider 的模型名
        for h in health:
            if isinstance(h, dict) and h.get("name") == selected:
                return (h.get("model") or "", selected)
        return ("", selected)

    # 如果选中结果失败，手动遍历降级
    for tier in _FALLOVER_TIER_ORDER:
        for h in health:
            if (
                isinstance(h, dict)
                and h.get("status") == "ok"
                and h.get("name") not in (failed_provider, "")
                and h.get("name") in all_online
                and h.get("tier") == tier
            ):
                return (h.get("model") or "", h.get("name") or "")

    # 兜底：任何在线但不是失败的
    for h in health:
        if isinstance(h, dict) and h.get("status") == "ok" and h.get("name") != failed_provider:
            return (h.get("model") or "", h.get("name") or "")

    return ("", "")


def _handler(_args: dict) -> str:
    task = _args.get("task", "通用任务")
    failed_model = _args.get("failed_model", "")
    failed_provider = _args.get("failed_provider", "")
    model, provider = auto_failover(task, failed_model, failed_provider)
    if model:
        return f"降级成功：{provider}/{model}"
    return "无可用降级通路，请检查量子路由状态"


registry.register(
    name="model_failover",
    toolset="skills",
    schema={
        "description": "模型故障自动降级——当模型返回429/503/超时时，调用量子路由查询可用模型并降级",
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "当前任务描述", "default": "通用任务"},
                "failed_model": {"type": "string", "description": "失败的模型名（可选）"},
                "failed_provider": {"type": "string", "description": "失败的供应商名（可选）"},
            },
            "required": [],
        },
    },
    handler=_handler,
)
