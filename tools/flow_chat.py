"""判定一个 langflow flow 是否为「ChatInput + ChatOutput」一进一出对话流。

**硬约定**(见 easyhermes-copilot-authored-flows-roles):工作流即工具(MCP)永远只支持
对话流 —— 一进一出,干净映射成 ``{message}→string``。两处必须用**同一个判据**,否则会漂移:
  - 注册口 ``tools/expose_flow_tool.expose_flow_as_tool`` —— 拦住非对话流被暴露成工具;
  - 资源采集 ``hermes_cli/org_client.gather_workflow_resources`` —— 只把对话流纳入可注册池。
所以判据集中放这里,两边都 import。纯函数、只依赖标准库。
"""

from __future__ import annotations


def flow_node_types(flow: dict) -> set[str]:
    """flow 里出现的节点组件类型集合。取 ``node.data.type``;为空时回退节点 id 前缀
    (形如 ``ChatInput-AbC12`` → ``ChatInput``),兼容不同 langflow 版本的导出结构。"""
    types: set[str] = set()
    nodes = ((flow.get("data") or {}).get("nodes")) or []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        t = str((n.get("data") or {}).get("type") or "").strip()
        if not t:
            t = str(n.get("id") or "").split("-", 1)[0].strip()
        if t:
            types.add(t)
    return types


def is_chat_flow(flow: dict) -> bool:
    """是否「ChatInput + ChatOutput」对话流(同时含对话入口与出口)。"""
    if not isinstance(flow, dict):
        return False
    types = flow_node_types(flow)
    return "ChatInput" in types and "ChatOutput" in types
