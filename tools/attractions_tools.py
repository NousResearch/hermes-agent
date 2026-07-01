#!/usr/bin/env python3
"""Attractions demo tool — send_attraction_card (push agent-generated attractions to SSE as a card)."""

from typing import Any, Callable, Dict, List, Optional


def send_attraction_card(
    attractions: List[Dict[str, Any]], push_card: Optional[Callable] = None
) -> str:
    """把景点作为 card 事件推送到前端 SSE。返回发送结果（JSON 字符串）。

    景点数据由 agent 自行组织（LLM 知识），经 attractions 参数传入——不再有
    find_attractions mock 数据源。push_card 由 tool_executor 注入（照 clarify
    模式），签名：push_card(card_type, data)。

    返回 JSON 字符串而非 dict：hermes make_tool_result_message 对 list/dict
    content 不 stringify（pass through），严格 provider 会因 content[].type
    缺失报 HTTP 400。字符串 content 是 OpenAI tool-result 惯例，所有 provider 接受。
    """
    import time, json

    time.sleep(2.5)  # 模拟工具执行耗时，测试前端流式（card 事件在 tool.started 后延迟推送）
    data = list(attractions or [])
    if push_card:
        push_card("attractions", data)
    return json.dumps({"sent": True, "count": len(data)}, ensure_ascii=False)


# --- OpenAI function-calling schema ---
SEND_ATTRACTION_CARD_SCHEMA = {
    "name": "send_attraction_card",
    "description": (
        "把景点列表以卡片形式推送给用户。当用户想看景点/地点推荐（如「X 有什么景点」"
        "「推荐几个地方」「做行程」）时调用。你（agent）自行组织景点数据传入 attractions，"
        "每个景点是 {'name': 名称, 'desc': 一句话亮点, 'image': 图片URL}，"
        "image 用 https://picsum.photos/seed/<景点拼音或英文>/400/240。"
        "发送后用简短文本引导，不要在文本里复述卡片内容。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "attractions": {
                "type": "array",
                "items": {"type": "object"},
                "description": "景点数组（你自行组织的 {name, desc, image}）",
            },
        },
        "required": ["attractions"],
    },
}


# --- Registry ---
from tools.registry import registry  # noqa: E402

registry.register(
    name="send_attraction_card",
    toolset="attractions",
    schema=SEND_ATTRACTION_CARD_SCHEMA,
    # push_card 由 tool_executor 注入；这里 handler 仅占位，真实调用走 tool_executor 分支
    handler=lambda args, **kw: send_attraction_card(attractions=args.get("attractions")),
    check_fn=lambda: True,
    emoji="💳",
)
