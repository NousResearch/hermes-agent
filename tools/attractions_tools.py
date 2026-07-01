#!/usr/bin/env python3
"""Attractions demo tools — find_attractions (mock) + send_attraction_card (push card to SSE)."""

from typing import Any, Callable, Dict, List, Optional


# mock 景点数据
ATTRACTIONS: Dict[str, List[Dict[str, Any]]] = {
    "北京": [
        {"name": "故宫", "desc": "明清两代皇宫，世界文化遗产。", "image": "https://picsum.photos/seed/gugong/400/240"},
        {"name": "长城", "desc": "世界中古七大奇迹之一。", "image": "https://picsum.photos/seed/changcheng/400/240"},
        {"name": "颐和园", "desc": "清代皇家园林。", "image": "https://picsum.photos/seed/yiheyuan/400/240"},
    ],
    "上海": [
        {"name": "外滩", "desc": "黄浦江畔的历史建筑群。", "image": "https://picsum.photos/seed/waitan/400/240"},
        {"name": "迪士尼乐园", "desc": "中国大陆首座迪士尼主题乐园。", "image": "https://picsum.photos/seed/disney/400/240"},
    ],
}


def find_attractions(city: str) -> str:
    """返回指定城市的景点列表（mock，JSON 字符串）。

    返回 JSON 字符串而非 list：hermes make_tool_result_message 对 list/dict content
    不 stringify（pass through），严格 provider 会因 content[].type 缺失报 HTTP 400。
    字符串 content 是 OpenAI tool-result 惯例，所有 provider 接受。
    """
    import json, time
    time.sleep(2.5)  # 模拟工具执行耗时，测试前端流式（tool.started → 等 → tool.completed）
    city = (city or "").strip()
    data = ATTRACTIONS.get(city, [{"name": f"{city or '该城市'}暂无景点数据", "desc": "请换个城市试试。", "image": ""}])
    return json.dumps(data, ensure_ascii=False)


def send_attraction_card(attractions: List[Dict[str, Any]], push_card: Optional[Callable] = None) -> Dict[str, Any]:
    """把景点作为 card 事件推送到前端 SSE。返回发送结果。

    push_card 由 tool_executor 注入（照 clarify 模式），签名：push_card(card_type, data)。
    """
    import time
    time.sleep(2.5)  # 模拟工具执行耗时，测试前端流式（card 事件在 tool.started 后延迟推送）
    data = list(attractions or [])
    if push_card:
        push_card("attractions", data)
    import json
    return json.dumps({"sent": True, "count": len(data)}, ensure_ascii=False)


# --- OpenAI function-calling schema ---
FIND_ATTRACTIONS_SCHEMA = {
    "name": "find_attractions",
    "description": "查找指定城市的著名景点。返回景点列表（含名称、简介、图片）。",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名，如「北京」「上海」"},
        },
        "required": ["city"],
    },
}

SEND_ATTRACTION_CARD_SCHEMA = {
    "name": "send_attraction_card",
    "description": (
        "把景点列表以卡片形式推送给用户。当用户要看景点时，先用 find_attractions 拿数据，"
        "再用本工具发送卡片。发送后用简短文本总结即可，不要在文本里重复景点完整信息。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "attractions": {
                "type": "array",
                "items": {"type": "object"},
                "description": "景点数组（find_attractions 的返回值）",
            },
        },
        "required": ["attractions"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error  # noqa: E402

registry.register(
    name="find_attractions",
    toolset="attractions",
    schema=FIND_ATTRACTIONS_SCHEMA,
    handler=lambda args, **kw: find_attractions(city=args.get("city", "")),
    check_fn=lambda: True,
    emoji="🏯",
)

registry.register(
    name="send_attraction_card",
    toolset="attractions",
    schema=SEND_ATTRACTION_CARD_SCHEMA,
    # push_card 由 tool_executor 注入（Task 3）；这里 handler 仅占位，真实调用走 tool_executor 分支
    handler=lambda args, **kw: send_attraction_card(attractions=args.get("attractions")),
    check_fn=lambda: True,
    emoji="💳",
)
