"""
AIWriteX Tools for Hermes Agent

提供 Hermes Agent 可调用的 AIWriteX 工具函数。
这些工具封装了 AIWriteX 的核心功能，供 Agent 在对话中使用。
"""

import json
import logging
from typing import Optional, List
from pathlib import Path

# 导入适配器
from hermes_agent.optional_skills.aiwrite_x import adapter

logger = logging.getLogger(__name__)


def aiwrite_generate_article(
    topic: str,
    platform: str = "wechat",
    urls: Optional[List[str]] = None,
    reference_ratio: float = 0.0,
    min_len: int = 1000,
    max_len: int = 2000,
    use_template: bool = False,
) -> str:
    """
    生成多平台文章。

    使用 AIWriteX 的多智能体协作系统生成高质量文章。
    支持微信公众号、小红书、百家号、知乎、豆瓣等平台。

    Args:
        topic: 文章主题（必填）
        platform: 目标平台，默认 "wechat"（微信公众号）
            可选: "wechat", "xiaohongshu", "baijiahao", "zhihu", "douban"
        urls: 参考文章 URL 列表，用于基于参考资料创作
        reference_ratio: 参考比例，默认 0.0（纯原创）
            范围: 0.0-1.0，0.3 表示 30% 参考，70% 原创
        min_len: 最小字数，默认 1000
        max_len: 最大字数，默认 2000
        use_template: 是否使用 HTML 模板排版，默认 False

    Returns:
        JSON 字符串: {
            "success": bool,
            "title": str,
            "content": str,
            "save_path": str,
            "publish_result": dict,
            "message": str
        }

    Example:
        >>> result = aiwrite_generate_article(
        ...     topic="人工智能的最新发展",
        ...     platform="wechat",
        ...     min_len=1500,
        ...     max_len=2500
        ... )
        >>> print(result)
        {"success": true, "title": "...", "content": "...", ...}
    """
    result = adapter.generate_article(
        topic=topic,
        platform=platform,
        urls=urls,
        reference_ratio=reference_ratio,
        min_len=min_len,
        max_len=max_len,
        use_template=use_template,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


def aiwrite_list_platforms() -> str:
    """
    获取 AIWriteX 支持的平台列表。

    Returns:
        JSON 字符串: 支持的平台名称列表

    Example:
        >>> platforms = aiwrite_list_platforms()
        >>> print(platforms)
        ["wechat", "xiaohongshu", "baijiahao", "zhihu", "douban"]
    """
    platforms = adapter.get_supported_platforms()
    return json.dumps(platforms, ensure_ascii=False, indent=2)


def aiwrite_check_dependencies() -> str:
    """
    检查 AIWriteX 依赖是否满足。

    检查 AIWriteX 模块是否可导入、依赖是否完整。

    Returns:
        JSON 字符串: {"ok": bool, "issues": list}

    Example:
        >>> result = aiwrite_check_dependencies()
        >>> print(result)
        {"ok": true, "issues": []}
    """
    result = adapter.check_dependencies()
    return json.dumps(result, ensure_ascii=False, indent=2)


# 工具注册信息（供 Hermes Agent 使用）
TOOL_INFO = {
    "name": "aiwrite_x",
    "description": "AIWriteX 多平台内容创作工具集",
    "tools": [
        {
            "name": "aiwrite_generate_article",
            "description": "生成多平台文章（微信公众号、小红书、百家号等）",
            "parameters": {
                "topic": {"type": "string", "required": True, "description": "文章主题"},
                "platform": {"type": "string", "default": "wechat", "description": "目标平台"},
                "urls": {"type": "array", "default": [], "description": "参考文章 URL 列表"},
                "reference_ratio": {"type": "number", "default": 0.0, "description": "参考比例 0.0-1.0"},
                "min_len": {"type": "integer", "default": 1000, "description": "最小字数"},
                "max_len": {"type": "integer", "default": 2000, "description": "最大字数"},
                "use_template": {"type": "boolean", "default": False, "description": "是否使用模板"},
            },
        },
        {
            "name": "aiwrite_list_platforms",
            "description": "获取支持的平台列表",
            "parameters": {},
        },
        {
            "name": "aiwrite_check_dependencies",
            "description": "检查 AIWriteX 依赖是否满足",
            "parameters": {},
        },
    ],
}
