"""
NarratoAI Tools for Hermes Agent

提供 Hermes Agent 可调用的 NarratoAI 工具函数。
这些工具封装了 NarratoAI 的核心功能，供 Agent 在对话中使用。
"""

import json
import logging
from typing import Optional
from pathlib import Path

# 导入适配器
from hermes_agent.optional_skills.narrato_ai import adapter

logger = logging.getLogger(__name__)


def narrato_generate_video(
    video_path: str,
    script_json: str,
    voice_name: str = "zh-CN-YunjianNeural",
    video_aspect: str = "9:16",
    tts_engine: str = "edge_tts",
    bgm_name: str = "random",
    output_dir: Optional[str] = None,
    n_threads: int = 8,
    subtitle_enabled: bool = True,
) -> str:
    """
    生成影视解说视频。

    使用 NarratoAI 将原视频和解说脚本合成为完整的影视解说视频。
    支持自动 TTS 配音、字幕生成、背景音乐混音。

    Args:
        video_path: 原视频文件路径（支持 mp4, mkv, avi 等格式）
        script_json: 解说脚本 JSON 文件路径或 JSON 字符串
            格式: [{"narration": "解说词", "OST": 0, "timestamp": "00:00:05,000 --> 00:00:15,000"}]
            OST: 0=仅解说, 1=仅原声, 2=解说+原声
        voice_name: TTS 语音名称，默认 "zh-CN-YunjianNeural"（男声）
            可选: zh-CN-XiaoxiaoNeural（女声）, en-US-JennyNeural（英文）等
        video_aspect: 视频比例，默认 "9:16"（竖屏）
            可选: "9:16", "16:9", "1:1", "3:4", "4:3"
        tts_engine: TTS 引擎，默认 "edge_tts"（免费）
            可选: "edge_tts", "azure", "doubao"（需要 API key）
        bgm_name: 背景音乐，默认 "random"（随机）
        output_dir: 输出目录，默认 NarratoAI/storage/tasks/
        n_threads: 处理线程数，默认 8
        subtitle_enabled: 是否启用字幕，默认 True

    Returns:
        JSON 字符串: {"success": bool, "task_id": str, "output_path": str, "message": str}

    Example:
        >>> result = narrato_generate_video(
        ...     video_path="/path/to/movie.mp4",
        ...     script_json="/path/to/script.json",
        ...     voice_name="zh-CN-XiaoxiaoNeural",
        ...     video_aspect="9:16"
        ... )
        >>> print(result)
        {"success": true, "task_id": "abc123", "output_path": "...", "message": "..."}
    """
    result = adapter.generate_narration_video(
        video_path=video_path,
        script_json=script_json,
        voice_name=voice_name,
        video_aspect=video_aspect,
        tts_engine=tts_engine,
        bgm_name=bgm_name,
        output_dir=output_dir,
        n_threads=n_threads,
        subtitle_enabled=subtitle_enabled,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


def narrato_generate_tts(
    text: str,
    voice_name: str = "zh-CN-YunjianNeural",
    tts_engine: str = "edge_tts",
    voice_rate: float = 1.0,
    output_path: Optional[str] = None,
) -> str:
    """
    生成 TTS 配音音频。

    使用 NarratoAI 的 TTS 功能将文本转换为语音音频。

    Args:
        text: 要转换的文本内容
        voice_name: 语音名称，默认 "zh-CN-YunjianNeural"
        tts_engine: TTS 引擎，默认 "edge_tts"
        voice_rate: 语速，默认 1.0（0.5-2.0）
        output_path: 输出音频路径，默认自动生成

    Returns:
        JSON 字符串: {"success": bool, "audio_path": str, "message": str}

    Example:
        >>> result = narrato_generate_tts(
        ...     text="大家好，欢迎观看本期视频",
        ...     voice_name="zh-CN-XiaoxiaoNeural"
        ... )
        >>> print(result)
        {"success": true, "audio_path": "/tmp/narrato_tts_xxx.mp3", "message": "..."}
    """
    result = adapter.generate_tts_audio(
        text=text,
        voice_name=voice_name,
        tts_engine=tts_engine,
        voice_rate=voice_rate,
        output_path=output_path,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


def narrato_list_voices(language: str = "zh") -> str:
    """
    获取可用的 TTS 语音列表。

    Args:
        language: 语言过滤，默认 "zh"（中文）
            可选: "zh", "en", "ja" 等

    Returns:
        JSON 字符串: 可用语音名称列表

    Example:
        >>> voices = narrato_list_voices("zh")
        >>> print(voices)
        ["zh-CN-XiaoxiaoNeural", "zh-CN-YunjianNeural", ...]
    """
    voices = adapter.get_available_voices(language)
    return json.dumps(voices, ensure_ascii=False, indent=2)


def narrato_check_dependencies() -> str:
    """
    检查 NarratoAI 依赖是否满足。

    检查 FFmpeg 是否安装、NarratoAI 模块是否可导入。

    Returns:
        JSON 字符串: {"ok": bool, "issues": list}

    Example:
        >>> result = narrato_check_dependencies()
        >>> print(result)
        {"ok": true, "issues": []}
    """
    result = adapter.check_dependencies()
    return json.dumps(result, ensure_ascii=False, indent=2)


# 工具注册信息（供 Hermes Agent 使用）
TOOL_INFO = {
    "name": "narrato_ai",
    "description": "NarratoAI 影视解说视频生成工具集",
    "tools": [
        {
            "name": "narrato_generate_video",
            "description": "生成影视解说视频（需要原视频和解说脚本）",
            "parameters": {
                "video_path": {"type": "string", "required": True},
                "script_json": {"type": "string", "required": True},
                "voice_name": {"type": "string", "default": "zh-CN-YunjianNeural"},
                "video_aspect": {"type": "string", "default": "9:16"},
                "tts_engine": {"type": "string", "default": "edge_tts"},
            },
        },
        {
            "name": "narrato_generate_tts",
            "description": "生成 TTS 配音音频",
            "parameters": {
                "text": {"type": "string", "required": True},
                "voice_name": {"type": "string", "default": "zh-CN-YunjianNeural"},
                "tts_engine": {"type": "string", "default": "edge_tts"},
            },
        },
        {
            "name": "narrato_list_voices",
            "description": "获取可用的 TTS 语音列表",
            "parameters": {
                "language": {"type": "string", "default": "zh"},
            },
        },
        {
            "name": "narrato_check_dependencies",
            "description": "检查 NarratoAI 依赖是否满足",
            "parameters": {},
        },
    ],
}
