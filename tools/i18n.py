#!/usr/bin/env python3
"""i18n - Internationalization support for Hermes Agent.

Provides language detection from config and translation functions for
localizing user-facing strings.
"""

import os
from typing import Optional

# Global language state
_current_language: str = ""

# Chinese translation dictionaries
_UI_ZH: dict[str, str] = {}

def _detect_language() -> str:
    """Detect the configured language from config.yaml.
    
    Checks approvals.language first, then falls back to display.language.
    Returns lowercase language code (e.g., 'zh', 'en').
    """
    global _current_language
    if _current_language:
        return _current_language
    
    try:
        from hermes_cli.config import load_config
        config = load_config()
        
        # Check approvals.language first (existing convention)
        approvals = config.get("approvals") or {}
        if isinstance(approvals, dict):
            lang = approvals.get("language", "")
            if lang:
                _current_language = lang.lower()
                return _current_language
        
        # Also check display.language as fallback
        display = config.get("display") or {}
        if isinstance(display, dict):
            lang = display.get("language", "")
            if lang:
                _current_language = lang.lower()
                return _current_language
    except Exception:
        pass
    
    _current_language = ""
    return ""

def _set_language(lang: str) -> None:
    """Manually set the language (for testing or external configuration)."""
    global _current_language, _UI_ZH
    _current_language = lang or ""
    if _current_language == "zh":
        _load_builtin_zh_translations()

def _load_builtin_zh_translations() -> None:
    """Load built-in Chinese translations for UI strings."""
    global _UI_ZH
    _UI_ZH = {
        # CLI output - compression
        "Compressing {count} messages (~{tokens:,} tokens)...": "正在压缩 {count} 条消息（约 {tokens:,} tokens）...",
        "Compressing {count} messages (~{tokens:,} tokens), focus: \"{topic}\"...": "正在压缩 {count} 条消息（约 {tokens:,} tokens），主题：\"{topic}\"...",
        "No changes from compression: {count} messages": "压缩未改变内容：{count} 条消息",
        "Rough transcript estimate: ~{from} tokens (unchanged)": "粗略估算：~{from} tokens（未改变）",
        "Rough transcript estimate: ~{from} → ~{to} tokens": "粗略估算：~{from} → ~{to} tokens",
        "Compressed: {from} → {to} messages": "已压缩：{from} → {to} 条消息",
        "Note: fewer messages can still raise this rough transcript estimate when compression rewrites the transcript into denser summaries.": "注意：消息数减少时，如果压缩将转录重写为更紧凑的摘要，粗略估算仍可能增加。",
        
        # CLI output - status
        "Cache write tokens:": "缓存写入 tokens:",
        "Output tokens:": "输出 tokens:",
        "Prompt tokens (total):": "提示 tokens（总计）:",
        "Completion tokens:": "补全 tokens:",
        "Total tokens:": "总 tokens:",
        "API calls:": "API 调用:",
        "Session duration:": "会话时长:",
        "Cost status:": "成本状态:",
        "Cost source:": "成本来源:",
        "Total cost:": "总成本:",
        "Current context:": "当前上下文:",
        "Messages:": "消息数:",
        "Compressions:": "压缩次数:",
        "Pricing unknown for {model}": "{model} 定价未知",
        "Note:": "注意：",
        
        # CLI output - MCP
        "Reconnected:": "已重连：",
        "Added:": "已添加：",
        "Removed:": "已移除：",
        "tool(s) available from {servers} server(s)": "个工具来自 {servers} 个服务器",
        "Agent updated — {count} tool(s) available": "助手已更新 — {count} 个工具可用",
        "MCP reload failed:": "MCP 重新加载失败：",
        
        # CLI output - voice
        "Silence detected, auto-stopping...": "检测到静音，自动停止...",
        "Recording...": "正在录音...",
        "No speech detected.": "未检测到语音。",
        "Transcribing...": "正在转录...",
        "Transcription failed:": "转录失败：",
        "Voice processing error:": "语音处理错误：",
        "No speech detected 3 times, continuous mode stopped.": "连续 3 次未检测到语音，连续模式已停止。",
        "Voice auto-restart failed:": "语音自动重启失败：",
        "TTS playback failed:": "TTS 播放失败：",
        "Unknown voice subcommand:": "未知的语音子命令：",
        "Voice mode is already enabled.": "语音模式已启用。",
        "Voice mode unavailable in this environment:": "此环境中不可用语音模式：",
        "Voice mode requirements not met:": "未满足语音模式要求：",
        "Voice mode enabled": "语音模式已启用",
        "Voice mode disabled.": "语音模式已禁用。",
        "Enable voice mode first: /voice on": "请先启用语音模式：/voice on",
        "Warning: No TTS provider available. Install edge-tts or set API keys.": "警告：没有可用的 TTS 提供商。请安装 edge-tts 或设置 API 密钥。",
        "Voice TTS {status}.": "Voice TTS {status}。",
        "Voice Mode Status": "语音模式状态",
        "Mode:": "模式：",
        "TTS:": "TTS：",
        "Recording:": "录音：",
        "Record key:": "录音键：",
        "Requirements:": "要求：",
        "Continuous voice mode stopped due to error.": "连续语音模式因错误而停止。",
        
        # CLI output - session
        "Session not found:": "未找到会话：",
        "Use a session ID from a previous CLI run (hermes sessions list).": "请使用之前 CLI 运行的会话 ID（hermes sessions list）。",
        "Session title applied:": "已应用会话标题：",
        "Could not apply pending title:": "无法应用待处理的标题：",
        "Failed to initialize agent:": "初始化助手失败：",
        "No checkpoints found for {cwd}": "未找到 {cwd} 的检查点",
        "Invalid checkpoint number. Use 1-{max}.": "无效的检查点编号。请使用 1-{max}。",
        "Restored {file_path} from checkpoint {checkpoint}: {reason}": "已从检查点 {checkpoint} 恢复 {file_path}：{reason}",
        "Restored to checkpoint {checkpoint}: {reason}": "已恢复到检查点 {checkpoint}：{reason}",
        "State snapshots ({path}/state-snapshots/):": "状态快照（{path}/state-snapshots/）：",
        "Snapshot created:": "已创建快照：",
        "Most recent:": "最新：",
        "Invalid snapshot number. Use 1-{max}.": "无效的快照编号。请使用 1-{max}。",
        
        # CLI output - general
        "Initializing agent...": "正在初始化助手...",
        "Warning:": "警告：",
        "Queued for the next turn:": "已排入下一轮：",
        "Recording cancelled.": "录音已取消。",
        "Suspend (Ctrl+Z) is not supported on Windows.": "Windows 不支持挂起（Ctrl+Z）。",
        "Voice recording failed:": "语音录音失败：",
        "Auto-attached image:": "自动附加图片：",
        "Detected file:": "检测到文件：",
        "image{'s' if n > 1 else ''} attached": "已附加 {n} 张图片",
        "Error:": "错误：",
        "hermes --resume {session_id}": "hermes --resume {session_id}",
        "hermes -c \"{title}\"": "hermes -c \"{title}\"",
        "Session:": "会话：",
        "Title:": "标题：",
        "Duration:": "时长：",
        "Messages:": "消息数：",
        "user, {calls} tool calls": "用户，{calls} 次工具调用",
        
        # Spinner verbs
        "pondering": "思考中",
        "contemplating": "琢磨中",
        "musing": "沉吟中",
        "calculating": "计算中",
        "researching": "研究中",
        "planning": "规划中",
        "analyzing": "分析中",
        "processing": "处理中",
        "considering": "考虑中",
        "evaluating": "评估中",
        
        # Tips
        "/btw <question> asks a quick side question without tools or history...": "/btw <问题> 可以快速提问而不使用工具或历史记录 — 适合澄清疑问。",
        "/background <prompt> runs a task in a separate session while keeping the current one available.": "/background <提示> 在独立会话中运行任务，同时保持当前会话可用。",
        
        # Approval messages
        "DANGEROUS COMMAND:": "⚠️ 危险命令：",
        "[o]nce": "[o] 仅此次",
        "[s]ession": "[s] 本次会话",
        "[a]lways": "[a] 始终允许",
        "[d]deny": "[d] 拒绝",
        "Always allow?": "是否始终允许？",
        "Allow this session?": "是否允许本次会话？",
        "Allow once?": "是否仅允许此次？",
        "Deny?": "拒绝？",
        
        # General UI
        "Welcome to Hermes Agent": "欢迎使用 Hermes Agent",
        "Type a message or /help for commands.": "输入消息或 /help 查看命令。",
        "Goodbye!": "再见！",
        "Hermes": "Hermes",
        "Agent": "助手",
        "Available Tools": "可用工具",
        "Available Skills": "可用技能",
        "MCP Servers": "MCP 服务器",
        "and more toolsets": "更多工具集",
        "and more": "更多",
        "tools": "个工具",
        "skills": "个技能",
        "server_count": "个服务器",
        "help_for_commands": "输入 /help 查看命令",
        "profile": "配置 profile",
        "commit_behind": "落后",
        "commits_behind": "个提交",
        "run_to_update": "运行以更新",
        "no_skills_installed": "未安装技能",
        "connected": "已连接",
        "failed": "失败",
        "tool_count": "个工具",
        "skill_count": "个技能",
    }

def _ui_zh(text: str) -> str:
    """Translate a UI string to Chinese if language is 'zh'.
    
    Args:
        text: The English string to translate.
        
    Returns:
        The Chinese translation if available, otherwise the original string.
    """
    if _current_language == "zh":
        return _UI_ZH.get(text, text)
    return text

def ui(text: str) -> str:
    """Convenience function for translating UI strings.
    
    Usage:
        print(ui("Compressing..."))  # → "正在压缩..." if language is zh
    """
    return _ui_zh(text)

def format_zh(template: str, **kwargs) -> str:
    """Format a template string with Chinese localization.
    
    Usage:
        format_zh("Compressing {count} messages...", count=100)
    """
    if _current_language == "zh":
        # Try to find a matching Chinese template
        for en_template, zh_template in _UI_ZH.items():
            if en_template == template:
                return zh_template.format(**kwargs)
    return template.format(**kwargs)

# Initialize language on module load
_detect_language()
