#!/usr/bin/env python3
"""i18n - Internationalization support for Hermes Agent.

Provides language detection from config and translation functions for
localizing user-facing strings.
"""

import os
from typing import Optional


def get_config_language() -> str:
    """Get the current language setting from config.
    
    Returns:
        Language code: 'zh' for Chinese, 'en' for English (default)
    """
    # Try to read from config without importing the full config module
    config_path = os.path.expanduser("~/.hermes/config.yaml")
    if not os.path.exists(config_path):
        return "en"
    
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        
        # Check approvals.language first, then display.language
        lang = config.get("approvals", {}).get("language")
        if not lang:
            lang = config.get("display", {}).get("language")
        if not lang:
            lang = config.get("language")
        
        return lang if lang in ("zh", "en") else "en"
    except Exception:
        return "en"


def is_chinese() -> bool:
    """Check if current language is Chinese."""
    return get_config_language() == "zh"


def format_zh(text: str, **kwargs) -> str:
    """Format text with Chinese translation if language is set to Chinese.
    
    Args:
        text: English text to translate
        **kwargs: Format arguments for f-string style formatting
        
    Returns:
        Translated text (Chinese if config language is 'zh', otherwise English)
        
    Example:
        >>> format_zh("Compressing {count} messages...", count=10)
        "正在压缩 10 条消息..."  # if language is zh
        "Compressing 10 messages..."  # if language is en
    """
    # Full-message translations (checked first)
    full_translations = {
        '📬 No home channel is set for {name}. A home channel is where Hermes delivers cron job results and cross-platform messages.\n\nType /sethome to make this chat your home channel, or ignore to skip.':
            '📬 {name} 未设置主频道。\n\n主频道是 Hermes 投递定时任务结果和跨平台消息的地方。\n\n输入 /sethome 将此聊天设为主频道，\n或忽略以跳过。',

        '⏳ Still working... ({elapsed} min elapsed{detail})':
            '⏳ 仍在工作中...（已运行 {elapsed} 分钟{detail})',

        'Sending after interrupt: \'{preview}\'':
            '中断后发送：\'{preview}\'',
    }

    translations = {
        # Compression feedback
        "Compressing": "正在压缩",
        "Compressed:": "已压缩:",
        "Rough transcript estimate:": "粗略转录估算:",
        
        # Session token usage
        "Session Token Usage": "会话 Token 使用",
        "Model:": "模型:",
        "Input tokens:": "输入 tokens:",
        "Output tokens:": "输出 tokens:",
        "Total tokens:": "总 tokens:",
        "Session messages:": "会话消息:",
        "Session context:": "会话上下文:",
        
        # Session list
        "Session:": "会话:",
        "Title:": "标题:",
        "Duration:": "时长:",
        "Messages:": "消息:",
        
        # Voice mode
        "Voice mode enabled": "语音模式已启用",
        "Voice mode disabled.": "语音模式已禁用。",
        "Voice mode is already enabled.": "语音模式已启用。",
        "Voice mode unavailable in this environment:": "语音模式在此环境中不可用：",
        "Voice mode requirements not met:": "语音模式要求未满足：",
        "Voice TTS": "语音 TTS",
        "Voice Mode Status": "语音模式状态",
        "Mode:": "模式:",
        "TTS:": "TTS:",
        "Recording:": "录音:",
        "Record key:": "录音键:",
        "Requirements:": "要求:",
        
        # Clarify timeout
        "clarify timed out after": "澄清超时，超时时间：",
        " — agent will decide)": "— 将由代理自行决定",
        
        # Approval timeout
        "Timeout — denying command": "超时 — 拒绝命令",
        
        # Approval choices
        "once": "仅一次",
        "session": "会话",
        "always": "始终",
        "deny": "拒绝",
        "view": "查看",
        "Allow once": "允许仅一次",
        "Allow for this session": "允许本次会话",
        "Add to permanent allowlist": "添加到永久白名单",
        "Deny": "拒绝",
        
        # Dangerous Command UI
        "Dangerous Command": "危险命令",
        "Show full command": "显示完整命令",
        
        # MCP reload
        "Reloading MCP servers...": "正在重新加载 MCP 服务器...",
        
        # Snapshot
        "No state snapshots yet.": "暂无状态快照。",
        "Create one:": "创建快照:",
        
        # Session not found
        "Session not found:": "会话未找到:",
        
        # Voice TTS status
        "Voice TTS enabled.": "语音 TTS 已启用。",
        "Voice TTS disabled.": "语音 TTS 已禁用。",
        
        # General
        "messages": "消息",
        "messages...": "消息...",

        # Gateway notifications
        "Still working...": "仍在工作中...",
        "min elapsed": "分钟已过",
        "iteration": "迭代",
        "running:": "运行中:",
        "waiting for non-streaming API response": "等待非流式 API 响应",
        "waiting for provider response (streaming)": "等待供应商响应（流式）",
        "waiting for stream response": "等待流式响应",
        "s, no chunks yet": "秒，尚无数据块",
        "starting new turn (cached)": "开始新轮次（缓存）",
        "initializing": "初始化中",

        # Home channel
        "No home channel is set for": "{0} 未设置主频道",
        "A home channel is where Hermes delivers cron job results": "主频道是 Hermes 投递定时任务结果的位置",
        "Type /sethome to make this chat your home channel,": "输入 /sethome 将此聊天设为主频道，",
        "or ignore to skip.": "或忽略以跳过。",

        # Banner
        "Session:": "会话:",

        # Interrupt messages
        "New message detected, interrupting...": "检测到新消息，正在中断...",
        "Sending after interrupt:": "中断后发送：",
        "Interrupting agent... (press Ctrl+C again to force exit)": "正在中断 agent...（再按 Ctrl+C 强制退出）",
        "Interrupted during API call.": "API 调用过程中被中断",
        "Interrupt requested": "请求中断",
        "Force exiting...": "强制退出...",
        "Interrupt: skipping": "中断：跳过",
        "Breaking out of tool loop due to interrupt...": "因中断跳出工具循环...",
        "Interrupt detected during retry wait, aborting.": "重试等待阶段检测到中断，已中止",
        "Interrupt detected during error handling, aborting retries.": "错误处理阶段检测到中断，已中止重试",
        "Suspend (Ctrl+Z) is not supported on Windows.": "Windows 系统不支持挂起操作（Ctrl+Z）",
        "Starting Hermes Gateway (messaging platforms)...": "正在启动 Hermes 网关（消息平台）...",
        "Starting conversation:": "开始对话：",
        "Error generating insights:": "生成洞察时出错：",
        "Error loading gateway config:": "加载网关配置时出错：",
        "Warning: Unknown toolsets:": "警告：未知的工具集：",
        "Warning: No TTS provider available. Install edge-tts or set API keys.": "警告：无可用的 TTS 服务提供商。安装 edge-tts 或设置 API 密钥。",
        "Loading skill:": "正在加载技能：",
        "Stopping": "正在停止",
        "background process(es)...": "后台进程...",
        
        # Tool status
        "preparing": "正在准备",
        "Session reset. New tool configuration is active.": "会话已重置。新工具配置已生效。",
        "Steer queued — arrives after the next tool call: ": "已排队引导 — 将在下一次工具调用后到达：",
        
        # Session management
        "Session not found: ": "会话未找到：",
        "Use a session ID from a previous CLI run (hermes sessions list).": "请使用之前 CLI 运行的会话 ID（hermes sessions list）。",
        "Session title applied: ": "会话标题已应用：",
        "Could not apply pending title: ": "无法应用待处理的标题：",
        "Title rejected: ": "标题被拒绝：",
        " — session started untitled.": " — 会话以无标题开始。",
        "Title is empty after cleanup — session started untitled.": "清理后标题为空 — 会话以无标题开始。",
        "Usage: /resume <session_id_or_title>": "用法：/resume <会话 ID 或标题>",
        "Tip:   Use /history or `hermes sessions list` to find sessions.": "提示：使用 /history 或 `hermes sessions list` 查找会话。",
        "Session database not available.": "会话数据库不可用。",
        "Use /history or `hermes sessions list` to see available sessions.": "使用 /history 或 `hermes sessions list` 查看可用会话。",
        "Already on that session.": "已在该会话中。",
        "↻ Resumed session {target_id}{title_part} — no messages, starting fresh.": "↻ 已恢复会话 {target_id}{title_part} — 无消息，重新开始。",
        "No conversation to branch — send a message first.": "无可分支的对话 — 请先发送一条消息。",
        "Failed to create branch session: ": "创建分支会话失败：",
        "Original session: ": "原始会话：",
        "Branch session:   ": "分支会话：   ",
        "⚠ Agent swap failed ({exc}); change applied to next session.": "⚠ Agent 切换失败 ({exc})；更改将应用于下一个会话。",
        "(session only — add --global to persist)": "（仅当前会话 — 添加 --global 以持久化）",
        "Title is empty after cleanup. Please use printable characters.": "清理后标题为空。请使用可打印字符。",
        "Session title set: ": "会话标题已设置：",
        "Session not found in database.": "在数据库中未找到会话。",
        " is already in use by session ": " 已被会话 ",
        "Session title queued: ": "会话标题已排队：",
        "Usage: /title <your session title>": "用法：/title <你的会话标题>",
        "Session ID: ": "会话 ID：",
        "Title: ": "标题：",
        "Title (pending): ": "标题（待处理）：",
        "No title set. Usage: /title <your session title>": "未设置标题。用法：/title <你的会话标题>",
        "The task runs in a separate session and results display here when done.": "任务在独立会话中运行，完成后结果将在此显示。",
        "Goals unavailable (no active session).": "目标不可用（无活动会话）。",
        "No goal to resume.": "无可恢复的目标。",
        "▶ Goal resumed: ": "▶ 目标已恢复：",
        "(session only)": "（仅当前会话）",
        "✓ {feature_name} set to {label} (session only)": "✓ {feature_name} 已设置为 {label}（仅当前会话）",
        "✓ Password received (cached for session)": "✓ 密码已接收（已缓存用于会话）",
        
        # Error messages
        "Failed to open external editor: ": "打开外部编辑器失败：",
        "Primary auth failed — switching to fallback: ": "主认证失败 — 切换到备用：",
        "(>_<) Clipboard has an image but extraction failed": "(>_<) 剪贴板有图片但提取失败",
        "Invalid response number. Use 1-{len(assistant)}.": "无效的回答编号。请使用 1-{len(assistant)}。",
        "Clipboard copy failed: ": "剪贴板复制失败：",
        "(>_<) File not found: ": "(>_<) 文件未找到：",
        "⚠ vision analysis failed — path included for retry": "⚠ 视觉分析失败 — 路径已包含以便重试",
        "⚠ vision analysis error — path included for retry": "⚠ 视觉分析错误 — 路径已包含以便重试",
        "Steer failed: ": "引导失败：",
        "Plugin command error: ": "插件命令错误：",
        "❌ Background task #{task_num} failed: ": "❌ 后台任务 #{task_num} 失败：",
        "Invalid goal: ": "无效目标：",
        "Transcription failed: ": "转录失败：",
        "Voice processing error: ": "语音处理错误：",
        "Voice auto-restart failed: ": "语音自动重启失败：",
        "TTS playback failed: ": "TTS 播放失败：",
        "Continuous voice mode stopped due to error.": "连续语音模式因错误而停止。",
        "Steer failed ({exc}) — queued for next turn.": "引导失败 ({exc}) — 已排队等待下一轮。",
        "Voice recording failed: ": "语音录制失败：",
        "Recording cancelled.": "录制已取消。",
        "No speech detected.": "未检测到语音。",
        "No speech detected 3 times, continuous mode stopped.": "未检测到语音 3 次，连续模式已停止。",
        "Unknown voice subcommand: ": "未知的语音子命令：",
        "Voice mode is already enabled.": "语音模式已启用。",
        "Voice mode unavailable in this environment:": "语音模式在此环境中不可用：",
        "Voice mode requirements not met:": "语音模式要求未满足：",
        "Voice mode enabled": "语音模式已启用",
        "{_ptt_display} to start/stop recording": "按 {_ptt_display} 开始/停止录制",
        "/voice tts  to toggle speech output": "/voice tts 切换语音输出",
        "/voice off  to disable voice mode": "/voice off 禁用语音模式",
        "Voice mode disabled.": "语音模式已禁用。",
        "Enable voice mode first: /voice on": "请先启用语音模式：/voice on",
        "Warning: No TTS provider available. Install edge-tts or set API keys.": "警告：无可用的 TTS 服务提供商。安装 edge-tts 或设置 API 密钥。",
        "Voice TTS {status}.": "语音 TTS {status}。",
        "Voice Mode Status": "语音模式状态",
        "Mode:      ": "模式：      ",
        "TTS:       ": "TTS：       ",
        "Recording: ": "录音：",
        "Record key: ": "录音键：",
        "Requirements:": "要求：",
        "Silence detected, auto-stopping...": "检测到静音，自动停止...",
        "Transcribing...": "正在转录...",
        "(clarify timed out after {timeout}s — agent will decide)": "(澄清超时 {timeout} 秒 — 代理将自行决定)",
        "⏱ Timeout — continuing without sudo": "⏱ 超时 — 不使用 sudo 继续",
        "⏱ Timeout — denying command": "⏱ 超时 — 拒绝命令",
        "⏭ Skipped": "⏭ 已跳过",
        "Initializing agent...": "正在初始化 agent...",
        "⚠ {w}": "⚠ {w}",
        "👁️  analyzing {img_path.name} ({size_kb}KB)...": "👁️  正在分析 {img_path.name} ({size_kb}KB)...",
        "✓ image analyzed": "✓ 图片已分析",
        "Nothing to copy yet.": "尚无内容可复制。",
        "Nothing to copy in assistant responses yet.": "助手回复中尚无内容可复制。",
        "Nothing to copy in that assistant response.": "该助手回复中尚无内容可复制。",
        "Copied assistant response #{idx + 1} to clipboard": "已将助手回复 #{idx + 1} 复制到剪贴板",
        "(._.) No image found in clipboard": "(._.) 剪贴板中未找到图片",
        "(._.) Not a supported image file: ": "(._.) 不支持的图片文件：",
        "📎 Attached image: ": "📎 已附加图片：",
        "Now type your prompt (or use --image in single-query mode): ": "现在输入你的提示（或在单查询模式中使用 --image）：",
        "📎 Image #{n} attached from clipboard": "📎 图片 #{n} 已从剪贴板附加",
        "📎 Auto-attached image: ": "📎 自动附加图片：",
        "📄 Detected file: ": "📄 检测到文件：",
        "📎 {n} image": "📎 {n} 张图片",
        " attached": " 已附加",
        "⚠ tirith security scanner enabled but not available ": "⚠ tirith 安全扫描器已启用但不可用 ",
        "⏩ Steered: ": "⏩ 已引导：",
        "⚡ Interrupting current task": "⚡ 正在中断当前任务",
        "ll respond to your message shortly.": "将很快回复你的消息。",
        "⚠️ Could not save config: {e}": "⚠️ 无法保存配置：{e}",
        "⚙️ Tool progress: **OFF** — no tool activity shown.": "⚙️ 工具进度：**关闭** — 不显示工具活动。",
        "⚙️ Tool progress: **VERBOSE** — every tool call with full arguments.": "⚙️ 工具进度：**详细** — 显示每次工具调用的完整参数。",
        "⚠️ Provider authentication failed: {exc}": "⚠️ 供应商认证失败：{exc}",
        "⚠️ Proxy error ({resp.status}): {error_text[:300]}": "⚠️ 代理错误 ({resp.status})：{error_text[:300]}",
        "⚠️ Proxy connection error: {e}": "⚠️ 代理连接错误：{e}",
        "⚠️ No activity for {_elapsed_warn} min. ": "⚠️ 无活动 {_elapsed_warn} 分钟。",
        "If the agent does not respond soon, it will ": "如果代理不久后无响应，它将 ",
        "⏱️ Agent inactive for {_timeout_mins} min — no tool calls ": "⏱️ Agent 无活动 {_timeout_mins} 分钟 — 无工具调用 ",
        "⚠️ No tools selected (all filtered out or unavailable)": "⚠️ 未选择任何工具（全部被过滤或不可用）",
        "⚠️ Unknown toolset: {toolset_name}": "⚠️ 未知的工具集：{toolset_name}",
        "⚠️ Invalid JSON in tool call arguments for ": "⚠️ 工具调用参数中的无效 JSON：",
        "⚠️ Truncated tool call arguments detected ": "⚠️ 检测到截断的工具调用参数 ",
        "⚠️ Unknown tool ": "⚠️ 未知工具 ",
        "⚠️ Injecting recovery tool results for invalid JSON...": "⚠️ 正在注入无效 JSON 的恢复工具结果...",
        "⚠️ Model returned empty after tool calls — ": "⚠️ 模型在工具调用后返回空 — ",
        "⚠️ Empty response from model — retrying ": "⚠️ 模型返回空响应 — 重试 ",
        "⚠️ Model returning empty responses — ": "⚠️ 模型返回空响应 — ",
        "⚠️ Model produced reasoning but no visible ": "⚠️ 模型生成了推理但无可见 ",
        "⚠️ Iteration budget exhausted ({api_call_count}/{self.max_iterations}) ": "⚠️ 迭代预算已耗尽 ({api_call_count}/{self.max_iterations}) ",
        "⚠️ Gateway is {self._status_action_gerund()} and is not accepting another turn right now.": "⚠️ 网关正在 {self._status_action_gerund()}，当前不接受新的轮次。",
        "⚠️ Gateway {action} — {hint}": "⚠️ 网关 {action} — {hint}",
        
        # General
        "error": "错误",
        "content": "内容",
        "unknown": "未知",
        "system": "系统",
        "timestamp": "时间戳",
        "session": "会话",
        "queue": "排队",
        "steer": "引导",
        "interrupt": "中断",
        "success": "成功",
        "failed": "失败",
        "retrying": "重试中",
        "fatal": "致命",
        "connected": "已连接",
        "connecting": "连接中",
        "draining": "排空中",
        "normal": "正常",
        "priority": "优先",
        "medium": "中等",
        "high": "高",
        "xhigh": "极高",
        "default": "默认",
        "standard": "标准",
       "show_reasoning": "显示推理",
        "background_process_notifications": "后台进程通知",
        "result": "结果",
        
        # CLI help and status
        "Nothing to copy yet.": "尚无内容可复制。",
        "Usage: /copy [number]": "用法：/copy [编号]",
        "Nothing to copy in assistant responses yet.": "助手回复中尚无内容可复制。",
        "Nothing to copy in that assistant response.": "该助手回复中尚无内容可复制。",
        "Prompt caching: enabled": "提示缓存：已启用",
        "Saved to config.yaml (--global)": "已保存到 config.yaml（--global）",
        "No authenticated providers found.": "未找到已认证的供应商。",
        "/model <name>                        switch model": "/model <名称>                        切换模型",
        "/model --provider <slug>             switch provider": "/model --provider <标识>             切换供应商",
        "✨ (◕‿◕)✨ Fresh start! Screen cleared and conversation reset.\n": "✨ (◕‿◕)✨ 全新开始！屏幕已清除，对话已重置。\n",
        "Steer rejected (empty payload).": "引导被拒绝（空负载）。",
        "Usage: /background <prompt>": "用法：/background <提示>",
        "Example: /background Summarize the top HN stories today": "示例：/background 总结今天的 HN 热门故事",
        "(>_<) Cannot start background task: no valid credentials.": "(>_<) 无法启动后台任务：无有效凭据。",
        "You can continue chatting — results will appear when done.\n": "你可以继续聊天 — 完成后结果将在此显示。\n",
        "(No response generated": "（未生成回复",
        "✓ Goal cleared.": "✓ 目标已清除。",
        "Usage: /footer [on|off|status]": "用法：/footer [开|关|状态]",
        "Failed to save runtime_footer setting to config.yaml": "无法保存 runtime_footer 设置到 config.yaml",
        "(._.) No image found in clipboard": "(._.) 剪贴板中未找到图片",
        "Usage: /voice [on|off|tts|status]": "用法：/voice [开|关|tts|状态]",
    }
    
    if not is_chinese():
        # English - just format the text
        if kwargs:
            try:
                return text.format(**kwargs)
            except (KeyError, IndexError, ValueError):
                return text
        return text

    # Chinese - translate and format
    # 1. Check full-message translations first
    if text in full_translations:
        result = full_translations[text]
        if kwargs:
            try:
                return result.format(**kwargs)
            except (KeyError, IndexError, ValueError):
                return result
        return result

    # 2. Fall back to substring replacements for known phrases
    #
    # IMPORTANT: Apply format() FIRST so named placeholders ({name}, {cmd})
    # are resolved to their actual values BEFORE substring replacement runs.
    # Otherwise a Chinese translation value containing {0} (positional
    # placeholder) can corrupt the result — format() with **kwargs then
    # raises IndexError ("Replacement index 0 out of range").
    result = text
    if kwargs:
        try:
            result = result.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            pass

    for en, zh in translations.items():
        if en in result:
            result = result.replace(en, zh)

    return result
