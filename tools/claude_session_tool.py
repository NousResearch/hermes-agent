"""tools/claude_session_tool.py — Hermes tool for Claude Code session management."""

import filecmp
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime
from typing import Optional

from tools.registry import registry, tool_error, tool_result
from tools.claude_session.errors import SessionError

# Module-level import of gateway session context — avoids repeated import on hot path.
try:
    from gateway.session_context import get_session_env
except ImportError:
    get_session_env = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session Registry（支持并行运行多个独立会话 + gateway session 隔离）
# ---------------------------------------------------------------------------
_sessions: dict = {}   # session_id → ClaudeSession 实例
_workdir_index: dict[tuple[str, str], list[str]] = {}  # (gateway_key, workdir) → list[session_id] (一对多，同一 workdir 支持多个会话)
_name_index: dict[tuple[str, str], str] = {}  # (gateway_key, name) → session_id (主索引)
_active_session: dict[str, str] = {}  # gateway_key → session_id (最近活跃)
_sessions_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Session Persistence — claude-session.json in project .claude/ directory
# ---------------------------------------------------------------------------
_CLAUDE_DIR = ".claude"
_SESSION_FILE = "claude-session.json"
_GLOBAL_INDEX_FILE = os.path.expanduser("~/.hermes/claude-session-dirs.json")
_global_index_lock = threading.Lock()


def _register_workdir(workdir: str) -> None:
    """Register a workdir in the global index for cross-session discovery."""
    try:
        with _global_index_lock:
            dirs = set()
            if os.path.isfile(_GLOBAL_INDEX_FILE):
                try:
                    with open(_GLOBAL_INDEX_FILE, "r") as f:
                        dirs = set(json.load(f))
                except Exception:
                    pass
            if workdir in dirs:
                return
            dirs.add(workdir)
            os.makedirs(os.path.dirname(_GLOBAL_INDEX_FILE), exist_ok=True)
            tmp = _GLOBAL_INDEX_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(sorted(dirs), f)
                f.write("\n")
            os.replace(tmp, _GLOBAL_INDEX_FILE)
            try:
                os.chmod(_GLOBAL_INDEX_FILE, 0o600)
            except Exception:
                pass
    except Exception as e:
        logger.debug("Failed to register workdir %s: %s", workdir, e)


def _get_known_workdirs() -> list:
    """Return all workdirs that have ever hosted a persisted session.
    Filters out paths that no longer exist on disk."""
    try:
        if not os.path.isfile(_GLOBAL_INDEX_FILE):
            return []
        with _global_index_lock:
            with open(_GLOBAL_INDEX_FILE, "r") as f:
                dirs = json.load(f)
        return [d for d in dirs if os.path.isdir(d)]
    except Exception:
        return []


def _validate_workdir(workdir: str) -> str:
    """Validate and normalize workdir. Returns realpath. Raises on unsafe paths."""
    # Resolve to absolute, real path (follows symlinks, eliminates ..)
    real = os.path.realpath(workdir)
    # Must exist and be a directory
    if not os.path.isdir(real):
        raise ValueError(f"workdir does not exist or is not a directory: {real}")
    # Block sensitive system directories
    blocked = ("/etc", "/proc", "/sys", "/dev", "/boot", "/root", "/usr", "/var")
    if any(real == b or real.startswith(b + "/") for b in blocked):
        raise ValueError(f"workdir in blocked system directory: {real}")
    return real


def _session_file_path(workdir: str) -> str:
    """Return the path to claude-session.json for a given workdir."""
    return os.path.join(workdir, _CLAUDE_DIR, _SESSION_FILE)


def _load_session_registry(workdir: str) -> dict:
    """Load session registry from .claude/claude-session.json. Returns {} if not found."""
    try:
        workdir = _validate_workdir(workdir)
    except ValueError as e:
        logger.warning("Workdir validation failed for load: %s", e)
        return {}
    path = _session_file_path(workdir)
    if not os.path.isfile(path):
        return {}
    # Verify no symlink tricks on the file itself
    if os.path.islink(path):
        logger.warning("Session file is a symlink, ignoring: %s", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Sanity: top-level must be a dict of session entries
        if not isinstance(data, dict):
            logger.warning("Session file root is not a dict, ignoring: %s", path)
            return {}
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load session registry from %s: %s", path, e)
        return {}


def _save_session_registry(workdir: str, data: dict) -> None:
    """Save session registry to .claude/claude-session.json. Creates .claude/ if needed."""
    try:
        workdir = _validate_workdir(workdir)
    except ValueError as e:
        logger.warning("Workdir validation failed for save: %s", e)
        return
    path = _session_file_path(workdir)
    claude_dir = os.path.dirname(path)
    try:
        # Create .claude dir if needed — check for existing symlink
        if os.path.islink(claude_dir) and not os.path.isdir(claude_dir):
            logger.warning(".claude is a dangling symlink, refusing: %s", claude_dir)
            return
        os.makedirs(claude_dir, exist_ok=True)
        # Write with restricted permissions (owner-only: 600)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.warning("Failed to save session registry to %s: %s", path, e)
    else:
        _register_workdir(workdir)


def _persist_session(workdir: str, name: str, claude_uuid: str, **kwargs) -> None:
    """Persist a name→uuid mapping to the project's .claude/claude-session.json.
    
    Enriched fields: model, permission_mode, resume_count, last_resume_status,
    jsonl_size, tmux_session, status.
    """
    data = _load_session_registry(workdir)
    now = datetime.now().isoformat()
    existing = data.get(name, {})
    
    # Increment resume_count if this is an auto-resume
    resume_count = existing.get("resume_count", 0)
    if kwargs.get("auto_resumed"):
        resume_count += 1
    
    data[name] = {
        "claude_session_uuid": claude_uuid,
        "workdir": workdir,
        "name": name,
        # Time tracking
        "created_at": existing.get("created_at", now),
        "updated_at": now,
        "last_active_at": now,
        # Usage tracking
        "resume_count": resume_count,
        "total_starts": existing.get("total_starts", 0) + 1,
        # Session config
        "model": kwargs.get("model", existing.get("model")),
        "permission_mode": kwargs.get("permission_mode", existing.get("permission_mode")),
        "tmux_session": kwargs.get("tmux_session", existing.get("tmux_session")),
        # Resume metadata
        "last_resume_status": (
            "auto_resumed" if kwargs.get("auto_resumed")
            else kwargs.get("resume_status", existing.get("last_resume_status"))
        ),
        "jsonl_size": kwargs.get("jsonl_size", existing.get("jsonl_size")),
        # Status
        "status": "active",
    }
    _save_session_registry(workdir, data)


def _update_session_status(workdir: str, name: str, status: str) -> None:
    """Update session status in persistence (e.g. 'stopped', 'active')."""
    data = _load_session_registry(workdir)
    if name in data and isinstance(data[name], dict):
        data[name]["status"] = status
        data[name]["last_active_at"] = datetime.now().isoformat()
        _save_session_registry(workdir, data)


def _load_persisted_session(workdir: str, name: str) -> Optional[str]:
    """Load a persisted claude_session_uuid for a given name. Returns None if not found."""
    data = _load_session_registry(workdir)
    entry = data.get(name)
    if entry and isinstance(entry, dict):
        return entry.get("claude_session_uuid")
    return None


def _load_persisted_sessions(workdir: str) -> dict:
    """Load all persisted sessions for a workdir. Returns the full registry dict."""
    return _load_session_registry(workdir)

# Per-gateway-session status observers — bridges session status to Telegram.
# Keyed by gateway_session_key so concurrent sessions route to the correct chat.
from typing import Callable
_status_observers: dict[str, Callable[[str, dict], None]] = {}  # gw_key → callback(session_id, info)
_status_observers_lock = threading.Lock()
# Gateway adapter registry: gw_key → {loop, send_func, edit_func, delete_func, chat_id, timestamp}
_gateway_adapters: dict[str, dict] = {}
_gateway_adapters_lock = threading.Lock()
_gateway_adapters_ttl = 300  # seconds — clean up stale entries after 5 minutes


def register_status_observer(callback, gateway_session_key: str = ""):
    """Register a status observer for a specific gateway session.

    Called by gateway/run.py to bridge ClaudeSession status updates
    to Telegram status messages. The callback receives (session_id, status_info).

    Uses per-gateway-session-key isolation so concurrent sessions (e.g. a DM
    and a group chat running in parallel) each route status updates to the
    correct chat instead of overwriting each other.
    """
    with _status_observers_lock:
        _status_observers[gateway_session_key] = callback


def unregister_status_observer(gateway_session_key: str = ""):
    """Remove the status observer for a specific gateway session."""
    with _status_observers_lock:
        _status_observers.pop(gateway_session_key, None)


def register_gateway_adapter(
    gateway_session_key: str,
    loop,
    send_func,
    edit_func,
    delete_func,
    chat_id: str,
):
    """Register Gateway adapter callbacks for StatusCard to use.

    Called by gateway/run.py when setting up the status bridge.
    StatusCard reads these to send/edit/delete messages via the Gateway's
    platform adapter (instead of creating its own Bot instance).
    """
    with _gateway_adapters_lock:
        _gateway_adapters[gateway_session_key] = {
            "loop": loop,
            "send_func": send_func,
            "edit_func": edit_func,
            "delete_func": delete_func,
            "chat_id": chat_id,
            "timestamp": time.time(),
        }


def _cleanup_stale_adapters():
    """Remove adapter entries older than TTL to prevent memory leaks."""
    cutoff = time.time() - _gateway_adapters_ttl
    with _gateway_adapters_lock:
        stale = [k for k, v in _gateway_adapters.items() if v.get("timestamp", 0) < cutoff]
        for k in stale:
            _gateway_adapters.pop(k, None)
        return stale


def unregister_gateway_adapter(gateway_session_key: str = ""):
    """Remove the adapter registration for a specific gateway session."""
    with _gateway_adapters_lock:
        _gateway_adapters.pop(gateway_session_key, None)
        # Also clean up any stale entries while we're here
        cutoff = time.time() - _gateway_adapters_ttl
        stale = [k for k, v in _gateway_adapters.items() if v.get("timestamp", 0) < cutoff]
        for k in stale:
            _gateway_adapters.pop(k, None)


def _get_gateway_session_key() -> str:
    """读取当前 gateway session_key（并发安全）。

    优先从 contextvars 读取（gateway 模式，每个 Telegram 群聊独立），
    回退到 os.environ（CLI/cron 模式），都为空则返回空串（无隔离）。
    """
    if get_session_env is not None:
        try:
            key = get_session_env("HERMES_SESSION_KEY", "")
            if key:
                return key
        except Exception:
            pass
    return os.environ.get("HERMES_SESSION_KEY", "")


def _safe_call_observer(observer: Callable[[str, dict], None], session_id: str, status_info: dict) -> None:
    """Safely call an observer with exception handling.

    Wraps observer callbacks to prevent crashes when the underlying resources
    (e.g., gateway session, event loop) have been cleaned up. Silently logs
    errors rather than propagating them to the Claude Code session manager.

    Args:
        observer: The observer callback to call
        session_id: Claude session ID
        status_info: Status information dictionary
    """
    try:
        observer(session_id, status_info)
    except Exception as e:
        logger.debug(
            "Observer callback error (session=%s, gateway_key=%s): %s",
            session_id,
            _get_gateway_session_key(),
            e,
        )


def _derive_session_name(workdir: str, gateway_session_key: str = "", name: Optional[str] = None) -> str:
    """基于 workdir + gateway session_key 生成确定性 tmux session 名。

    gateway 模式下，同一 workdir 的不同 Telegram 群聊会得到不同的 tmux 名。
    CLI/cron 模式下（gateway_session_key 为空），退化为纯 workdir 哈希。
    格式：hermes-{sha256前8位}

    如果提供了 name，将其纳入哈希计算，使同一 workdir 下不同命名的会话
    得到不同的 tmux session 名。
    """
    abs_path = os.path.abspath(workdir)
    parts = [abs_path]
    if gateway_session_key:
        parts.append(gateway_session_key)
    if name:
        parts.append(name)
    combined = ":".join(parts)
    h = hashlib.sha256(combined.encode()).hexdigest()[:8]
    return f"hermes-{h}"


def _validate_session_name(name: str, gw_key: str) -> Optional[str]:
    """验证 session name，返回错误信息或 None。

    规则：
      - 非空，1-64 字符
      - 只允许 [a-zA-Z0-9_-]
      - 同一 gateway_key 下不能重复
    """
    if not name:
        return "session name cannot be empty"
    if len(name) > 64:
        return f"session name too long ({len(name)} > 64)"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        return "session name must contain only letters, digits, underscores, and hyphens"
    with _sessions_lock:
        existing_id = _name_index.get((gw_key, name))
        if existing_id and existing_id in _sessions:
            return f"name '{name}' already in use by session {existing_id[:8]}"
    return None


def _touch_active(gw_key: str, session_id: str, already_locked: bool = False):
    """更新活跃 session 记录（gateway_key → 最新 session_id）。

    在 start / send / interact 时调用，确保 _active_session 始终指向
    该 gateway 下最近交互的会话。

    Args:
        already_locked: 为 True 时跳过加锁（调用方已持有 _sessions_lock）。
    """
    if already_locked:
        _active_session[gw_key] = session_id
    else:
        with _sessions_lock:
            _active_session[gw_key] = session_id


def _get_session(session_id: str = None, gateway_session_key: str = "", strict: bool = False):
    """获取指定会话，无 session_id 时返回当前 gateway session 的最近会话。

    Args:
        session_id: 目标会话 ID。None 时按 gateway_session_key 过滤后返回最近的会话。
        gateway_session_key: 当前 gateway session key，用于隔离不同 Telegram 群聊。
        strict: 为 True 时，指定了 session_id 但找不到则返回 None（不回退），
                用于 stop/操作类 action 防止操作错误会话。
    """
    with _sessions_lock:
        if session_id:
            if session_id in _sessions:
                return _sessions[session_id]
            # session_id 已明确指定但找不到
            if strict:
                logger.warning(
                    "session_id=%s not found in registry (known: %s). "
                    "Possible gateway restart lost in-memory state.",
                    session_id, list(_sessions.keys()),
                )
                return None
        # 优先使用 _active_session 记录（由 start/send/switch 更新）。
        # 注意：CLI/部分 gateway 路径的 key 可能是空串，空串也是有效作用域。
        active_id = _active_session.get(gateway_session_key)
        if active_id and active_id in _sessions:
            return _sessions[active_id]

        # 回退：按 gateway session-key 过滤后返回最后创建的会话。
        sessions_for_gateway = [
            mgr for mgr in _sessions.values()
            if getattr(mgr, "_gateway_session_key", "") == gateway_session_key
        ]
        if sessions_for_gateway:
            return sessions_for_gateway[-1]

        # 最后兜底：跨作用域返回全局最后一个（主要兼容旧 CLI/cron 行为）。
        if _sessions:
            return list(_sessions.values())[-1]
    return None


def _get_session_by_workdir(workdir: str, gateway_session_key: str = ""):
    """通过 (gateway_session_key, workdir) 查找已注册的会话。

    无锁，调用方需持有 _sessions_lock。
    """
    abs_path = os.path.abspath(workdir)
    idx_key = (gateway_session_key, abs_path)
    sid = _workdir_index.get(idx_key)
    if sid and sid != "__starting__" and sid in _sessions:
        return _sessions[sid]
    return None


def _resolve_target_session(args: dict, gw_key: str):
    """根据 args 中的 session_id / name 解析目标会话。

    优先级：session_id > name > _active_session[gw_key] > 最近会话。
    返回 (mgr, error_json) 元组，mgr 为 None 时 error_json 非空。
    """
    sid = args.get("session_id")
    name = args.get("name")

    if sid:
        mgr = _get_session(sid, gateway_session_key=gw_key, strict=True)
        if mgr is None:
            return None, tool_error(f"Session '{sid}' not found in registry.")
        return mgr, None

    if name:
        with _sessions_lock:
            resolved_id = _name_index.get((gw_key, name))
            # Cross-gateway fallback: search all gateway contexts for the name
            if not resolved_id:
                for (gk, n), sid in _name_index.items():
                    if n == name and sid and sid != "__starting__":
                        resolved_id = sid
                        break
        if not resolved_id:
            return None, tool_error(f"No session named '{name}' in current gateway context.")
        mgr = _get_session(resolved_id, gateway_session_key=gw_key, strict=True)
        if mgr is None:
            return None, tool_error(f"Session '{name}' (id={resolved_id[:8]}) no longer exists.")
        return mgr, None

    # 无指定时回退到 _active_session 或最近会话
    mgr = _get_session(None, gateway_session_key=gw_key)
    if mgr is None:
        return None, tool_error("No active session. Use 'start' first.")
    return mgr, None


def _list_sessions(gateway_key: str) -> dict:
    """列出当前 gateway 下的所有 session，包含名称、workdir 和状态。"""
    sessions = []
    with _sessions_lock:
        for sid, mgr in _sessions.items():
            if getattr(mgr, "_gateway_session_key", "") != gateway_key:
                continue
            # 优先从 mgr 属性读取 name，回退到 _name_index 反查
            session_name = getattr(mgr, "_session_name", None)
            if not session_name:
                for (gk, n), mapped_sid in _name_index.items():
                    if gk == gateway_key and mapped_sid == sid:
                        session_name = n
                        break
            sessions.append({
                "session_id": sid,
                "name": session_name,
                "workdir": getattr(mgr, "_workdir", None),
                "state": mgr._sm.current_state if hasattr(mgr, "_sm") else "UNKNOWN",
                "active": getattr(mgr, "_session_active", False),
            })
        active_id = _active_session.get(gateway_key)
    return {
        "sessions": sessions,
        "active_session_id": active_id,
        "total": len(sessions),
    }


def _switch_session(name: str, gateway_key: str) -> dict:
    """切换活跃 session 到指定 name。"""
    with _sessions_lock:
        target_id = _name_index.get((gateway_key, name))
        if not target_id:
            return {"error": f"No session named '{name}' in current gateway context."}
        target_mgr = _sessions.get(target_id)
        if not target_mgr or not getattr(target_mgr, "_session_active", False):
            return {"error": f"Session '{name}' exists but is not active (may have been stopped)."}
        _active_session[gateway_key] = target_id
    return {
        "switched_to": name,
        "session_id": target_id,
        "state": target_mgr._sm.current_state if hasattr(target_mgr, "_sm") else "UNKNOWN",
    }


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CLAUDE_SESSION_SCHEMA = {
    "name": "claude_session",
    "description": (
        "Interactive Claude Code session via tmux — PREFERRED way to delegate coding tasks.\n"
        "Actions: start|send|type|submit|status|wait_for_idle|output|respond_permission|"
        "respond_interview|stop|history|events|list|switch|diagnose|doctor_fix\n\n"
        "WHEN TO USE: Complex multi-file coding tasks, tasks needing real-time monitoring, "
        "long-running sessions with state tracking, parallel named sessions (name REQUIRED for start).\n\n"
        "WHEN NOT TO USE: Simple shell commands (-> terminal), non-Claude reasoning (-> delegate_task), "
        "one-shot questions (-> terminal with 'claude -p').\n\n"
        "Provides real-time state awareness (IDLE/THINKING/TOOL_CALL/PERMISSION/INTERVIEW), "
        "atomic send, and permission handling.\n"
        "Load 'claude-session' skill for detailed workflows, collaboration principles, and troubleshooting."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "start", "send", "type", "submit", "send_text", "cancel_input",
                    "status", "wait_for_idle", "wait_for_state",
                    "output", "jsonl_output", "respond_permission", "respond_interview", "stop", "history", "events",
                    "list", "switch",
                    "diagnose", "doctor_fix",
                ],
                "description": "Action to perform on the Claude session",
            },
            # 多会话路由
            "session_id": {
                "type": "string",
                "description": "目标会话ID（可选，默认最近活跃的会话）",
            },
            "name": {
                "type": "string",
                "description": "Named session identifier — REQUIRED. Used with start/list/switch/stop for multi-session management. Must be 1-64 chars, [a-zA-Z0-9_-].",
            },
            # start
            "workdir": {
                "type": "string",
                "description": "Working directory for 'start' action",
            },
            "session_name": {
                "type": "string",
                "description": "tmux session name (default: hermes-{sha256[:8]} based on workdir)",
            },
            "model": {
                "type": "string",
                "description": "Claude model to use (e.g. 'sonnet', 'opus')",
            },
            "permission_mode": {
                "type": "string",
                "enum": ["normal", "skip"],
                "description": "Permission mode: 'normal' (Claude asks) or 'skip' (auto-approve)",
            },
            "on_event": {
                "type": "string",
                "enum": ["notify", "queue", "none"],
                "description": "Event delivery mode (default: 'notify')",
            },
            "resume_uuid": {
                "type": "string",
                "description": "Claude Code session UUID to resume (optional). If provided, starts with --resume to restore history.",
            },
            "force_new": {
                "type": "boolean",
                "description": "For 'start': force creating a new session instead of auto-resuming an existing one with the same name.",
            },
            # send / type / start
            "message": {
                "type": "string",
                "description": "Message text for 'send' action. For 'start' action, if provided, automatically sends this message after session starts successfully.",
            },
            "text": {
                "type": "string",
                "description": "Text for 'type' action (no Enter)",
            },
            # wait_for_idle / wait_for_state
            "timeout": {
                "type": "integer",
                "description": "Max seconds to wait (default: 900 for wait_for_idle, 60 for wait_for_state). Claude Code tasks typically take 3-30 minutes. Use 900 for normal tasks, 1800 for heavy analysis.",
                "minimum": 1,
            },
            "target_state": {
                "type": "string",
                "description": "Target state for 'wait_for_state' action",
            },
            # output
            "offset": {
                "type": "integer",
                "description": "Line offset for 'output' action",
            },
            "limit": {
                "type": "integer",
                "description": "Max lines for 'output' action. Default: 50. Use 100-500 for detailed review.",
                "minimum": 1,
            },
            # jsonl_output
            "last_reply": {
                "type": "boolean",
                "description": "For 'jsonl_output': return the last assistant text reply (use when tmux output is truncated)",
            },
            "last_n": {
                "type": "integer",
                "description": "For 'jsonl_output': return the last N assistant text replies. Omit for summary only.",
                "minimum": 1,
            },
            "max_length": {
                "type": "integer",
                "description": "For 'jsonl_output': max character length for each reply. Replies exceeding this are truncated. Default: 15000.",
                "minimum": 1000,
            },
            # respond_permission
            "response": {
                "type": "string",
                "enum": ["allow", "deny"],
                "description": "Permission response for 'respond_permission' action",
            },
            # respond_interview
            "option": {
                "type": "string",
                "description": "Option for 'respond_interview' action. IMPORTANT: analyze the interview context (question + options) from wait_for_idle result and choose the best answer for the task. A number (e.g. '1') to select that numbered option, 'enter' to confirm current selection, 'escape' to cancel, or text to type as custom input.",
            },
            # events
            "since_turn": {
                "type": "integer",
                "description": "Filter events since turn ID for 'events' action",
            },
            # doctor_fix
            "apply": {
                "type": "boolean",
                "description": "For 'doctor_fix': False=analyze only (default), True=execute fixes",
            },
            "strategy": {
                "type": "string",
                "enum": ["project", "user", "merge"],
                "description": "For 'doctor_fix': merge strategy — 'project' (default), 'user', or 'merge'",
            },
        },
        "required": ["action"],
    },
}


# ---------------------------------------------------------------------------
# Output streaming helper
# ---------------------------------------------------------------------------

def _setup_output_streamer(mgr, gw_key: str):
    """Create a SessionOutputStreamer if a gateway adapter is available.

    Wires the streamer into the session's observer callback chain so
    incremental output is pushed to the user's chat in real-time.
    """
    from tools.claude_session.stream_output import SessionOutputStreamer
    try:
        with _gateway_adapters_lock:
            adapter_info = _gateway_adapters.get(gw_key)
        if not adapter_info or not adapter_info.get("loop"):
            return None
        chat_id = adapter_info.get("chat_id", "")
        if not chat_id:
            return None

        streamer = SessionOutputStreamer(mgr, adapter_info)

        # Chain into existing status callback (preserves StatusCard updates)
        existing_cb = mgr._status_callback

        def _chained(info: dict) -> None:
            if existing_cb:
                try:
                    existing_cb(info)
                except Exception:
                    pass
            streamer.on_observer_update(info)

        mgr._status_callback = _chained
        mgr._output_streamer = streamer
        return streamer
    except Exception as e:
        logger.debug("Output streamer setup failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def _handle_claude_session(args, **kw):
    """Dispatch claude_session tool calls (支持多会话路由 + gateway session 隔离 + named sessions)."""
    action = args.get("action", "")
    gw_key = _get_gateway_session_key()

    # ── start：创建新实例并注册（name 必填 + workdir 可选）──
    if action == "start":
        # 每次 start 时清理过期 adapter，防止断开时未正确注销导致内存泄漏
        _cleanup_stale_adapters()

        from tools.claude_session.session import ClaudeSession

        # name 必须提供（第一性原理：明确会话用途，避免自动命名的混淆）
        session_name_arg = args.get("name")
        if not session_name_arg:
            return json.dumps({"error": "name is required for 'start' action. Provide a meaningful name to identify the session."}, ensure_ascii=False)

        # 验证 name
        err = _validate_session_name(session_name_arg, gw_key)
        if err:
            return json.dumps({"error": err}, ensure_ascii=False)

        workdir = args.get("workdir", ".")
        abs_workdir = os.path.abspath(workdir)
        idx_key = (gw_key, session_name_arg)  # 改为基于 name 的索引键

        # 生成确定性 tmux session 名（基于 name，不再依赖 workdir）
        sn = _derive_session_name(abs_workdir, gw_key, name=session_name_arg)

        # 查找已有会话（锁内仅读取注册表，不执行耗时操作）
        # check-and-set: name 查找 + 占位符注册在同一锁内，防止并发 start
        # 双重创建（TOCTOU fix）。
        _existing_resp = None
        _old_mgr_to_stop = None  # 需要在锁外 stop 的旧 session
        _old_sid_to_cleanup = None  # 需要在锁外清理注册表的旧 session_id
        with _sessions_lock:
            existing_sid = _name_index.get((gw_key, session_name_arg))
            if existing_sid == "__starting__":
                # 另一个线程正在创建同名会话 — 拒绝本次请求
                return json.dumps(
                    {"error": f"Session '{session_name_arg}' is currently being created by another request. Retry in a moment."},
                    ensure_ascii=False,
                )
            if existing_sid and existing_sid in _sessions and _sessions[existing_sid]._session_active:
                existing_mgr = _sessions[existing_sid]
                # 检查关键参数是否变化：workdir/model/permission_mode/resume_uuid
                old_workdir = getattr(existing_mgr, "_workdir", "")
                old_model = getattr(existing_mgr, "_model", None)  # may not exist
                old_perm = getattr(existing_mgr, "_permission_mode", "normal")
                new_model = args.get("model")
                new_perm = args.get("permission_mode", "normal")
                has_resume = bool(args.get("resume_uuid"))

                params_changed = (
                    old_workdir != abs_workdir
                    or old_perm != new_perm
                    or has_resume
                    or (new_model is not None and new_model != (old_model or None))
                )

                if params_changed:
                    # 参数变化：需要 stop 旧 session 再 start 新的
                    logger.info(
                        "Session '%s' params changed (workdir=%s→%s, perm=%s→%s, resume=%s), auto-stopping old session",
                        session_name_arg, old_workdir, abs_workdir, old_perm, new_perm, has_resume,
                    )
                    # 锁内只做轻量操作：清理注册表 + 预占，耗时 stop 延迟到锁外
                    existing_mgr._status_callback = None
                    # 立即清理注册表，释放 name 槽位
                    _sessions.pop(existing_sid, None)
                    keys_to_remove = []
                    for k, v_list in _workdir_index.items():
                        if existing_sid in v_list:
                            updated_list = [sid for sid in v_list if sid != existing_sid]
                            if updated_list:
                                _workdir_index[k] = updated_list
                            else:
                                keys_to_remove.append(k)
                    for k in keys_to_remove:
                        _workdir_index.pop(k, None)
                    name_keys = [k for k, v in _name_index.items() if v == existing_sid]
                    for k in name_keys:
                        _name_index.pop(k, None)
                    if _active_session.get(gw_key) == existing_sid:
                        _active_session.pop(gw_key, None)
                    # 预占新槽位
                    _name_index[(gw_key, session_name_arg)] = "__starting__"
                    workdir_idx_key = (gw_key, abs_workdir)
                    existing_list = _workdir_index.get(workdir_idx_key, [])
                    _workdir_index[workdir_idx_key] = existing_list + ["__starting__"]
                    # 记录待 stop 的实例（锁外执行）
                    _old_mgr_to_stop = existing_mgr
                    _old_sid_to_cleanup = existing_sid
                else:
                    # 参数没变：复用已有 session
                    _touch_active(gw_key, existing_sid, already_locked=True)
                    _existing_resp = {
                        "session_id": existing_sid,
                        "tmux_session": existing_mgr._tmux.session_name if existing_mgr._tmux else None,
                        "state": existing_mgr._sm.current_state,
                        "permission_mode": existing_mgr._permission_mode,
                        "claude_session_uuid": existing_mgr._claude_session_uuid,
                        "note": f"Session '{session_name_arg}' already active",
                        "name": session_name_arg,
                    }
            else:
                # 预占槽位，防止并发 start 时双重创建
                _name_index[(gw_key, session_name_arg)] = "__starting__"
                workdir_idx_key = (gw_key, abs_workdir)
                existing_list = _workdir_index.get(workdir_idx_key, [])
                _workdir_index[workdir_idx_key] = existing_list + ["__starting__"]

        # 锁外执行耗时操作：stop 旧 session（tmux kill + observer stop）
        if _old_mgr_to_stop is not None:
            try:
                _old_mgr_to_stop.stop()
            except Exception as e:
                logger.warning("Auto-stop old session '%s' (sid=%s) failed: %s", session_name_arg, _old_sid_to_cleanup[:8] if _old_sid_to_cleanup else "?", e)

        # 已有活跃会话（参数未变）：在锁外处理 message
        if _existing_resp is not None:
            if args.get("message"):
                try:
                    send_result = existing_mgr.send(args["message"])
                    if "error" in send_result:
                        _existing_resp["send_error"] = send_result["error"]
                    else:
                        _existing_resp["message_sent"] = True
                        _existing_resp["send_state"] = send_result.get("state")
                except Exception as e:
                    logger.warning("Auto-send on existing session failed (session=%s): %s", existing_sid[:8], e)
                    _existing_resp["send_error"] = str(e)
            return json.dumps(_existing_resp, ensure_ascii=False)

        try:
            # Build status_card_config from gateway adapter registry
            _status_card_config = None
            if get_session_env is not None:
                try:
                    _sc_platform = get_session_env("HERMES_SESSION_PLATFORM", "")
                    _sc_chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "")
                    logger.debug(
                        "StatusCard config: platform=%s chat_id=%s gw_key=%s adapters=%s",
                        _sc_platform, _sc_chat_id, gw_key, list(_gateway_adapters.keys()),
                    )
                    if _sc_platform == "telegram" and _sc_chat_id:
                        # Read adapter from gateway registry (set by gateway/run.py)
                        with _gateway_adapters_lock:
                            _adapter_info = _gateway_adapters.get(gw_key)
                        if _adapter_info:
                            _status_card_config = {
                                "chat_id": _sc_chat_id,
                                "loop": _adapter_info["loop"],
                                "send_func": _adapter_info["send_func"],
                                "edit_func": _adapter_info["edit_func"],
                                "delete_func": _adapter_info["delete_func"],
                            }
                            logger.info("StatusCard config built for gw_key=%s", gw_key)
                        else:
                            logger.warning("StatusCard: no adapter registered for gw_key=%s (registered: %s)", gw_key, list(_gateway_adapters.keys()))
                except Exception as _sce:
                    logger.warning("StatusCard config build error: %s", _sce)

            mgr = ClaudeSession()
            mgr._gateway_session_key = gw_key
            mgr._session_name = session_name_arg
            result = mgr.start(
                workdir=abs_workdir,
                session_name=sn,
                model=args.get("model"),
                permission_mode=args.get("permission_mode", "normal"),
                on_event=args.get("on_event", "notify"),
                completion_queue=kw.get("completion_queue"),
                resume_uuid=args.get("resume_uuid"),
                force_new=args.get("force_new", False),
                status_card_config=_status_card_config,
            )

        except SessionError as e:
            # 启动异常时清理占位
            with _sessions_lock:
                _name_index.pop((gw_key, session_name_arg), None)
                workdir_idx_key = (gw_key, abs_workdir)
                session_list = _workdir_index.get(workdir_idx_key, [])
                if "__starting__" in session_list:
                    updated_list = session_list.copy()
                    updated_list.remove("__starting__")
                    _workdir_index[workdir_idx_key] = updated_list
            return json.dumps(e.to_dict() | {"name": session_name_arg}, ensure_ascii=False)
        except Exception as e:
            # 启动异常时清理占位
            with _sessions_lock:
                _name_index.pop((gw_key, session_name_arg), None)
                workdir_idx_key = (gw_key, abs_workdir)
                session_list = _workdir_index.get(workdir_idx_key, [])
                if "__starting__" in session_list:
                    updated_list = session_list.copy()
                    updated_list.remove("__starting__")
                    _workdir_index[workdir_idx_key] = updated_list
            return json.dumps({"error": f"Failed to create session: {e}"}, ensure_ascii=False)

        # 仅启动成功时注册到会话表和索引
        if "error" not in result:
            sid = result.get("session_id")
            if sid:
                with _sessions_lock:
                    _sessions[sid] = mgr
                    _name_index[(gw_key, session_name_arg)] = sid  # 基于 name 的主索引
                    workdir_idx_key = (gw_key, abs_workdir)
                    session_list = _workdir_index.get(workdir_idx_key, [])
                    # 将占位符替换为实际的 session_id，或添加到列表
                    if "__starting__" in session_list:
                        updated_list = [sid if x == "__starting__" else x for x in session_list]
                        _workdir_index[workdir_idx_key] = updated_list
                    else:
                        _workdir_index[workdir_idx_key] = session_list + [sid]
                _touch_active(gw_key, sid)
                # Persist name→uuid mapping to .claude/claude-session.json
                claude_uuid = result.get("claude_session_uuid")
                if claude_uuid and session_name_arg:
                    _persist_session(
                        abs_workdir, session_name_arg, claude_uuid,
                        model=args.get("model"),
                        permission_mode=args.get("permission_mode", "normal"),
                        tmux_session=result.get("tmux_session"),
                        auto_resumed=result.get("auto_resumed", False),
                        resume_status=(
                            "auto_resumed" if result.get("auto_resumed")
                            else ("manual_resume" if result.get("resumed_from") else "new")
                        ),
                    )
                # Attach status observer for this gateway session (per-key isolation).
                # Only set the bridge callback if StatusCard hasn't already set one.
                # StatusCard's _status_callback sends real-time Telegram updates;
                # the bridge callback routes to gateway's StatusMessageManager (now removed).
                # Co-existence: if both exist, chain them.

                with _status_observers_lock:
                    _observer = _status_observers.get(gw_key)
                if _observer:
                    _existing = mgr._status_callback
                    _bridge = lambda info, _sid=sid, _obs=_observer: _safe_call_observer(_obs, _sid, info)
                    if _existing:
                        # Chain: call both sequentially, each with its own exception handling
                        def _chained(info, _ex=_existing, _br=_bridge):
                            for fn in (_ex, _br):
                                try:
                                    fn(info)
                                except Exception as e:
                                    logger.warning("Chained callback error: %s", e)
                        mgr._status_callback = _chained
                    else:
                        mgr._status_callback = _bridge
                # 返回结果中附带 name
                if session_name_arg:
                    result["name"] = session_name_arg
                # Auto-inject persistence context: top-5 recently active peers
                try:
                    _all_persisted = _load_session_registry(abs_workdir)
                    _peers = {
                        k: v for k, v in _all_persisted.items()
                        if k != session_name_arg and isinstance(v, dict)
                    }
                    if _peers:
                        # Partition: valid entries have last_active_at; rest go to end
                        _valid = [(k, v) for k, v in _peers.items() if v.get("last_active_at")]
                        _no_ts = [(k, v) for k, v in _peers.items() if not v.get("last_active_at")]
                        _valid.sort(key=lambda x: x[1]["last_active_at"], reverse=True)
                        _top5 = (_valid + _no_ts)[:5]
                        result["persistence_context"] = {
                            "total_peers": len(_peers),
                            "recent_sessions": {
                                k: {
                                    "status": v.get("status"),
                                    "last_resume_status": v.get("last_resume_status"),
                                    "last_active_at": v.get("last_active_at"),
                                }
                                for k, v in _top5
                            },
                            "hint": (
                                f"Top {len(_top5)} of {len(_peers)} persisted peers "
                                "sorted by last_active_at. Use list_persisted for full list."
                            ),
                        }
                except Exception:
                    pass  # non-critical enrichment
                # send 逻辑在 try/except 之外，避免 send 异常触发 start 清理
                if args.get("message"):
                    try:
                        send_result = mgr.send(args["message"])
                        if "error" in send_result:
                            result["send_error"] = send_result["error"]
                        else:
                            result["message_sent"] = True
                            result["send_state"] = send_result.get("state")
                    except Exception as e:
                        logger.warning("Auto-send after start failed (session=%s): %s", sid[:8], e)
                        result["send_error"] = str(e)
        else:
            # 启动失败时清理占位
            with _sessions_lock:
                workdir_idx_key = (gw_key, abs_workdir)
                session_list = _workdir_index.get(workdir_idx_key, [])
                if "__starting__" in session_list:
                    updated_list = [sid for sid in session_list if sid != "__starting__"]
                    if updated_list:
                        _workdir_index[workdir_idx_key] = updated_list
                    else:
                        _workdir_index.pop(workdir_idx_key, None)
        return json.dumps(result, ensure_ascii=False)

    # ── list：列出当前 gateway 下的所有会话 + 可恢复的持久化会话 ──
    if action == "list":
        result = _list_sessions(gw_key)
        # Enrich with resumable persisted sessions not currently in gateway
        try:
            _active_names = {s.get("name") for s in result.get("sessions", []) if s.get("name")}
            _resumable = {}
            _scanned_workdirs = set()
            # Collect all stopped sessions with (name, workdir) pairs first
            _raw_entries = []
            for s in result.get("sessions", []):
                wd = s.get("workdir")
                if wd and wd not in _scanned_workdirs:
                    _scanned_workdirs.add(wd)
                    try:
                        _persisted = _load_session_registry(wd)
                        for _name, _entry in _persisted.items():
                            if (isinstance(_entry, dict)
                                    and _name not in _active_names
                                    and _entry.get("status") == "stopped"):
                                _raw_entries.append((_name, wd, _entry))
                    except Exception:
                        pass
            # Also scan globally known workdirs (for gateway restart scenario)
            for wd in _get_known_workdirs():
                if wd not in _scanned_workdirs:
                    _scanned_workdirs.add(wd)
                    try:
                        _persisted = _load_session_registry(wd)
                        for _name, _entry in _persisted.items():
                            if (isinstance(_entry, dict)
                                    and _name not in _active_names
                                    and _entry.get("status") == "stopped"):
                                _raw_entries.append((_name, wd, _entry))
                    except Exception:
                        pass
            # Build deduped dict — use @basename suffix for same-name collisions
            _name_counts = {}
            for _name, _wd, _ in _raw_entries:
                _name_counts[_name] = _name_counts.get(_name, 0) + 1
            for _name, _wd, _entry in _raw_entries:
                _key = _name if _name_counts[_name] == 1 else f"{_name}@{os.path.basename(_wd)}"
                _resumable[_key] = {
                    "name": _name,
                    "workdir": _wd,
                    "last_active_at": _entry.get("last_active_at"),
                    "resume_count": _entry.get("resume_count", 0),
                    "model": _entry.get("model"),
                }
            if _resumable:
                # Partition: valid entries have last_active_at; rest go to end
                _valid_r = [(k, v) for k, v in _resumable.items() if v.get("last_active_at")]
                _no_ts_r = [(k, v) for k, v in _resumable.items() if not v.get("last_active_at")]
                _valid_r.sort(key=lambda x: x[1]["last_active_at"], reverse=True)
                _top10 = (_valid_r + _no_ts_r)[:10]
                result["resumable"] = {
                    "total": len(_resumable),
                    "sessions": dict(_top10),
                    "hint": "These stopped sessions can be resumed with: "
                            "claude_session(action='start', name='<name>', workdir='<workdir>')",
                }
        except Exception:
            pass  # non-critical enrichment
        return json.dumps(result, ensure_ascii=False)

    # ── list_persisted：列出 workdir 下持久化的会话（v4.4+）──
    if action == "list_persisted":
        workdir = args.get("workdir", ".")
        abs_workdir = os.path.abspath(workdir)
        try:
            validated_wd = _validate_workdir(abs_workdir)
        except ValueError as e:
            return json.dumps({"error": str(e), "workdir": abs_workdir}, ensure_ascii=False)
        persisted = _load_persisted_sessions(validated_wd)
        # Cross-reference with active sessions in current gateway
        active_names = set()
        with _sessions_lock:
            for (g, n), sid in _name_index.items():
                if g == gw_key and sid in _sessions and _sessions[sid]._session_active:
                    active_names.add(n)
        # Enrich with active marker. NOTE: 'status' reflects the persisted
        # state at last write (active / stopped), NOT the current process
        # state. Use 'active_in_gateway' for current liveness — a session
        # with status='active' may not be running (e.g. after process restart
        # or stop without the persistence write succeeding).
        for name, entry in persisted.items():
            if isinstance(entry, dict):
                entry["active_in_gateway"] = name in active_names
        return json.dumps({
            "workdir": validated_wd,
            "count": len(persisted),
            "field_legend": {
                "status": "Persisted status at last write (active|stopped). "
                          "NOT a live process indicator.",
                "active_in_gateway": "True if this session is currently "
                                     "running in the active gateway context. "
                                     "Use this for liveness checks.",
                "last_active_at": "ISO timestamp of the most recent persistence write.",
                "resume_count": "Number of times this session has been auto-resumed.",
            },
            "sessions": persisted,
        }, ensure_ascii=False)

    # ── switch：切换活跃会话 ──
    if action == "switch":
        name = args.get("name")
        if not name:
            return tool_error("name is required for switch action")
        result = _switch_session(name, gw_key)
        return json.dumps(result, ensure_ascii=False)

    # ── stop：停止并从注册表和索引移除 ──
    if action == "stop":
        specified_id = args.get("session_id")
        name = args.get("name")

        # name → 解析 session_id
        if name and not specified_id:
            with _sessions_lock:
                specified_id = _name_index.get((gw_key, name))
            if not specified_id:
                return tool_error(f"No session named '{name}' in current gateway context.")

        mgr = _get_session(specified_id, gateway_session_key=gw_key, strict=bool(specified_id))
        if mgr is None:
            return tool_error(
                f"Session '{specified_id or name}' not found in registry. "
                "It may have been lost after a gateway restart. "
                "Use tmux directly to clean up orphaned sessions."
            )
        # Clear callback before stop to prevent late callbacks to cleaned-up resources.
        mgr._status_callback = None
        try:
            result = mgr.stop()
        except SessionError as e:
            result = e.to_dict()
        if result.get("stopped"):
            stopped_id = result.get("session_id")
            with _sessions_lock:
                _sessions.pop(stopped_id, None)
                # 清理 _workdir_index（值是列表，需要从列表中移除 session_id）
                keys_to_remove = []
                for k, v_list in _workdir_index.items():
                    if stopped_id in v_list:
                        updated_list = [sid for sid in v_list if sid != stopped_id]
                        if updated_list:
                            _workdir_index[k] = updated_list
                        else:
                            keys_to_remove.append(k)
                for k in keys_to_remove:
                    _workdir_index.pop(k, None)
                # 清理 _name_index
                name_keys = [k for k, v in _name_index.items() if v == stopped_id]
                for k in name_keys:
                    _name_index.pop(k, None)
                # 清理 _active_session（如果指向刚停止的会话）
                if _active_session.get(gw_key) == stopped_id:
                    _active_session.pop(gw_key, None)
            # Update persistence: mark session as stopped in .claude/claude-session.json
            # so future list_persisted runs show accurate status.
            stopped_name = getattr(mgr, "_session_name", None) or name
            stopped_workdir = getattr(mgr, "_workdir", None)
            if stopped_name and stopped_workdir:
                try:
                    _update_session_status(stopped_workdir, stopped_name, "stopped")
                except Exception as e:
                    logger.debug("Failed to update persistence status for stopped session %s: %s", stopped_name, e)
        return json.dumps(result, ensure_ascii=False)

    # ── diagnose：不需要会话实例 ──
    if action == "diagnose":
        result = _diagnose_claude_session()
        return json.dumps(result, ensure_ascii=False)

    # ── doctor_fix：诊断并修复技能文件同步 ──
    if action == "doctor_fix":
        result = _doctor_fix_skills(
            apply=args.get("apply", False),
            strategy=args.get("strategy", "project"),
        )
        return json.dumps(result, ensure_ascii=False)

    # ── 其他动作：通过 _resolve_target_session 路由（支持 session_id / name / 活跃回退）──
    mgr, resolve_err = _resolve_target_session(args, gw_key)
    if mgr is None:
        # 只读查询 action：无会话时返回优雅默认值
        if action == "status":
            return json.dumps({"state": "DISCONNECTED"}, ensure_ascii=False)
        if action == "output":
            return json.dumps({"lines": [], "offset": 0, "total": 0}, ensure_ascii=False)
        if action == "jsonl_output":
            return json.dumps({"error": "No active session", "replies": []}, ensure_ascii=False)
        if action == "events":
            return json.dumps({"events": []}, ensure_ascii=False)
        if action == "history":
            return json.dumps({"total_turns": 0, "turns": []}, ensure_ascii=False)
        return resolve_err or tool_error("No active session. Use 'start' first.")

    # 交互类 action 更新 _active_session
    if action in ("send", "type", "submit", "send_text", "respond_permission", "respond_interview", "cancel_input"):
        _touch_active(gw_key, mgr._session_id)

    if action == "send":
        message = args.get("message")
        if not message:
            return tool_error("message is required for send action")
        try:
            result = mgr.send(message)
        except SessionError as e:
            return json.dumps(e.to_dict(), ensure_ascii=False)
    elif action == "type":
        text = args.get("text")
        if not text:
            return tool_error("text is required for type action")
        try:
            result = mgr.type_text(text)
        except SessionError as e:
            return json.dumps(e.to_dict(), ensure_ascii=False)
    elif action == "submit":
        try:
            result = mgr.submit()
        except SessionError as e:
            return json.dumps(e.to_dict(), ensure_ascii=False)
    elif action == "send_text":
        text = args.get("text")
        if not text:
            return tool_error("text is required for send_text action")
        try:
            result = mgr.send_text(text)
        except SessionError as e:
            return json.dumps(e.to_dict(), ensure_ascii=False)
    elif action == "cancel_input":
        try:
            result = mgr.cancel_input()
        except SessionError as e:
            return json.dumps(e.to_dict(), ensure_ascii=False)
    elif action == "status":
        result = mgr.status()
    elif action == "wait_for_idle":
        try:
            # Set up output streamer if gateway adapter is available
            _streamer = _setup_output_streamer(mgr, gw_key)

            result = mgr.wait_for_idle(timeout=args.get("timeout", 900))

            # Finalize streamer (sends final edit without cursor)
            if _streamer:
                _streamer.finish()
                mgr._output_streamer = None
                if _streamer.already_sent:
                    result["output_streamed"] = True

        except SessionError as e:
            if mgr._output_streamer:
                mgr._output_streamer.finish()
                mgr._output_streamer = None
            return json.dumps(e.to_dict(), ensure_ascii=False)
    elif action == "wait_for_state":
        target = args.get("target_state")
        if not target:
            return tool_error("target_state is required for wait_for_state action")
        try:
            result = mgr.wait_for_state(target_state=target, timeout=args.get("timeout", 60))
        except SessionError as e:
            return json.dumps(e.to_dict(), ensure_ascii=False)
    elif action == "output":
        result = mgr.output(
            offset=args.get("offset", 0),
            limit=args.get("limit", 50),
        )
    elif action == "jsonl_output":
        result = mgr.jsonl_output(
            last_reply=args.get("last_reply", False),
            last_n=args.get("last_n", 0),
            max_length=args.get("max_length", 15000),
        )
    elif action == "respond_permission":
        response = args.get("response")
        if not response:
            return tool_error("response is required for respond_permission action")
        try:
            result = mgr.respond_permission(response)
        except SessionError as e:
            return json.dumps(e.to_dict(), ensure_ascii=False)
    elif action == "respond_interview":
        option = args.get("option")
        if not option:
            return tool_error("option is required for respond_interview action")
        try:
            result = mgr.respond_interview(option)
        except SessionError as e:
            return json.dumps(e.to_dict(), ensure_ascii=False)
    elif action == "history":
        result = mgr.history()
    elif action == "events":
        result = mgr.events(since_turn=args.get("since_turn", 0))
    else:
        return tool_error(
            f"Unknown action: {action}. "
            "Valid: start, send, type, submit, cancel_input, status, "
            "wait_for_idle, wait_for_state, output, respond_permission, "
            "respond_interview, stop, history, events, list, switch, "
            "diagnose, doctor_fix"
        )

    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _check_claude_session():
    """Check if tmux (hard dep) and claude CLI (soft dep) are available.
    
    Only tmux is required for the tool to register. Claude CLI availability
    is logged as a warning but does not prevent registration, because the
    user might install it later.
    """
    tmux_ok = shutil.which("tmux") is not None
    claude_ok = shutil.which("claude") is not None
    
    if not claude_ok:
        logger.warning(
            "claude_session: Claude Code CLI not found in PATH. "
            "Install with: npm install -g @anthropic-ai/claude-code"
        )
    
    if not tmux_ok:
        logger.warning(
            "claude_session: tmux not found in PATH. "
            "Install with: apt install tmux / brew install tmux"
        )
    
    return tmux_ok


def _get_active_sessions_output() -> list:
    """Gather output and state info from all active sessions for diagnose.

    Returns a list of dicts, each with session metadata, recent output,
    and state duration — used by session-level diagnose checks.
    """
    # Phase 1: collect references under lock (fast, no nested locks)
    with _sessions_lock:
        snapshot = [
            (sid, mgr) for sid, mgr in _sessions.items()
            if getattr(mgr, '_session_active', False)
        ]

    # Phase 2: read state/output outside _sessions_lock
    # (each mgr has its own internal locks)
    sessions_info = []
    for sid, mgr in snapshot:
        try:
            state = mgr._sm.current_state
            duration = mgr._sm.state_duration()
            output_tail = mgr._buf.last_n_chars(2000)
            sessions_info.append({
                "session_id": sid,
                "state": state,
                "state_duration_seconds": round(duration, 1),
                "output_tail": output_tail,
            })
        except Exception:
            logger.warning("Failed to gather session %s info for diagnose", sid[:8], exc_info=True)
    return sessions_info


def _extract_mcp_failure_count(text: str) -> int:
    """Extract MCP server failure count from output text.

    Matches patterns like "3 MCP servers failed · /mcp".
    """
    m = re.search(r"(\d+)\s+MCP\s+servers?\s+failed", text)
    return int(m.group(1)) if m else 0


def _diagnose_claude_session() -> dict:
    """Diagnose claude_session dependencies and configuration.

    Returns a structured report of all dependencies, their status,
    and remediation hints. Used by the 'diagnose' action.
    """
    import os

    checks = []
    all_ok = True
    
    # 1. tmux
    tmux_path = shutil.which("tmux")
    checks.append({
        "dependency": "tmux",
        "status": "ok" if tmux_path else "missing",
        "path": tmux_path,
        "hint": "Install: apt install tmux / brew install tmux" if not tmux_path else None,
        "required": True,
    })
    if not tmux_path:
        all_ok = False
    
    # 2. claude CLI
    claude_path = shutil.which("claude")
    checks.append({
        "dependency": "Claude Code CLI",
        "status": "ok" if claude_path else "missing",
        "path": claude_path,
        "hint": "Install: npm install -g @anthropic-ai/claude-code" if not claude_path else None,
        "required": True,
    })
    if not claude_path:
        all_ok = False
    
    # 3. HERMES_STREAM_STALE_TIMEOUT
    timeout_val = os.environ.get("HERMES_STREAM_STALE_TIMEOUT", "")
    timeout_ok = timeout_val.isdigit() and int(timeout_val) >= 300
    checks.append({
        "dependency": "HERMES_STREAM_STALE_TIMEOUT",
        "status": "ok" if timeout_ok else ("not_set" if not timeout_val else "too_low"),
        "value": timeout_val or "(not set)",
        "hint": (
            "Set to >= 300 in ~/.hermes/.env to prevent Stream Stalled errors"
            if not timeout_ok else None
        ),
        "required": False,
    })
    
    # 4. tmux version
    tmux_version = ""
    if tmux_path:
        try:
            import subprocess
            result = subprocess.run(
                [tmux_path, "-V"], capture_output=True, text=True, timeout=5
            )
            tmux_version = result.stdout.strip()
        except Exception:
            tmux_version = "unknown"
    checks.append({
        "dependency": "tmux version",
        "status": "ok" if tmux_version else "unknown",
        "value": tmux_version or "unknown",
        "required": False,
    })
    
    # 5. Claude Code version
    claude_version = ""
    if claude_path:
        try:
            import subprocess
            result = subprocess.run(
                [claude_path, "--version"], capture_output=True, text=True, timeout=10
            )
            claude_version = result.stdout.strip()
        except Exception:
            claude_version = "unknown"
    checks.append({
        "dependency": "Claude Code version",
        "status": "ok" if claude_version else "unknown",
        "value": claude_version or "unknown",
        "required": False,
    })

    # 6. 残留 tmux session 检测
    orphaned_sessions = []
    if tmux_path:
        try:
            import subprocess
            result = subprocess.run(
                ["tmux", "list-sessions"], capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                with _sessions_lock:
                    known_tmux_names = set()
                    for mgr in _sessions.values():
                        if mgr._tmux:
                            known_tmux_names.add(mgr._tmux.session_name)

                for line in result.stdout.strip().splitlines():
                    name = line.split(":")[0].strip()
                    if name.startswith("hermes-") and name not in known_tmux_names:
                        orphaned_sessions.append(name)

            orphan_count = len(orphaned_sessions)
            checks.append({
                "dependency": "orphaned tmux sessions",
                "status": "ok" if orphan_count == 0 else "warning",
                "value": f"{orphan_count} orphaned session(s)" if orphan_count else "none",
                "sessions": orphaned_sessions[:10],  # 最多显示 10 个
                "hint": (
                    "Orphaned hermes-* sessions detected. These may cause startup hangs. "
                    "Clean up with: tmux kill-session -t <name>  "
                    "or kill all: for s in $(tmux list-sessions 2>/dev/null | grep '^hermes-' | cut -d: -f1); do tmux kill-session -t \"$s\"; done"
                    if orphan_count > 0 else None
                ),
                "required": False,
            })
        except Exception:
            pass

    # ── Session-level checks: detect hung/stuck sessions ──

    active_sessions = _get_active_sessions_output()
    session_critical = False

    for sinfo in active_sessions:
        sid_short = sinfo["session_id"][:8]
        state = sinfo["state"]
        duration = sinfo["state_duration_seconds"]
        output_tail = sinfo["output_tail"]

        # P0-1: THINKING state duration check
        if state == "THINKING":
            if duration > 300:
                checks.append({
                    "dependency": f"session {sid_short} THINKING duration",
                    "status": "critical",
                    "value": f"{duration}s (>300s threshold)",
                    "session_id": sinfo["session_id"],
                    "hint": "THINKING state too long, likely hung. Try cancel_input or stop+restart.",
                    "required": False,
                })
                session_critical = True
            elif duration > 120:
                checks.append({
                    "dependency": f"session {sid_short} THINKING duration",
                    "status": "warning",
                    "value": f"{duration}s (>120s threshold)",
                    "session_id": sinfo["session_id"],
                    "hint": "THINKING state running long. Monitor or consider cancel_input.",
                    "required": False,
                })

        # P0-2: Repeated permission prompt fingerprint (bypass permissions on)
        # Match 3+ occurrences of the "bypass permissions on" line
        perm_matches = re.findall(
            r"bypass permissions on \(shift\+tab to cycle\)", output_tail
        )
        if len(perm_matches) >= 3:
            checks.append({
                "dependency": f"session {sid_short} startup hang",
                "status": "critical",
                "value": f"'bypass permissions on' repeated {len(perm_matches)} times",
                "session_id": sinfo["session_id"],
                "hint": "New version startup hang detected. Try cancel_input to unblock.",
                "required": False,
            })
            session_critical = True

        # P1-3: MCP server failures
        mcp_count = _extract_mcp_failure_count(output_tail)
        if mcp_count > 0:
            checks.append({
                "dependency": f"session {sid_short} MCP servers",
                "status": "warning",
                "value": f"{mcp_count} MCP server(s) failed",
                "session_id": sinfo["session_id"],
                "hint": "MCP initialization may block startup. Run /mcp in Claude to check.",
                "required": False,
            })

        # P1-4: CLI migration prompt
        if re.search(r"switched from npm to native installer", output_tail):
            checks.append({
                "dependency": f"session {sid_short} CLI migration",
                "status": "info",
                "value": "npm → native installer migration detected",
                "session_id": sinfo["session_id"],
                "hint": "Run 'claude install' to complete migration.",
                "required": False,
            })

        # P1-5: tmux focus-events warning
        if re.search(r"tmux focus-events off", output_tail):
            checks.append({
                "dependency": f"session {sid_short} tmux config",
                "status": "info",
                "value": "tmux focus-events is off",
                "session_id": sinfo["session_id"],
                "hint": "Add 'set -g focus-events on' to ~/.tmux.conf for better experience.",
                "required": False,
            })

    # Build status and summary
    # Three distinct states: ready / session_issues / missing_deps
    critical_count = sum(1 for c in checks if c.get("status") == "critical")
    warning_count = sum(1 for c in checks if c.get("status") == "warning")

    if not all_ok:
        top_status = "missing_deps"
        summary = "Missing required dependencies. See hints above."
    elif session_critical:
        top_status = "session_issues"
        summary = f"{critical_count} critical issue(s) in active sessions. See hints above."
    else:
        top_status = "ready"
        summary = "All dependencies met — claude_session is ready to use."
        if warning_count > 0:
            summary += f" ({warning_count} warning(s) from active sessions)"

    return {
        "status": top_status,
        "checks": checks,
        "summary": summary,
    }


def _doctor_fix_skills(apply: bool = False, strategy: str = "project") -> dict:
    """诊断并修复 claude-session 技能文件同步问题。

    两阶段操作：
      - apply=False（默认）：仅分析，不执行任何修改
      - apply=True：根据 strategy 执行修复

    Args:
        apply: False=仅分析返回报告，True=执行修复操作
        strategy: 合并策略，仅在 apply=True 且有差异时生效
            - "project": 优先项目版本（备份用户目录后创建软链接）
            - "user": 保留用户版本（将用户修改复制到项目目录）
            - "merge": 逐文件合并，项目独有的文件从项目复制，其余保留用户版本
    """
    # 定位两个目录
    user_skill_dir = os.path.expanduser("~/.hermes/skills/claude-session")
    # 项目目录：基于当前文件位置推导
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(_this_dir)  # tools/ → project root
    project_skill_dir = os.path.join(project_root, "skills", "claude-session")

    report = {
        "user_dir": user_skill_dir,
        "project_dir": project_skill_dir,
        "apply": apply,
        "strategy": strategy,
        "steps": [],
        "actions_taken": [],
        "status": "ok",
    }

    # ── Step 1: 项目目录必须存在 ──
    if not os.path.isdir(project_skill_dir):
        report["status"] = "error"
        report["steps"].append({
            "step": "check_project_dir",
            "result": "missing",
            "message": f"Project skill directory not found: {project_skill_dir}",
        })
        logger.error("doctor_fix: project skill dir missing: %s", project_skill_dir)
        return report

    report["steps"].append({
        "step": "check_project_dir",
        "result": "ok",
        "path": project_skill_dir,
        "files": _list_skill_files(project_skill_dir),
    })

    # ── Step 2: 用户目录状态检测 ──
    if not os.path.exists(user_skill_dir):
        report["steps"].append({"step": "check_user_dir", "result": "missing"})
        if not apply:
            report["status"] = "needs_fix"
            report["actions_available"] = [_action_create_symlink()]
            return report
        action_result = _create_symlink(user_skill_dir, project_skill_dir)
        report["actions_taken"].append(action_result)
        report["status"] = "fixed" if action_result["success"] else "error"
        logger.info("doctor_fix: created symlink %s -> %s", user_skill_dir, project_skill_dir)
        return report

    # 用户目录存在，判断类型
    is_link = os.path.islink(user_skill_dir)
    if is_link:
        link_target = os.readlink(user_skill_dir)
        resolved = os.path.realpath(user_skill_dir)
        project_resolved = os.path.realpath(project_skill_dir)

        # 断链检测
        if not os.path.exists(resolved):
            report["steps"].append({
                "step": "check_user_dir",
                "result": "symlink_broken",
                "target": link_target,
                "resolved": resolved,
            })
            logger.warning("Broken symlink: %s -> %s", user_skill_dir, link_target)
            if not apply:
                report["status"] = "needs_fix"
                report["actions_available"] = [_action_fix_broken_symlink(link_target)]
                return report
            return _do_fix_broken_symlink(report, user_skill_dir, project_skill_dir)

        if resolved == project_resolved:
            report["steps"].append({
                "step": "check_user_dir",
                "result": "symlink_ok",
                "target": link_target,
                "resolved": resolved,
            })
            report["status"] = "ok"
            return report
        else:
            report["steps"].append({
                "step": "check_user_dir",
                "result": "symlink_wrong",
                "current_target": link_target,
                "resolved": resolved,
                "expected": project_resolved,
            })
            if not apply:
                report["status"] = "needs_fix"
                report["actions_available"] = [_action_fix_wrong_symlink(link_target, project_resolved)]
                return report
            return _do_fix_wrong_symlink(report, user_skill_dir, project_skill_dir)

    # ── Step 3: 硬拷贝 — 比较差异 ──
    report["steps"].append({"step": "check_user_dir", "result": "hardcopy"})

    diff_result = _compare_skill_dirs(user_skill_dir, project_skill_dir)
    report["steps"].append({
        "step": "compare_content",
        "result": diff_result["summary"],
        "details": diff_result["details"],
        "summary_human": _format_diff_summary_human(diff_result),
    })

    # 顶层 diff_summary
    user_files = _list_skill_files(user_skill_dir)
    project_files = _list_skill_files(project_skill_dir)
    report["diff_summary"] = {
        "total_files": len(set(user_files) | set(project_files)),
        "differing_files": len(diff_result["details"]),
        "newer_in_project": [d["file"] for d in diff_result["details"]
                             if d["status"] in ("project_newer", "missing_in_user")],
        "newer_in_user": [d["file"] for d in diff_result["details"]
                          if d["status"] in ("user_newer", "missing_in_project")],
    }

    if diff_result["identical"]:
        if not apply:
            report["status"] = "needs_fix"
            report["actions_available"] = [_action_replace_hardcopy()]
            return report
        return _do_replace_hardcopy(report, user_skill_dir, project_skill_dir)

    # 有差异 → 根据分类和 strategy 决定
    diff_class = _classify_diff_status(diff_result["details"])

    # project_newer 且 strategy=project → 自动修复（安全操作）
    if diff_class == "project_newer" and strategy in ("project", "merge"):
        if not apply:
            report["status"] = "needs_fix"
            report["actions_available"] = [_action_backup_and_symlink()]
            return report
        return _do_backup_and_symlink(report, user_skill_dir, project_skill_dir)

    # user_newer 且 strategy=user → 同步用户修改到项目
    if diff_class == "user_newer" and strategy == "user":
        if not apply:
            report["status"] = "needs_fix"
            report["actions_available"] = [_action_sync_user_to_project(diff_result["details"])]
            return report
        return _do_sync_user_to_project(report, user_skill_dir, project_skill_dir, diff_result["details"])

    # merge 策略：逐文件合并
    if strategy == "merge":
        if not apply:
            report["status"] = "needs_fix"
            report["actions_available"] = [_action_merge_files(diff_result["details"])]
            return report
        return _do_merge_files(report, user_skill_dir, project_skill_dir, diff_result["details"])

    # 所有其他情况（user_newer+project / both_modified / 策略不匹配）
    report["status"] = "needs_user_decision"
    logger.warning("doctor_fix: diff_class=%s, needs user decision", diff_class)
    report["actions_available"] = _build_actions_for_decision(diff_class, diff_result["details"])
    return report


# ---------------------------------------------------------------------------
# doctor_fix: execute helpers (only run when apply=True)
# ---------------------------------------------------------------------------

def _do_fix_broken_symlink(report, user_dir, project_dir):
    try:
        os.remove(user_dir)
    except OSError as e:
        report["actions_taken"].append({"action": "remove_broken_symlink", "success": False, "error": str(e)})
        report["status"] = "error"
        return report
    r = _create_symlink(user_dir, project_dir)
    r["action"] = "fix_broken_symlink"
    report["actions_taken"].append(r)
    report["status"] = "fixed" if r["success"] else "error"
    return report


def _do_fix_wrong_symlink(report, user_dir, project_dir):
    try:
        os.remove(user_dir)
    except OSError as e:
        report["actions_taken"].append({"action": "remove_wrong_symlink", "success": False, "error": str(e)})
        report["status"] = "error"
        return report
    r = _create_symlink(user_dir, project_dir)
    r["action"] = "fix_symlink"
    report["actions_taken"].append(r)
    report["status"] = "fixed" if r["success"] else "error"
    return report


def _do_replace_hardcopy(report, user_dir, project_dir):
    try:
        shutil.rmtree(user_dir)
    except OSError as e:
        report["actions_taken"].append({"action": "remove_hardcopy", "success": False, "error": str(e)})
        report["status"] = "error"
        return report
    r = _create_symlink(user_dir, project_dir)
    r["action"] = "replace_hardcopy_with_symlink"
    r["reason"] = "files_identical"
    report["actions_taken"].append(r)
    report["status"] = "fixed" if r["success"] else "error"
    logger.info("doctor_fix: replaced identical hardcopy with symlink: %s", user_dir)
    return report


def _do_backup_and_symlink(report, user_dir, project_dir):
    backup_dir = user_dir + f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        os.rename(user_dir, backup_dir)
    except OSError as e:
        report["actions_taken"].append({"action": "backup_user_dir", "success": False, "error": str(e)})
        report["status"] = "error"
        return report
    report["actions_taken"].append({"action": "backup_user_dir", "success": True, "backup_path": backup_dir})
    r = _create_symlink(user_dir, project_dir)
    r["action"] = "replace_with_symlink_after_backup"
    report["actions_taken"].append(r)
    report["status"] = "fixed" if r["success"] else "error"
    logger.info("doctor_fix: backed up to %s, created symlink", backup_dir)
    return report


def _do_sync_user_to_project(report, user_dir, project_dir, details):
    """将用户修改的文件复制到项目目录，然后替换用户目录为软链接。"""
    for d in details:
        if d["status"] in ("user_newer", "missing_in_project"):
            src = os.path.join(user_dir, d["file"])
            dst = os.path.join(project_dir, d["file"])
            try:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                report["actions_taken"].append({"action": "sync_file_to_project", "file": d["file"], "success": True})
            except OSError as e:
                report["actions_taken"].append({"action": "sync_file_to_project", "file": d["file"], "success": False, "error": str(e)})
                report["status"] = "error"
                return report
    # 同步完成后替换用户目录为软链接
    try:
        shutil.rmtree(user_dir)
    except OSError as e:
        report["actions_taken"].append({"action": "remove_user_dir", "success": False, "error": str(e)})
        report["status"] = "error"
        return report
    r = _create_symlink(user_dir, project_dir)
    r["action"] = "sync_user_to_project_then_symlink"
    report["actions_taken"].append(r)
    report["status"] = "fixed" if r["success"] else "error"
    logger.info("doctor_fix: synced user changes to project, created symlink")
    return report


def _do_merge_files(report, user_dir, project_dir, details):
    """逐文件合并：项目独有的从项目复制，其余保留用户版本，然后创建软链接。"""
    for d in details:
        if d["status"] in ("missing_in_user",):
            # 项目独有的文件 → 复制到用户目录
            src = os.path.join(project_dir, d["file"])
            dst = os.path.join(user_dir, d["file"])
            try:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                report["actions_taken"].append({"action": "copy_project_file", "file": d["file"], "success": True})
            except OSError as e:
                report["actions_taken"].append({"action": "copy_project_file", "file": d["file"], "success": False, "error": str(e)})
        # user_newer / missing_in_project → 保留用户版本（不动）
        # project_newer / both_modified → 保留用户版本（用户优先）
    # 合并完成后替换用户目录为软链接
    try:
        shutil.rmtree(user_dir)
    except OSError as e:
        report["actions_taken"].append({"action": "remove_user_dir", "success": False, "error": str(e)})
        report["status"] = "error"
        return report
    r = _create_symlink(user_dir, project_dir)
    r["action"] = "merge_then_symlink"
    report["actions_taken"].append(r)
    report["status"] = "fixed" if r["success"] else "error"
    logger.info("doctor_fix: merged files, created symlink")
    return report


# ---------------------------------------------------------------------------
# doctor_fix: actions_available builders (for apply=False reports)
# ---------------------------------------------------------------------------

def _action_create_symlink():
    return {
        "action": "create_symlink",
        "description": "Create symlink to project directory",
        "command": "claude_session(action='doctor_fix', apply=True)",
    }


def _action_fix_broken_symlink(broken_target):
    return {
        "action": "fix_broken_symlink",
        "description": f"Remove broken symlink (points to {broken_target}) and recreate",
        "command": "claude_session(action='doctor_fix', apply=True)",
    }


def _action_fix_wrong_symlink(current, expected):
    return {
        "action": "fix_wrong_symlink",
        "description": f"Redirect symlink from {current} to {expected}",
        "command": "claude_session(action='doctor_fix', apply=True)",
    }


def _action_replace_hardcopy():
    return {
        "action": "replace_hardcopy_with_symlink",
        "description": "User directory is identical to project — replace with symlink",
        "command": "claude_session(action='doctor_fix', apply=True)",
    }


def _action_backup_and_symlink():
    return {
        "action": "backup_and_symlink",
        "description": "Backup user directory, then create symlink to project version",
        "command": "claude_session(action='doctor_fix', apply=True)",
    }


def _action_sync_user_to_project(details):
    files = [d["file"] for d in details if d["status"] in ("user_newer", "missing_in_project")]
    return {
        "action": "sync_user_to_project",
        "description": f"Copy {len(files)} user-modified file(s) to project, then create symlink",
        "files": files,
        "command": "claude_session(action='doctor_fix', apply=True, strategy='user')",
    }


def _action_merge_files(details):
    project_only = [d["file"] for d in details if d["status"] == "missing_in_user"]
    return {
        "action": "merge_files",
        "description": (
            f"Copy {len(project_only)} project-only file(s) to user dir, "
            "keep user versions for rest, then create symlink"
        ),
        "project_only_files": project_only,
        "command": "claude_session(action='doctor_fix', apply=True, strategy='merge')",
    }


def _build_actions_for_decision(diff_class, details):
    """构建 needs_user_decision 状态下的可用操作列表。"""
    actions = []

    # 始终提供"使用项目版本"选项
    actions.append({
        "action": "use_project_version",
        "description": "Backup user directory and create symlink to project version",
        "command": "claude_session(action='doctor_fix', apply=True, strategy='project')",
    })

    # 如果用户有更新，提供"同步用户修改"选项
    user_files = [d["file"] for d in details if d["status"] in ("user_newer", "missing_in_project")]
    if user_files:
        actions.append({
            "action": "sync_user_changes",
            "description": f"Sync {len(user_files)} user-modified file(s) to project, then symlink",
            "files": user_files,
            "command": "claude_session(action='doctor_fix', apply=True, strategy='user')",
        })

    # merge 选项
    actions.append({
        "action": "merge",
        "description": "Copy project-only files to user dir, keep user versions for rest, then symlink",
        "command": "claude_session(action='doctor_fix', apply=True, strategy='merge')",
    })

    return actions


# ---------------------------------------------------------------------------
# doctor_fix: pure utility helpers
# ---------------------------------------------------------------------------

def _list_skill_files(directory: str) -> list:
    """列出技能目录中的所有文件（相对路径）。"""
    files = []
    for root, _dirs, filenames in os.walk(directory):
        for fn in filenames:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, directory)
            files.append(rel)
    return sorted(files)


def _create_symlink(link_path: str, target_path: str) -> dict:
    """创建软链接，确保父目录存在。"""
    parent = os.path.dirname(link_path)
    try:
        os.makedirs(parent, exist_ok=True)
        os.symlink(target_path, link_path)
        return {
            "action": "create_symlink",
            "success": True,
            "link": link_path,
            "target": target_path,
        }
    except OSError as e:
        return {
            "action": "create_symlink",
            "success": False,
            "error": str(e),
        }


def _compare_skill_dirs(user_dir: str, project_dir: str) -> dict:
    """比较两个技能目录的内容差异。

    Returns:
        dict with keys:
            identical: bool
            summary: str  ("identical" | "project_newer" | "user_newer" | "both_modified")
            details: list of per-file comparison dicts
    """
    user_files = set(_list_skill_files(user_dir))
    project_files = set(_list_skill_files(project_dir))

    all_files = sorted(user_files | project_files)
    details = []
    project_newer_count = 0
    user_newer_count = 0
    both_modified_count = 0

    for rel in all_files:
        user_path = os.path.join(user_dir, rel)
        proj_path = os.path.join(project_dir, rel)

        entry = {"file": rel}

        if not os.path.exists(user_path):
            entry["status"] = "missing_in_user"
            project_newer_count += 1
        elif not os.path.exists(proj_path):
            entry["status"] = "missing_in_project"
            user_newer_count += 1
        else:
            # 比较内容
            if filecmp.cmp(user_path, proj_path, shallow=False):
                entry["status"] = "identical"
            else:
                user_mtime = os.path.getmtime(user_path)
                proj_mtime = os.path.getmtime(proj_path)
                entry["user_mtime"] = datetime.fromtimestamp(user_mtime).isoformat()
                entry["project_mtime"] = datetime.fromtimestamp(proj_mtime).isoformat()

                if proj_mtime > user_mtime:
                    entry["status"] = "project_newer"
                    project_newer_count += 1
                elif user_mtime > proj_mtime:
                    entry["status"] = "user_newer"
                    user_newer_count += 1
                else:
                    # 同一秒修改但内容不同
                    entry["status"] = "both_modified"
                    both_modified_count += 1

                # 生成 diff 摘要
                entry["diff_summary"] = _diff_summary(user_path, proj_path)

        if entry["status"] != "identical":
            details.append(entry)

    identical = len(details) == 0
    if identical:
        summary = "identical"
    elif both_modified_count > 0 or (user_newer_count > 0 and project_newer_count > 0):
        summary = "both_modified"
    elif user_newer_count == 0 and project_newer_count > 0:
        summary = "project_newer"
    elif project_newer_count == 0 and user_newer_count > 0:
        summary = "user_newer"
    else:
        summary = "both_modified"

    return {"identical": identical, "summary": summary, "details": details}


def _classify_diff_status(details: list) -> str:
    """根据差异详情分类状态。"""
    has_user_newer = any(
        d["status"] in ("user_newer", "missing_in_project") for d in details
    )
    has_project_newer = any(
        d["status"] in ("project_newer", "missing_in_user") for d in details
    )
    has_both_modified = any(d["status"] == "both_modified" for d in details)

    if has_both_modified or (has_user_newer and has_project_newer):
        return "both_modified"
    if has_project_newer:
        return "project_newer"
    if has_user_newer:
        return "user_newer"
    return "identical"


def _diff_summary(file_a: str, file_b: str) -> str:
    """生成两个文件的简要 diff 摘要。"""
    if not shutil.which("diff"):
        return "files differ (diff not available)"

    try:
        result_stat = subprocess.run(
            ["diff", file_a, file_b],
            capture_output=True, text=True, timeout=5,
        )
        diff_lines = [l for l in result_stat.stdout.splitlines()
                      if l.startswith(("<", ">"))]
        return f"{len(diff_lines)} lines differ"
    except (subprocess.TimeoutExpired, FileNotFoundError,
            subprocess.SubprocessError):
        return "files differ"


def _format_diff_summary_human(diff_result: dict) -> str:
    """生成人类可读的差异摘要。"""
    if diff_result["identical"]:
        return "All files identical"

    parts = []
    for d in diff_result["details"]:
        if d["status"] == "project_newer":
            parts.append(f"[project_newer] {d['file']} ({d['project_mtime']}, {d.get('diff_summary', '')})")
        elif d["status"] == "user_newer":
            parts.append(f"[user_newer] {d['file']} ({d['user_mtime']}, {d.get('diff_summary', '')})")
        elif d["status"] == "missing_in_user":
            parts.append(f"[project_only] {d['file']}")
        elif d["status"] == "missing_in_project":
            parts.append(f"[user_only] {d['file']}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="claude_session",
    toolset="claude_session",
    schema=CLAUDE_SESSION_SCHEMA,
    handler=_handle_claude_session,
    check_fn=_check_claude_session,
    emoji="🤖",
    max_result_size_chars=200_000,
)
