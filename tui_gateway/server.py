import atexit
import concurrent.futures
import copy
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from hermes_constants import get_hermes_home
from hermes_cli.env_loader import load_hermes_dotenv

_hermes_home = get_hermes_home()
load_hermes_dotenv(hermes_home=_hermes_home, project_env=Path(__file__).parent.parent / ".env")

try:
    from hermes_cli.banner import prefetch_update_check
    prefetch_update_check()
except Exception:
    pass

from tui_gateway.render import make_stream_renderer, render_diff, render_message

_sessions: dict[str, dict] = {}
_methods: dict[str, callable] = {}
_pending: dict[str, tuple[str, threading.Event]] = {}
_answers: dict[str, str] = {}
_db = None
_stdout_lock = threading.Lock()
_cfg_lock = threading.Lock()
_cfg_cache: dict | None = None
_cfg_mtime: float | None = None
_SLASH_WORKER_TIMEOUT_S = max(5.0, float(os.environ.get("HERMES_TUI_SLASH_TIMEOUT_S", "45") or 45))

# ── Async RPC dispatch (#12546) ──────────────────────────────────────
# A handful of handlers block the dispatcher loop in entry.py for seconds
# to minutes (slash.exec, cli.exec, shell.exec, session.resume,
# session.branch, skills.manage).  While they're running, inbound RPCs —
# notably approval.respond and session.interrupt — sit unread in the
# stdin pipe.  We route only those slow handlers onto a small thread pool;
# everything else stays on the main thread so ordering stays sane for the
# fast path.  write_json is already _stdout_lock-guarded, so concurrent
# response writes are safe.
_LONG_HANDLERS = frozenset(
    {
        "cli.exec",
        "session.branch",
        "session.resume",
        "shell.exec",
        "skills.manage",
        "slash.exec",
    }
)

_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=max(2, int(os.environ.get("HERMES_TUI_RPC_POOL_WORKERS", "4") or 4)),
    thread_name_prefix="tui-rpc",
)
atexit.register(lambda: _pool.shutdown(wait=False, cancel_futures=True))

# Reserve real stdout for JSON-RPC only; redirect Python's stdout to stderr
# so stray print() from libraries/tools becomes harmless gateway.stderr instead
# of corrupting the JSON protocol.
_real_stdout = sys.stdout
sys.stdout = sys.stderr


class _SlashWorker:
    """Persistent HermesCLI subprocess for slash commands."""

    def __init__(self, session_key: str, model: str):
        self._lock = threading.Lock()
        self._seq = 0
        self.stderr_tail: list[str] = []
        self.stdout_queue: queue.Queue[dict | None] = queue.Queue()

        argv = [sys.executable, "-m", "tui_gateway.slash_worker", "--session-key", session_key]
        if model:
            argv += ["--model", model]

        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, cwd=os.getcwd(), env=os.environ.copy(),
        )
        threading.Thread(target=self._drain_stdout, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stdout(self):
        for line in (self.proc.stdout or []):
            try:
                self.stdout_queue.put(json.loads(line))
            except json.JSONDecodeError:
                continue
        self.stdout_queue.put(None)

    def _drain_stderr(self):
        for line in (self.proc.stderr or []):
            if text := line.rstrip("\n"):
                self.stderr_tail = (self.stderr_tail + [text])[-80:]

    def run(self, command: str) -> str:
        if self.proc.poll() is not None:
            raise RuntimeError("slash worker exited")

        with self._lock:
            self._seq += 1
            rid = self._seq
            self.proc.stdin.write(json.dumps({"id": rid, "command": command}) + "\n")
            self.proc.stdin.flush()

            while True:
                try:
                    msg = self.stdout_queue.get(timeout=_SLASH_WORKER_TIMEOUT_S)
                except queue.Empty:
                    raise RuntimeError("slash worker timed out")
                if msg is None:
                    break
                if msg.get("id") != rid:
                    continue
                if not msg.get("ok"):
                    raise RuntimeError(msg.get("error", "slash worker failed"))
                return str(msg.get("output", "")).rstrip()

            raise RuntimeError(f"slash worker closed pipe{': ' + chr(10).join(self.stderr_tail[-8:]) if self.stderr_tail else ''}")

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait(timeout=1)
        except Exception:
            try: self.proc.kill()
            except Exception: pass


atexit.register(lambda: [
    s.get("slash_worker") and s["slash_worker"].close()
    for s in _sessions.values()
])


# ── Plumbing ──────────────────────────────────────────────────────────

def _get_db():
    global _db
    if _db is None:
        from hermes_state import SessionDB
        _db = SessionDB()
    return _db


def write_json(obj: dict) -> bool:
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    try:
        with _stdout_lock:
            _real_stdout.write(line)
            _real_stdout.flush()
        return True
    except BrokenPipeError:
        return False


def _emit(event: str, sid: str, payload: dict | None = None):
    params = {"type": event, "session_id": sid}
    if payload is not None:
        params["payload"] = payload
    write_json({"jsonrpc": "2.0", "method": "event", "params": params})


def _status_update(sid: str, kind: str, text: str | None = None):
    body = (text if text is not None else kind).strip()
    if not body:
        return
    _emit("status.update", sid, {"kind": kind if text is not None else "status", "text": body})


def _estimate_image_tokens(width: int, height: int) -> int:
    """Very rough UI estimate for image prompt cost.

    Uses 512px tiles at ~85 tokens/tile as a lightweight cross-provider hint.
    This is intentionally approximate and only used for attachment display.
    """
    if width <= 0 or height <= 0:
        return 0
    return max(1, (width + 511) // 512) * max(1, (height + 511) // 512) * 85


def _image_meta(path: Path) -> dict:
    meta = {"name": path.name}
    try:
        from PIL import Image

        with Image.open(path) as img:
            width, height = img.size
        meta["width"] = int(width)
        meta["height"] = int(height)
        meta["token_estimate"] = _estimate_image_tokens(int(width), int(height))
    except Exception:
        pass
    return meta


def _ok(rid, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _err(rid, code: int, msg: str) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}


def method(name: str):
    def dec(fn):
        _methods[name] = fn
        return fn
    return dec


def handle_request(req: dict) -> dict | None:
    fn = _methods.get(req.get("method", ""))
    if not fn:
        return _err(req.get("id"), -32601, f"unknown method: {req.get('method')}")
    return fn(req.get("id"), req.get("params", {}))


def dispatch(req: dict) -> dict | None:
    """Route inbound RPCs — long handlers to the pool, everything else inline.

    Returns a response dict when handled inline. Returns None when the
    handler was scheduled on the pool; the worker writes its own
    response via write_json when done.
    """
    if req.get("method") not in _LONG_HANDLERS:
        return handle_request(req)

    def run():
        try:
            resp = handle_request(req)
        except Exception as exc:
            resp = _err(req.get("id"), -32000, f"handler error: {exc}")
        if resp is not None:
            write_json(resp)

    _pool.submit(run)

    return None


def _wait_agent(session: dict, rid: str, timeout: float = 30.0) -> dict | None:
    ready = session.get("agent_ready")
    if ready is not None and not ready.wait(timeout=timeout):
        return _err(rid, 5032, "agent initialization timed out")
    err = session.get("agent_error")
    return _err(rid, 5032, err) if err else None


def _sess_nowait(params, rid):
    s = _sessions.get(params.get("session_id") or "")
    return (s, None) if s else (None, _err(rid, 4001, "session not found"))


def _sess(params, rid):
    s, err = _sess_nowait(params, rid)
    return (None, err) if err else (s, _wait_agent(s, rid))


def _normalize_completion_path(path_part: str) -> str:
    expanded = os.path.expanduser(path_part)
    if os.name != "nt":
        normalized = expanded.replace("\\", "/")
        if len(normalized) >= 3 and normalized[1] == ":" and normalized[2] == "/" and normalized[0].isalpha():
            return f"/mnt/{normalized[0].lower()}/{normalized[3:]}"
    return expanded


# ── Config I/O ────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    global _cfg_cache, _cfg_mtime
    try:
        import yaml
        p = _hermes_home / "config.yaml"
        mtime = p.stat().st_mtime if p.exists() else None
        with _cfg_lock:
            if _cfg_cache is not None and _cfg_mtime == mtime:
                return copy.deepcopy(_cfg_cache)
        if p.exists():
            with open(p) as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}
        with _cfg_lock:
            _cfg_cache = copy.deepcopy(data)
            _cfg_mtime = mtime
        return data
    except Exception:
        pass
    return {}


def _save_cfg(cfg: dict):
    global _cfg_cache, _cfg_mtime
    import yaml
    path = _hermes_home / "config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    with _cfg_lock:
        _cfg_cache = copy.deepcopy(cfg)
        try:
            _cfg_mtime = path.stat().st_mtime
        except Exception:
            _cfg_mtime = None


def _set_session_context(session_key: str) -> list:
    try:
        from gateway.session_context import set_session_vars
        return set_session_vars(session_key=session_key)
    except Exception:
        return []


def _clear_session_context(tokens: list) -> None:
    if not tokens:
        return
    try:
        from gateway.session_context import clear_session_vars
        clear_session_vars(tokens)
    except Exception:
        pass


def _enable_gateway_prompts() -> None:
    """Route approvals through gateway callbacks instead of CLI input()."""
    os.environ["HERMES_GATEWAY_SESSION"] = "1"
    os.environ["HERMES_EXEC_ASK"] = "1"
    os.environ["HERMES_INTERACTIVE"] = "1"


# ── Blocking prompt factory ──────────────────────────────────────────

def _block(event: str, sid: str, payload: dict, timeout: int = 300) -> str:
    rid = uuid.uuid4().hex[:8]
    ev = threading.Event()
    _pending[rid] = (sid, ev)
    payload["request_id"] = rid
    _emit(event, sid, payload)
    ev.wait(timeout=timeout)
    _pending.pop(rid, None)
    return _answers.pop(rid, "")


def _clear_pending(sid: str | None = None) -> None:
    """Release pending prompts with an empty answer.

    When *sid* is provided, only prompts owned by that session are
    released — critical for session.interrupt, which must not
    collaterally cancel clarify/sudo/secret prompts on unrelated
    sessions sharing the same tui_gateway process.  When *sid* is
    None, every pending prompt is released (used during shutdown).
    """
    for rid, (owner_sid, ev) in list(_pending.items()):
        if sid is None or owner_sid == sid:
            _answers[rid] = ""
            ev.set()


# ── Agent factory ────────────────────────────────────────────────────

def resolve_skin() -> dict:
    try:
        from hermes_cli.skin_engine import init_skin_from_config, get_active_skin
        init_skin_from_config(_load_cfg())
        skin = get_active_skin()
        return {
            "name": skin.name,
            "colors": skin.colors,
            "branding": skin.branding,
            "banner_logo": skin.banner_logo,
            "banner_hero": skin.banner_hero,
            "tool_prefix": skin.tool_prefix,
            "help_header": (skin.branding or {}).get("help_header", ""),
        }
    except Exception:
        return {}


def _resolve_model() -> str:
    env = os.environ.get("HERMES_MODEL", "")
    if env:
        return env
    m = _load_cfg().get("model", "")
    if isinstance(m, dict):
        return m.get("default", "")
    if isinstance(m, str) and m:
        return m
    return "anthropic/claude-sonnet-4"


def _write_config_key(key_path: str, value):
    cfg = _load_cfg()
    current = cfg
    keys = key_path.split(".")
    for key in keys[:-1]:
        if key not in current or not isinstance(current.get(key), dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    _save_cfg(cfg)


def _load_reasoning_config() -> dict | None:
    from hermes_constants import parse_reasoning_effort

    effort = str(_load_cfg().get("agent", {}).get("reasoning_effort", "") or "").strip()
    return parse_reasoning_effort(effort)


def _load_service_tier() -> str | None:
    raw = str(_load_cfg().get("agent", {}).get("service_tier", "") or "").strip().lower()
    if not raw or raw in {"normal", "default", "standard", "off", "none"}:
        return None
    if raw in {"fast", "priority", "on"}:
        return "priority"
    return None


def _load_show_reasoning() -> bool:
    return bool(_load_cfg().get("display", {}).get("show_reasoning", False))


def _load_tool_progress_mode() -> str:
    raw = _load_cfg().get("display", {}).get("tool_progress", "all")
    if raw is False:
        return "off"
    if raw is True:
        return "all"
    mode = str(raw or "all").strip().lower()
    return mode if mode in {"off", "new", "all", "verbose"} else "all"


def _load_enabled_toolsets() -> list[str] | None:
    try:
        from hermes_cli.config import load_config
        from hermes_cli.tools_config import _get_platform_tools

        enabled = sorted(_get_platform_tools(load_config(), "cli", include_default_mcp_servers=False))
        return enabled or None
    except Exception:
        return None


def _session_tool_progress_mode(sid: str) -> str:
    return str(_sessions.get(sid, {}).get("tool_progress_mode", "all") or "all")


def _tool_progress_enabled(sid: str) -> bool:
    return _session_tool_progress_mode(sid) != "off"


def _restart_slash_worker(session: dict):
    worker = session.get("slash_worker")
    if worker:
        try:
            worker.close()
        except Exception:
            pass
    try:
        session["slash_worker"] = _SlashWorker(session["session_key"], getattr(session.get("agent"), "model", _resolve_model()))
    except Exception:
        session["slash_worker"] = None


def _persist_model_switch(result) -> None:
    from hermes_cli.config import save_config

    cfg = _load_cfg()
    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        model_cfg = {}
        cfg["model"] = model_cfg

    model_cfg["default"] = result.new_model
    model_cfg["provider"] = result.target_provider
    if result.base_url:
        model_cfg["base_url"] = result.base_url
    else:
        model_cfg.pop("base_url", None)
    save_config(cfg)


def _apply_model_switch(sid: str, session: dict, raw_input: str) -> dict:
    from hermes_cli.model_switch import parse_model_flags, switch_model
    from hermes_cli.runtime_provider import resolve_runtime_provider

    model_input, explicit_provider, persist_global = parse_model_flags(raw_input)
    if not model_input:
        raise ValueError("model value required")

    agent = session.get("agent")
    if agent:
        current_provider = getattr(agent, "provider", "") or ""
        current_model = getattr(agent, "model", "") or ""
        current_base_url = getattr(agent, "base_url", "") or ""
        current_api_key = getattr(agent, "api_key", "") or ""
    else:
        runtime = resolve_runtime_provider(requested=None)
        current_provider = str(runtime.get("provider", "") or "")
        current_model = _resolve_model()
        current_base_url = str(runtime.get("base_url", "") or "")
        current_api_key = str(runtime.get("api_key", "") or "")

    result = switch_model(
        raw_input=model_input,
        current_provider=current_provider,
        current_model=current_model,
        current_base_url=current_base_url,
        current_api_key=current_api_key,
        is_global=persist_global,
        explicit_provider=explicit_provider,
    )
    if not result.success:
        raise ValueError(result.error_message or "model switch failed")

    if agent:
        agent.switch_model(
            new_model=result.new_model,
            new_provider=result.target_provider,
            api_key=result.api_key,
            base_url=result.base_url,
            api_mode=result.api_mode,
        )
        _restart_slash_worker(session)
        _emit("session.info", sid, _session_info(agent))

    os.environ["HERMES_MODEL"] = result.new_model
    # Keep the process-level provider env var in sync with the user's explicit
    # choice so any ambient re-resolution (credential pool refresh, compressor
    # rebuild, aux clients) resolves to the new provider instead of the
    # original one persisted in config or env.
    if result.target_provider:
        os.environ["HERMES_INFERENCE_PROVIDER"] = result.target_provider
    if persist_global:
        _persist_model_switch(result)
    return {"value": result.new_model, "warning": result.warning_message or ""}


def _compress_session_history(session: dict, focus_topic: str | None = None) -> tuple[int, dict]:
    from agent.model_metadata import estimate_messages_tokens_rough

    agent = session["agent"]
    history = list(session.get("history", []))
    if len(history) < 4:
        return 0, _get_usage(agent)
    approx_tokens = estimate_messages_tokens_rough(history)
    compressed, _ = agent._compress_context(
        history,
        getattr(agent, "_cached_system_prompt", "") or "",
        approx_tokens=approx_tokens,
        focus_topic=focus_topic or None,
    )
    session["history"] = compressed
    session["history_version"] = int(session.get("history_version", 0)) + 1
    return len(history) - len(compressed), _get_usage(agent)


def _get_usage(agent) -> dict:
    g = lambda k, fb=None: getattr(agent, k, 0) or (getattr(agent, fb, 0) if fb else 0)
    usage = {
        "model": getattr(agent, "model", "") or "",
        "input": g("session_input_tokens", "session_prompt_tokens"),
        "output": g("session_output_tokens", "session_completion_tokens"),
        "cache_read": g("session_cache_read_tokens"),
        "cache_write": g("session_cache_write_tokens"),
        "prompt": g("session_prompt_tokens"),
        "completion": g("session_completion_tokens"),
        "total": g("session_total_tokens"),
        "calls": g("session_api_calls"),
    }
    comp = getattr(agent, "context_compressor", None)
    if comp:
        ctx_used = getattr(comp, "last_prompt_tokens", 0) or usage["total"] or 0
        ctx_max = getattr(comp, "context_length", 0) or 0
        if ctx_max:
            usage["context_used"] = ctx_used
            usage["context_max"] = ctx_max
            usage["context_percent"] = max(0, min(100, round(ctx_used / ctx_max * 100)))
        usage["compressions"] = getattr(comp, "compression_count", 0) or 0
    try:
        from agent.usage_pricing import CanonicalUsage, estimate_usage_cost
        cost = estimate_usage_cost(
            usage["model"],
            CanonicalUsage(
                input_tokens=usage["input"],
                output_tokens=usage["output"],
                cache_read_tokens=usage["cache_read"],
                cache_write_tokens=usage["cache_write"],
            ),
            provider=getattr(agent, "provider", None),
            base_url=getattr(agent, "base_url", None),
        )
        usage["cost_status"] = cost.status
        if cost.amount_usd is not None:
            usage["cost_usd"] = float(cost.amount_usd)
    except Exception:
        pass
    return usage


def _probe_credentials(agent) -> str:
    """Light credential check at session creation — returns warning or ''."""
    try:
        key = getattr(agent, "api_key", "") or ""
        provider = getattr(agent, "provider", "") or ""
        if not key or key == "no-key-required":
            return f"No API key configured for provider '{provider}'. First message will fail."
    except Exception:
        pass
    return ""


def _session_info(agent) -> dict:
    info: dict = {
        "model": getattr(agent, "model", ""),
        "tools": {},
        "skills": {},
        "cwd": os.getcwd(),
        "version": "",
        "release_date": "",
        "update_behind": None,
        "update_command": "",
        "usage": _get_usage(agent),
    }
    try:
        from hermes_cli import __version__, __release_date__
        info["version"] = __version__
        info["release_date"] = __release_date__
    except Exception:
        pass
    try:
        from model_tools import get_toolset_for_tool
        for t in getattr(agent, "tools", []) or []:
            name = t["function"]["name"]
            info["tools"].setdefault(get_toolset_for_tool(name) or "other", []).append(name)
    except Exception:
        pass
    try:
        from hermes_cli.banner import get_available_skills
        info["skills"] = get_available_skills()
    except Exception:
        pass
    try:
        from tools.mcp_tool import get_mcp_status
        info["mcp_servers"] = get_mcp_status()
    except Exception:
        info["mcp_servers"] = []
    try:
        from hermes_cli.banner import get_update_result
        from hermes_cli.config import recommended_update_command
        info["update_behind"] = get_update_result(timeout=0.5)
        info["update_command"] = recommended_update_command()
    except Exception:
        pass
    return info


def _tool_ctx(name: str, args: dict) -> str:
    try:
        from agent.display import build_tool_preview
        return build_tool_preview(name, args, max_len=80) or ""
    except Exception:
        return ""


def _fmt_tool_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{round(seconds)}s"
    mins, secs = divmod(int(round(seconds)), 60)
    return f"{mins}m {secs}s" if secs else f"{mins}m"


def _count_list(obj: object, *path: str) -> int | None:
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return len(cur) if isinstance(cur, list) else None


def _tool_summary(name: str, result: str, duration_s: float | None) -> str | None:
    try:
        data = json.loads(result)
    except Exception:
        data = None

    dur = _fmt_tool_duration(duration_s)
    suffix = f" in {dur}" if dur else ""
    text = None

    if name == "web_search" and isinstance(data, dict):
        n = _count_list(data, "data", "web")
        if n is not None:
            text = f"Did {n} {'search' if n == 1 else 'searches'}"

    elif name == "web_extract" and isinstance(data, dict):
        n = _count_list(data, "results") or _count_list(data, "data", "results")
        if n is not None:
            text = f"Extracted {n} {'page' if n == 1 else 'pages'}"

    return f"{text or 'Completed'}{suffix}" if (text or dur) else None


def _on_tool_start(sid: str, tool_call_id: str, name: str, args: dict):
    session = _sessions.get(sid)
    if session is not None:
        try:
            from agent.display import capture_local_edit_snapshot

            snapshot = capture_local_edit_snapshot(name, args)
            if snapshot is not None:
                session.setdefault("edit_snapshots", {})[tool_call_id] = snapshot
        except Exception:
            pass
        session.setdefault("tool_started_at", {})[tool_call_id] = time.time()
    if _tool_progress_enabled(sid):
        _emit("tool.start", sid, {"tool_id": tool_call_id, "name": name, "context": _tool_ctx(name, args)})


def _on_tool_complete(sid: str, tool_call_id: str, name: str, args: dict, result: str):
    payload = {"tool_id": tool_call_id, "name": name}
    session = _sessions.get(sid)
    snapshot = None
    started_at = None
    if session is not None:
        snapshot = session.setdefault("edit_snapshots", {}).pop(tool_call_id, None)
        started_at = session.setdefault("tool_started_at", {}).pop(tool_call_id, None)
    duration_s = time.time() - started_at if started_at else None
    if duration_s is not None:
        payload["duration_s"] = duration_s
    summary = _tool_summary(name, result, duration_s)
    if summary:
        payload["summary"] = summary
    try:
        from agent.display import render_edit_diff_with_delta

        rendered: list[str] = []
        if render_edit_diff_with_delta(name, result, function_args=args, snapshot=snapshot, print_fn=rendered.append):
            payload["inline_diff"] = "\n".join(rendered)
    except Exception:
        pass
    if _tool_progress_enabled(sid) or payload.get("inline_diff"):
        _emit("tool.complete", sid, payload)


def _on_tool_progress(
    sid: str,
    event_type: str,
    name: str | None = None,
    preview: str | None = None,
    _args: dict | None = None,
    **_kwargs,
):
    if not _tool_progress_enabled(sid):
        return
    if event_type == "tool.started" and name:
        _emit("tool.progress", sid, {"name": name, "preview": preview or ""})
        return
    if event_type == "reasoning.available" and preview:
        _emit("reasoning.available", sid, {"text": str(preview)})
        return
    if event_type.startswith("subagent."):
        payload = {
            "goal": str(_kwargs.get("goal") or ""),
            "task_count": int(_kwargs.get("task_count") or 1),
            "task_index": int(_kwargs.get("task_index") or 0),
        }
        if name:
            payload["tool_name"] = str(name)
        if preview:
            payload["text"] = str(preview)
        if _kwargs.get("status"):
            payload["status"] = str(_kwargs["status"])
        if _kwargs.get("summary"):
            payload["summary"] = str(_kwargs["summary"])
        if _kwargs.get("duration_seconds") is not None:
            payload["duration_seconds"] = float(_kwargs["duration_seconds"])
        if preview and event_type == "subagent.tool":
            payload["tool_preview"] = str(preview)
            payload["text"] = str(preview)
        _emit(event_type, sid, payload)


def _agent_cbs(sid: str) -> dict:
    return dict(
        tool_start_callback=lambda tc_id, name, args: _on_tool_start(sid, tc_id, name, args),
        tool_complete_callback=lambda tc_id, name, args, result: _on_tool_complete(sid, tc_id, name, args, result),
        tool_progress_callback=lambda event_type, name=None, preview=None, args=None, **kwargs: _on_tool_progress(
            sid, event_type, name, preview, args, **kwargs
        ),
        tool_gen_callback=lambda name: _tool_progress_enabled(sid) and _emit("tool.generating", sid, {"name": name}),
        thinking_callback=lambda text: _emit("thinking.delta", sid, {"text": text}),
        reasoning_callback=lambda text: _emit("reasoning.delta", sid, {"text": text}),
        status_callback=lambda kind, text=None: _status_update(sid, str(kind), None if text is None else str(text)),
        clarify_callback=lambda q, c: _block("clarify.request", sid, {"question": q, "choices": c}),
    )


def _wire_callbacks(sid: str):
    from tools.terminal_tool import set_sudo_password_callback
    from tools.skills_tool import set_secret_capture_callback

    set_sudo_password_callback(lambda: _block("sudo.request", sid, {}, timeout=120))

    def secret_cb(env_var, prompt, metadata=None):
        pl = {"prompt": prompt, "env_var": env_var}
        if metadata:
            pl["metadata"] = metadata
        val = _block("secret.request", sid, pl)
        if not val:
            return {"success": True, "stored_as": env_var, "validated": False, "skipped": True, "message": "skipped"}
        from hermes_cli.config import save_env_value_secure
        return {**save_env_value_secure(env_var, val), "skipped": False, "message": "ok"}

    set_secret_capture_callback(secret_cb)


def _resolve_personality_prompt(cfg: dict) -> str:
    """Resolve the active personality into a system prompt string."""
    name = (cfg.get("display", {}).get("personality", "") or "").strip().lower()
    if not name or name in ("default", "none", "neutral"):
        return ""
    try:
        from cli import load_cli_config

        personalities = load_cli_config().get("agent", {}).get("personalities", {})
    except Exception:
        try:
            from hermes_cli.config import load_config as _load_full_cfg

            personalities = _load_full_cfg().get("agent", {}).get("personalities", {})
        except Exception:
            personalities = cfg.get("agent", {}).get("personalities", {})
    pval = personalities.get(name)
    if pval is None:
        return ""
    return _render_personality_prompt(pval)


def _render_personality_prompt(value) -> str:
    if isinstance(value, dict):
        parts = [value.get("system_prompt", "")]
        if value.get("tone"):
            parts.append(f'Tone: {value["tone"]}')
        if value.get("style"):
            parts.append(f'Style: {value["style"]}')
        return "\n".join(p for p in parts if p)
    return str(value)


def _available_personalities(cfg: dict | None = None) -> dict:
    try:
        from cli import load_cli_config

        return load_cli_config().get("agent", {}).get("personalities", {}) or {}
    except Exception:
        try:
            from hermes_cli.config import load_config as _load_full_cfg

            return _load_full_cfg().get("agent", {}).get("personalities", {}) or {}
        except Exception:
            cfg = cfg or _load_cfg()
            return cfg.get("agent", {}).get("personalities", {}) or {}


def _validate_personality(value: str, cfg: dict | None = None) -> tuple[str, str]:
    raw = str(value or "").strip()
    name = raw.lower()
    if not name or name in ("none", "default", "neutral"):
        return "", ""

    personalities = _available_personalities(cfg)
    if name not in personalities:
        names = sorted(personalities)
        available = ", ".join(f"`{n}`" for n in names)
        base = f"Unknown personality: `{raw}`."
        if available:
            base += f"\n\nAvailable: `none`, {available}"
        else:
            base += "\n\nNo personalities configured."
        raise ValueError(base)

    return name, _render_personality_prompt(personalities[name])


def _apply_personality_to_session(sid: str, session: dict, new_prompt: str) -> tuple[bool, dict | None]:
    if not session:
        return False, None

    try:
        info = _reset_session_agent(sid, session)
        return True, info
    except Exception:
        if session.get("agent"):
            agent = session["agent"]
            agent.ephemeral_system_prompt = new_prompt or None
            agent._cached_system_prompt = None
            info = _session_info(agent)
            _emit("session.info", sid, info)
            return False, info
        return False, None


def _background_agent_kwargs(agent, task_id: str) -> dict:
    cfg = _load_cfg()

    return {
        "base_url": getattr(agent, "base_url", None) or None,
        "api_key": getattr(agent, "api_key", None) or None,
        "provider": getattr(agent, "provider", None) or None,
        "api_mode": getattr(agent, "api_mode", None) or None,
        "acp_command": getattr(agent, "acp_command", None) or None,
        "acp_args": getattr(agent, "acp_args", None) or None,
        "model": getattr(agent, "model", None) or _resolve_model(),
        "max_iterations": int(cfg.get("max_turns", 25) or 25),
        "enabled_toolsets": getattr(agent, "enabled_toolsets", None) or _load_enabled_toolsets(),
        "quiet_mode": True,
        "verbose_logging": False,
        "ephemeral_system_prompt": getattr(agent, "ephemeral_system_prompt", None) or None,
        "providers_allowed": getattr(agent, "providers_allowed", None),
        "providers_ignored": getattr(agent, "providers_ignored", None),
        "providers_order": getattr(agent, "providers_order", None),
        "provider_sort": getattr(agent, "provider_sort", None),
        "provider_require_parameters": getattr(agent, "provider_require_parameters", False),
        "provider_data_collection": getattr(agent, "provider_data_collection", None),
        "session_id": task_id,
        "reasoning_config": getattr(agent, "reasoning_config", None) or _load_reasoning_config(),
        "service_tier": getattr(agent, "service_tier", None) or _load_service_tier(),
        "request_overrides": dict(getattr(agent, "request_overrides", {}) or {}),
        "platform": "tui",
        "session_db": _get_db(),
        "fallback_model": getattr(agent, "_fallback_model", None),
    }


def _reset_session_agent(sid: str, session: dict) -> dict:
    tokens = _set_session_context(session["session_key"])
    try:
        new_agent = _make_agent(sid, session["session_key"], session_id=session["session_key"])
    finally:
        _clear_session_context(tokens)
    session["agent"] = new_agent
    session["attached_images"] = []
    session["edit_snapshots"] = {}
    session["image_counter"] = 0
    session["running"] = False
    session["show_reasoning"] = _load_show_reasoning()
    session["tool_progress_mode"] = _load_tool_progress_mode()
    session["tool_started_at"] = {}
    with session["history_lock"]:
        session["history"] = []
        session["history_version"] = int(session.get("history_version", 0)) + 1
    info = _session_info(new_agent)
    _emit("session.info", sid, info)
    _restart_slash_worker(session)
    return info


def _make_agent(sid: str, key: str, session_id: str | None = None):
    from run_agent import AIAgent
    cfg = _load_cfg()
    system_prompt = cfg.get("agent", {}).get("system_prompt", "") or ""
    if not system_prompt:
        system_prompt = _resolve_personality_prompt(cfg)
    return AIAgent(
        model=_resolve_model(),
        quiet_mode=True,
        verbose_logging=_load_tool_progress_mode() == "verbose",
        reasoning_config=_load_reasoning_config(),
        service_tier=_load_service_tier(),
        enabled_toolsets=_load_enabled_toolsets(),
        platform="tui",
        session_id=session_id or key, session_db=_get_db(),
        ephemeral_system_prompt=system_prompt or None,
        **_agent_cbs(sid),
    )


def _init_session(sid: str, key: str, agent, history: list, cols: int = 80):
    _sessions[sid] = {
        "agent": agent,
        "session_key": key,
        "history": history,
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": cols,
        "slash_worker": None,
        "show_reasoning": _load_show_reasoning(),
        "tool_progress_mode": _load_tool_progress_mode(),
        "edit_snapshots": {},
        "tool_started_at": {},
    }
    try:
        _sessions[sid]["slash_worker"] = _SlashWorker(key, getattr(agent, "model", _resolve_model()))
    except Exception:
        # Defer hard-failure to slash.exec; chat still works without slash worker.
        _sessions[sid]["slash_worker"] = None
    try:
        from tools.approval import register_gateway_notify, load_permanent_allowlist
        register_gateway_notify(key, lambda data: _emit("approval.request", sid, data))
        load_permanent_allowlist()
    except Exception:
        pass
    _wire_callbacks(sid)
    _emit("session.info", sid, _session_info(agent))


def _new_session_key() -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _with_checkpoints(session, fn):
    return fn(session["agent"]._checkpoint_mgr, os.getenv("TERMINAL_CWD", os.getcwd()))


def _resolve_checkpoint_hash(mgr, cwd: str, ref: str) -> str:
    try:
        checkpoints = mgr.list_checkpoints(cwd)
        idx = int(ref) - 1
    except ValueError:
        return ref
    if 0 <= idx < len(checkpoints):
        return checkpoints[idx].get("hash", ref)
    raise ValueError(f"Invalid checkpoint number. Use 1-{len(checkpoints)}.")


def _enrich_with_attached_images(user_text: str, image_paths: list[str]) -> str:
    """Pre-analyze attached images via vision and prepend descriptions to user text."""
    import asyncio, json as _json
    from tools.vision_tools import vision_analyze_tool

    prompt = (
        "Describe everything visible in this image in thorough detail. "
        "Include any text, code, data, objects, people, layout, colors, "
        "and any other notable visual information."
    )

    parts: list[str] = []
    for path in image_paths:
        p = Path(path)
        if not p.exists():
            continue
        hint = f"[You can examine it with vision_analyze using image_url: {p}]"
        try:
            r = _json.loads(asyncio.run(vision_analyze_tool(image_url=str(p), user_prompt=prompt)))
            desc = r.get("analysis", "") if r.get("success") else None
            parts.append(f"[The user attached an image:\n{desc}]\n{hint}" if desc
                         else f"[The user attached an image but analysis failed.]\n{hint}")
        except Exception:
            parts.append(f"[The user attached an image but analysis failed.]\n{hint}")

    text = user_text or ""
    prefix = "\n\n".join(parts)
    if prefix:
        return f"{prefix}\n\n{text}" if text else prefix
    return text or "What do you see in this image?"


def _history_to_messages(history: list[dict]) -> list[dict]:
    messages = []
    tool_call_args = {}

    for m in history:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in ("user", "assistant", "tool", "system"):
            continue
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                tc_id = tc.get("id", "")
                if tc_id and fn.get("name"):
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    tool_call_args[tc_id] = (fn["name"], args)
            if not (m.get("content") or "").strip():
                continue
        if role == "tool":
            tc_id = m.get("tool_call_id", "")
            tc_info = tool_call_args.get(tc_id) if tc_id else None
            name = (tc_info[0] if tc_info else None) or m.get("tool_name") or "tool"
            args = (tc_info[1] if tc_info else None) or {}
            messages.append({"role": "tool", "name": name, "context": _tool_ctx(name, args)})
            continue
        if not (m.get("content") or "").strip():
            continue
        messages.append({"role": role, "text": m.get("content") or ""})

    return messages


# ── Methods: session ─────────────────────────────────────────────────

@method("session.create")
def _(rid, params: dict) -> dict:
    sid = uuid.uuid4().hex[:8]
    key = _new_session_key()
    cols = int(params.get("cols", 80))
    _enable_gateway_prompts()

    ready = threading.Event()

    _sessions[sid] = {
        "agent": None,
        "agent_error": None,
        "agent_ready": ready,
        "attached_images": [],
        "cols": cols,
        "edit_snapshots": {},
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "image_counter": 0,
        "running": False,
        "session_key": key,
        "show_reasoning": _load_show_reasoning(),
        "slash_worker": None,
        "tool_progress_mode": _load_tool_progress_mode(),
        "tool_started_at": {},
    }

    def _build() -> None:
        session = _sessions.get(sid)
        if session is None:
            # session.close ran before the build thread got scheduled.
            ready.set()
            return

        # Track what we allocate so we can clean up if session.close
        # races us to the finish line.  session.close pops _sessions[sid]
        # unconditionally and tries to close the slash_worker it finds;
        # if _build is still mid-construction when close runs, close
        # finds slash_worker=None / notify unregistered and returns
        # cleanly — leaving us, the build thread, to later install the
        # worker + notify on an orphaned session dict.  The finally
        # block below detects the orphan and cleans up instead of
        # leaking a subprocess and a global notify registration.
        worker = None
        notify_registered = False
        try:
            tokens = _set_session_context(key)
            try:
                agent = _make_agent(sid, key)
            finally:
                _clear_session_context(tokens)

            _get_db().create_session(key, source="tui", model=_resolve_model())
            session["agent"] = agent

            try:
                worker = _SlashWorker(key, getattr(agent, "model", _resolve_model()))
                session["slash_worker"] = worker
            except Exception:
                pass

            try:
                from tools.approval import register_gateway_notify, load_permanent_allowlist
                register_gateway_notify(key, lambda data: _emit("approval.request", sid, data))
                notify_registered = True
                load_permanent_allowlist()
            except Exception:
                pass

            _wire_callbacks(sid)

            info = _session_info(agent)
            warn = _probe_credentials(agent)
            if warn:
                info["credential_warning"] = warn
            _emit("session.info", sid, info)
        except Exception as e:
            session["agent_error"] = str(e)
            _emit("error", sid, {"message": f"agent init failed: {e}"})
        finally:
            # Orphan check: if session.close raced us and popped
            # _sessions[sid] while we were building, the dict we just
            # populated is unreachable.  Clean up the subprocess and
            # the global notify registration ourselves — session.close
            # couldn't see them at the time it ran.
            if _sessions.get(sid) is not session:
                if worker is not None:
                    try:
                        worker.close()
                    except Exception:
                        pass
                if notify_registered:
                    try:
                        from tools.approval import unregister_gateway_notify
                        unregister_gateway_notify(key)
                    except Exception:
                        pass
            ready.set()

    threading.Thread(target=_build, daemon=True).start()

    return _ok(rid, {
        "session_id": sid,
        "info": {
            "model": _resolve_model(),
            "tools": {},
            "skills": {},
            "cwd": os.getenv("TERMINAL_CWD", os.getcwd()),
        },
    })


@method("session.list")
def _(rid, params: dict) -> dict:
    try:
        # Show sessions from every adapter — users resume telegram/discord/etc
        # sessions by pasting the id directly, so the picker should surface them
        # too.  Children (subagents/compression runs) stay filtered out via the
        # hermes_state default.
        limit = params.get("limit", 20)
        rows = _get_db().list_sessions_rich(source=None, limit=limit)
        return _ok(rid, {"sessions": [
            {"id": s["id"], "title": s.get("title") or "", "preview": s.get("preview") or "",
             "started_at": s.get("started_at") or 0, "message_count": s.get("message_count") or 0,
             "source": s.get("source") or ""}
            for s in rows
        ]})
    except Exception as e:
        return _err(rid, 5006, str(e))


@method("session.resume")
def _(rid, params: dict) -> dict:
    target = params.get("session_id", "")
    if not target:
        return _err(rid, 4006, "session_id required")
    db = _get_db()
    found = db.get_session(target)
    if not found:
        found = db.get_session_by_title(target)
        if found:
            target = found["id"]
        else:
            return _err(rid, 4007, "session not found")
    sid = uuid.uuid4().hex[:8]
    _enable_gateway_prompts()
    try:
        db.reopen_session(target)
        history = db.get_messages_as_conversation(target)
        messages = _history_to_messages(history)
        tokens = _set_session_context(target)
        try:
            agent = _make_agent(sid, target, session_id=target)
        finally:
            _clear_session_context(tokens)
        _init_session(sid, target, agent, history, cols=int(params.get("cols", 80)))
    except Exception as e:
        return _err(rid, 5000, f"resume failed: {e}")
    return _ok(
        rid,
        {
            "session_id": sid,
            "resumed": target,
            "message_count": len(messages),
            "messages": messages,
            "info": _session_info(agent),
        },
    )


@method("session.title")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    title, key = params.get("title", ""), session["session_key"]
    if not title:
        return _ok(rid, {"title": _get_db().get_session_title(key) or "", "session_key": key})
    try:
        _get_db().set_session_title(key, title)
        return _ok(rid, {"title": title})
    except Exception as e:
        return _err(rid, 5007, str(e))


@method("session.usage")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    return err or _ok(rid, _get_usage(session["agent"]))


@method("session.history")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    return err or _ok(
        rid,
        {
            "count": len(session.get("history", [])),
            "messages": _history_to_messages(list(session.get("history", []))),
        },
    )


@method("session.undo")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    # Reject during an in-flight turn.  If we mutated history while
    # the agent thread is running, prompt.submit's post-run history
    # write would either clobber the undo (version matches) or
    # silently drop the agent's output (version mismatch, see below).
    # Neither is what the user wants — make them /interrupt first.
    if session.get("running"):
        return _err(rid, 4009, "session busy — /interrupt the current turn before /undo")
    removed = 0
    with session["history_lock"]:
        history = session.get("history", [])
        while history and history[-1].get("role") in ("assistant", "tool"):
            history.pop()
            removed += 1
        if history and history[-1].get("role") == "user":
            history.pop()
            removed += 1
        if removed:
            session["history_version"] = int(session.get("history_version", 0)) + 1
    return _ok(rid, {"removed": removed})


@method("session.compress")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    if session.get("running"):
        return _err(rid, 4009, "session busy — /interrupt the current turn before /compress")
    try:
        with session["history_lock"]:
            removed, usage = _compress_session_history(session, str(params.get("focus_topic", "") or "").strip())
            messages = list(session.get("history", []))
        info = _session_info(session["agent"])
        _emit("session.info", params.get("session_id", ""), info)
        return _ok(rid, {"status": "compressed", "removed": removed, "usage": usage, "info": info, "messages": messages})
    except Exception as e:
        return _err(rid, 5005, str(e))


@method("session.save")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    import time as _time
    filename = os.path.abspath(f"hermes_conversation_{_time.strftime('%Y%m%d_%H%M%S')}.json")
    try:
        with open(filename, "w") as f:
            json.dump({"model": getattr(session["agent"], "model", ""), "messages": session.get("history", [])},
                      f, indent=2, ensure_ascii=False)
        return _ok(rid, {"file": filename})
    except Exception as e:
        return _err(rid, 5011, str(e))


@method("session.close")
def _(rid, params: dict) -> dict:
    sid = params.get("session_id", "")
    session = _sessions.pop(sid, None)
    if not session:
        return _ok(rid, {"closed": False})
    try:
        from tools.approval import unregister_gateway_notify

        unregister_gateway_notify(session["session_key"])
    except Exception:
        pass
    try:
        worker = session.get("slash_worker")
        if worker:
            worker.close()
    except Exception:
        pass
    return _ok(rid, {"closed": True})


@method("session.branch")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    db = _get_db()
    old_key = session["session_key"]
    with session["history_lock"]:
        history = [dict(msg) for msg in session.get("history", [])]
    if not history:
        return _err(rid, 4008, "nothing to branch — send a message first")
    new_key = _new_session_key()
    branch_name = params.get("name", "")
    try:
        if branch_name:
            title = branch_name
        else:
            current = db.get_session_title(old_key) or "branch"
            title = db.get_next_title_in_lineage(current) if hasattr(db, "get_next_title_in_lineage") else f"{current} (branch)"
        db.create_session(new_key, source="tui", model=_resolve_model(), parent_session_id=old_key)
        for msg in history:
            db.append_message(session_id=new_key, role=msg.get("role", "user"), content=msg.get("content"))
        db.set_session_title(new_key, title)
    except Exception as e:
        return _err(rid, 5008, f"branch failed: {e}")
    new_sid = uuid.uuid4().hex[:8]
    try:
        tokens = _set_session_context(new_key)
        try:
            agent = _make_agent(new_sid, new_key, session_id=new_key)
        finally:
            _clear_session_context(tokens)
        _init_session(new_sid, new_key, agent, list(history), cols=session.get("cols", 80))
    except Exception as e:
        return _err(rid, 5000, f"agent init failed on branch: {e}")
    return _ok(rid, {"session_id": new_sid, "title": title, "parent": old_key})


@method("session.interrupt")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    if hasattr(session["agent"], "interrupt"):
        session["agent"].interrupt()
    # Scope the pending-prompt release to THIS session.  A global
    # _clear_pending() would collaterally cancel clarify/sudo/secret
    # prompts on unrelated sessions sharing the same tui_gateway
    # process, silently resolving them to empty strings.
    _clear_pending(params.get("session_id", ""))
    try:
        from tools.approval import resolve_gateway_approval
        resolve_gateway_approval(session["session_key"], "deny", resolve_all=True)
    except Exception:
        pass
    return _ok(rid, {"status": "interrupted"})


@method("session.steer")
def _(rid, params: dict) -> dict:
    """Inject a user message into the next tool result without interrupting.

    Mirrors AIAgent.steer(). Safe to call while a turn is running — the text
    lands on the last tool result of the next tool batch and the model sees
    it on its next iteration. No interrupt, no new user turn, no role
    alternation violation.
    """
    text = (params.get("text") or "").strip()
    if not text:
        return _err(rid, 4002, "text is required")
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    agent = session.get("agent")
    if agent is None or not hasattr(agent, "steer"):
        return _err(rid, 4010, "agent does not support steer")
    try:
        accepted = agent.steer(text)
    except Exception as exc:
        return _err(rid, 5000, f"steer failed: {exc}")
    return _ok(rid, {"status": "queued" if accepted else "rejected", "text": text})


@method("terminal.resize")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    session["cols"] = int(params.get("cols", 80))
    return _ok(rid, {"cols": session["cols"]})


# ── Methods: prompt ──────────────────────────────────────────────────

@method("prompt.submit")
def _(rid, params: dict) -> dict:
    sid, text = params.get("session_id", ""), params.get("text", "")
    session, err = _sess(params, rid)
    if err:
        return err
    with session["history_lock"]:
        if session.get("running"):
            return _err(rid, 4009, "session busy")
        session["running"] = True
        history = list(session["history"])
        history_version = int(session.get("history_version", 0))
        images = list(session.get("attached_images", []))
        session["attached_images"] = []
    agent = session["agent"]
    _emit("message.start", sid)

    def run():
        approval_token = None
        session_tokens = []
        try:
            from tools.approval import reset_current_session_key, set_current_session_key
            approval_token = set_current_session_key(session["session_key"])
            session_tokens = _set_session_context(session["session_key"])
            cols = session.get("cols", 80)
            streamer = make_stream_renderer(cols)
            prompt = text

            if isinstance(prompt, str) and "@" in prompt:
                from agent.context_references import preprocess_context_references
                from agent.model_metadata import get_model_context_length

                ctx_len = get_model_context_length(
                    getattr(agent, "model", "") or _resolve_model(),
                    base_url=getattr(agent, "base_url", "") or "",
                    api_key=getattr(agent, "api_key", "") or "",
                )
                ctx = preprocess_context_references(
                    prompt,
                    cwd=os.environ.get("TERMINAL_CWD", os.getcwd()),
                    allowed_root=os.environ.get("TERMINAL_CWD", os.getcwd()),
                    context_length=ctx_len,
                )
                if ctx.blocked:
                    _emit("error", sid, {"message": "\n".join(ctx.warnings) or "Context injection refused."})
                    return
                prompt = ctx.message

            prompt = _enrich_with_attached_images(prompt, images) if images else prompt

            def _stream(delta):
                payload = {"text": delta}
                if streamer and (r := streamer.feed(delta)) is not None:
                    payload["rendered"] = r
                _emit("message.delta", sid, payload)

            result = agent.run_conversation(
                prompt, conversation_history=list(history),
                stream_callback=_stream,
            )

            last_reasoning = None
            status_note = None
            if isinstance(result, dict):
                if isinstance(result.get("messages"), list):
                    with session["history_lock"]:
                        current_version = int(session.get("history_version", 0))
                        if current_version == history_version:
                            session["history"] = result["messages"]
                            session["history_version"] = history_version + 1
                        else:
                            # History mutated externally during the turn
                            # (undo/compress/retry/rollback now guard on
                            # session.running, but this is the defensive
                            # backstop for any path that slips past).
                            # Surface the desync rather than silently
                            # dropping the agent's output — the UI can
                            # show the response and warn that it was
                            # not persisted.
                            print(
                                f"[tui_gateway] prompt.submit: history_version mismatch "
                                f"(expected={history_version} current={current_version}) — "
                                f"agent output NOT written to session history",
                                file=sys.stderr,
                            )
                            status_note = (
                                "History changed during this turn — the response above is visible "
                                "but was not saved to session history."
                            )
                raw = result.get("final_response", "")
                status = "interrupted" if result.get("interrupted") else "error" if result.get("error") else "complete"
                lr = result.get("last_reasoning")
                if isinstance(lr, str) and lr.strip():
                    last_reasoning = lr.strip()
            else:
                raw = str(result)
                status = "complete"

            payload = {"text": raw, "usage": _get_usage(agent), "status": status}
            if last_reasoning:
                payload["reasoning"] = last_reasoning
            if status_note:
                payload["warning"] = status_note
            rendered = render_message(raw, cols)
            if rendered:
                payload["rendered"] = rendered
            _emit("message.complete", sid, payload)
        except Exception as e:
            _emit("error", sid, {"message": str(e)})
        finally:
            try:
                if approval_token is not None:
                    reset_current_session_key(approval_token)
            except Exception:
                pass
            _clear_session_context(session_tokens)
            with session["history_lock"]:
                session["running"] = False

    threading.Thread(target=run, daemon=True).start()
    return _ok(rid, {"status": "streaming"})


@method("clipboard.paste")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    try:
        from hermes_cli.clipboard import has_clipboard_image, save_clipboard_image
    except Exception as e:
        return _err(rid, 5027, f"clipboard unavailable: {e}")

    session["image_counter"] = session.get("image_counter", 0) + 1
    img_dir = _hermes_home / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / f"clip_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session['image_counter']}.png"

    # Save-first: mirrors CLI keybinding path; more robust than has_image() precheck
    if not save_clipboard_image(img_path):
        session["image_counter"] = max(0, session["image_counter"] - 1)
        msg = "Clipboard has image but extraction failed" if has_clipboard_image() else "No image found in clipboard"
        return _ok(rid, {"attached": False, "message": msg})

    session.setdefault("attached_images", []).append(str(img_path))
    return _ok(
        rid,
        {
            "attached": True,
            "path": str(img_path),
            "count": len(session["attached_images"]),
            **_image_meta(img_path),
        },
    )


@method("image.attach")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    raw = str(params.get("path", "") or "").strip()
    if not raw:
        return _err(rid, 4015, "path required")
    try:
        from cli import _IMAGE_EXTENSIONS, _detect_file_drop, _resolve_attachment_path, _split_path_input

        dropped = _detect_file_drop(raw)
        if dropped:
            image_path = dropped["path"]
            remainder = dropped["remainder"]
        else:
            path_token, remainder = _split_path_input(raw)
            image_path = _resolve_attachment_path(path_token)
            if image_path is None:
                return _err(rid, 4016, f"image not found: {path_token}")
        if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            return _err(rid, 4016, f"unsupported image: {image_path.name}")
        session.setdefault("attached_images", []).append(str(image_path))
        return _ok(
            rid,
            {
                "attached": True,
                "path": str(image_path),
                "count": len(session["attached_images"]),
                "remainder": remainder,
                "text": remainder or f"[User attached image: {image_path.name}]",
                **_image_meta(image_path),
            },
        )
    except Exception as e:
        return _err(rid, 5027, str(e))


@method("input.detect_drop")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    try:
        from cli import _detect_file_drop

        raw = str(params.get("text", "") or "")
        dropped = _detect_file_drop(raw)
        if not dropped:
            return _ok(rid, {"matched": False})

        drop_path = dropped["path"]
        remainder = dropped["remainder"]
        if dropped["is_image"]:
            session.setdefault("attached_images", []).append(str(drop_path))
            text = remainder or f"[User attached image: {drop_path.name}]"
            return _ok(
                rid,
                {
                    "matched": True,
                    "is_image": True,
                    "path": str(drop_path),
                    "count": len(session["attached_images"]),
                    "text": text,
                    **_image_meta(drop_path),
                },
            )

        text = f"[User attached file: {drop_path}]" + (f"\n{remainder}" if remainder else "")
        return _ok(
            rid,
            {
                "matched": True,
                "is_image": False,
                "path": str(drop_path),
                "name": drop_path.name,
                "text": text,
            },
        )
    except Exception as e:
        return _err(rid, 5027, str(e))


@method("prompt.background")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    text, parent = params.get("text", ""), params.get("session_id", "")
    if not text:
        return _err(rid, 4012, "text required")
    task_id = f"bg_{uuid.uuid4().hex[:6]}"

    def run():
        session_tokens = _set_session_context(task_id)
        try:
            from run_agent import AIAgent
            result = AIAgent(**_background_agent_kwargs(session["agent"], task_id)).run_conversation(
                user_message=text,
                task_id=task_id,
            )
            _emit("background.complete", parent, {"task_id": task_id,
                  "text": result.get("final_response", str(result)) if isinstance(result, dict) else str(result)})
        except Exception as e:
            _emit("background.complete", parent, {"task_id": task_id, "text": f"error: {e}"})
        finally:
            _clear_session_context(session_tokens)

    threading.Thread(target=run, daemon=True).start()
    return _ok(rid, {"task_id": task_id})


@method("prompt.btw")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    text, sid = params.get("text", ""), params.get("session_id", "")
    if not text:
        return _err(rid, 4012, "text required")
    snapshot = list(session.get("history", []))

    def run():
        session_tokens = _set_session_context(session["session_key"])
        try:
            from run_agent import AIAgent
            result = AIAgent(model=_resolve_model(), quiet_mode=True, platform="tui",
                             max_iterations=8, enabled_toolsets=[]).run_conversation(text, conversation_history=snapshot)
            _emit("btw.complete", sid, {"text": result.get("final_response", str(result)) if isinstance(result, dict) else str(result)})
        except Exception as e:
            _emit("btw.complete", sid, {"text": f"error: {e}"})
        finally:
            _clear_session_context(session_tokens)

    threading.Thread(target=run, daemon=True).start()
    return _ok(rid, {"status": "running"})


# ── Methods: respond ─────────────────────────────────────────────────

def _respond(rid, params, key):
    r = params.get("request_id", "")
    entry = _pending.get(r)
    if not entry:
        return _err(rid, 4009, f"no pending {key} request")
    _, ev = entry
    _answers[r] = params.get(key, "")
    ev.set()
    return _ok(rid, {"status": "ok"})


@method("clarify.respond")
def _(rid, params: dict) -> dict:
    return _respond(rid, params, "answer")

@method("sudo.respond")
def _(rid, params: dict) -> dict:
    return _respond(rid, params, "password")

@method("secret.respond")
def _(rid, params: dict) -> dict:
    return _respond(rid, params, "value")

@method("approval.respond")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    try:
        from tools.approval import resolve_gateway_approval
        return _ok(rid, {"resolved": resolve_gateway_approval(
            session["session_key"], params.get("choice", "deny"), resolve_all=params.get("all", False))})
    except Exception as e:
        return _err(rid, 5004, str(e))


# ── Methods: config ──────────────────────────────────────────────────

@method("config.set")
def _(rid, params: dict) -> dict:
    key, value = params.get("key", ""), params.get("value", "")
    session = _sessions.get(params.get("session_id", ""))

    if key == "model":
        try:
            if not value:
                return _err(rid, 4002, "model value required")
            if session:
                # Reject during an in-flight turn.  agent.switch_model()
                # mutates self.model / self.provider / self.base_url /
                # self.client in place; the worker thread running
                # agent.run_conversation is reading those on every
                # iteration.  A mid-turn swap can send an HTTP request
                # with the new base_url but old model (or vice versa),
                # producing 400/404s the user never asked for.  Parity
                # with the gateway's running-agent /model guard.
                if session.get("running"):
                    return _err(
                        rid, 4009,
                        "session busy — /interrupt the current turn before switching models",
                    )
                result = _apply_model_switch(params.get("session_id", ""), session, value)
            else:
                result = _apply_model_switch("", {"agent": None}, value)
            return _ok(rid, {"key": key, "value": result["value"], "warning": result["warning"]})
        except Exception as e:
            return _err(rid, 5001, str(e))

    if key == "verbose":
        cycle = ["off", "new", "all", "verbose"]
        cur = session.get("tool_progress_mode", _load_tool_progress_mode()) if session else _load_tool_progress_mode()
        if value and value != "cycle":
            nv = str(value).strip().lower()
            if nv not in cycle:
                return _err(rid, 4002, f"unknown verbose mode: {value}")
        else:
            try:
                idx = cycle.index(cur)
            except ValueError:
                idx = 2
            nv = cycle[(idx + 1) % len(cycle)]
        _write_config_key("display.tool_progress", nv)
        if session:
            session["tool_progress_mode"] = nv
            agent = session.get("agent")
            if agent is not None:
                agent.verbose_logging = nv == "verbose"
        return _ok(rid, {"key": key, "value": nv})

    if key == "yolo":
        try:
            if session:
                from tools.approval import (
                    disable_session_yolo,
                    enable_session_yolo,
                    is_session_yolo_enabled,
                )

                current = is_session_yolo_enabled(session["session_key"])
                if current:
                    disable_session_yolo(session["session_key"])
                    nv = "0"
                else:
                    enable_session_yolo(session["session_key"])
                    nv = "1"
            else:
                current = bool(os.environ.get("HERMES_YOLO_MODE"))
                if current:
                    os.environ.pop("HERMES_YOLO_MODE", None)
                    nv = "0"
                else:
                    os.environ["HERMES_YOLO_MODE"] = "1"
                    nv = "1"
            return _ok(rid, {"key": key, "value": nv})
        except Exception as e:
            return _err(rid, 5001, str(e))

    if key == "reasoning":
        try:
            from hermes_constants import parse_reasoning_effort

            arg = str(value or "").strip().lower()
            if arg in ("show", "on"):
                _write_config_key("display.show_reasoning", True)
                if session:
                    session["show_reasoning"] = True
                return _ok(rid, {"key": key, "value": "show"})
            if arg in ("hide", "off"):
                _write_config_key("display.show_reasoning", False)
                if session:
                    session["show_reasoning"] = False
                return _ok(rid, {"key": key, "value": "hide"})

            parsed = parse_reasoning_effort(arg)
            if parsed is None:
                return _err(rid, 4002, f"unknown reasoning value: {value}")
            _write_config_key("agent.reasoning_effort", arg)
            if session and session.get("agent") is not None:
                session["agent"].reasoning_config = parsed
            return _ok(rid, {"key": key, "value": arg})
        except Exception as e:
            return _err(rid, 5001, str(e))

    if key == "details_mode":
        nv = str(value or "").strip().lower()
        allowed_dm = frozenset({"hidden", "collapsed", "expanded"})
        if nv not in allowed_dm:
            return _err(rid, 4002, f"unknown details_mode: {value}")
        _write_config_key("display.details_mode", nv)
        return _ok(rid, {"key": key, "value": nv})

    if key == "thinking_mode":
        nv = str(value or "").strip().lower()
        allowed_tm = frozenset({"collapsed", "truncated", "full"})
        if nv not in allowed_tm:
            return _err(rid, 4002, f"unknown thinking_mode: {value}")
        _write_config_key("display.thinking_mode", nv)
        # Backward compatibility bridge: keep details_mode aligned.
        _write_config_key("display.details_mode", "expanded" if nv == "full" else "collapsed")
        return _ok(rid, {"key": key, "value": nv})

    if key in ("compact", "statusbar"):
        raw = str(value or "").strip().lower()
        cfg0 = _load_cfg()
        d0 = cfg0.get("display") if isinstance(cfg0.get("display"), dict) else {}
        def_key = "tui_compact" if key == "compact" else "tui_statusbar"
        cur_b = bool(d0.get(def_key, False if key == "compact" else True))
        if raw in ("", "toggle"):
            nv_b = not cur_b
        elif raw == "on":
            nv_b = True
        elif raw == "off":
            nv_b = False
        else:
            return _err(rid, 4002, f"unknown {key} value: {value}")
        _write_config_key(f"display.{def_key}", nv_b)
        out = "on" if nv_b else "off"
        return _ok(rid, {"key": key, "value": out})

    if key in ("prompt", "personality", "skin"):
        try:
            cfg = _load_cfg()
            if key == "prompt":
                if value == "clear":
                    cfg.pop("custom_prompt", None)
                    nv = ""
                else:
                    cfg["custom_prompt"] = value
                    nv = value
                _save_cfg(cfg)
            elif key == "personality":
                sid_key = params.get("session_id", "")
                pname, new_prompt = _validate_personality(str(value or ""), cfg)
                _write_config_key("display.personality", pname)
                _write_config_key("agent.system_prompt", new_prompt)
                nv = str(value or "default")
                history_reset, info = _apply_personality_to_session(sid_key, session, new_prompt)
            else:
                _write_config_key(f"display.{key}", value)
                nv = value
                if key == "skin":
                    _emit("skin.changed", "", resolve_skin())
            resp = {"key": key, "value": nv}
            if key == "personality":
                resp["history_reset"] = history_reset
                if info is not None:
                    resp["info"] = info
            return _ok(rid, resp)
        except Exception as e:
            return _err(rid, 5001, str(e))

    return _err(rid, 4002, f"unknown config key: {key}")


@method("config.get")
def _(rid, params: dict) -> dict:
    key = params.get("key", "")
    if key == "provider":
        try:
            from hermes_cli.models import list_available_providers, normalize_provider
            model = _resolve_model()
            parts = model.split("/", 1)
            return _ok(rid, {"model": model, "provider": normalize_provider(parts[0]) if len(parts) > 1 else "unknown",
                             "providers": list_available_providers()})
        except Exception as e:
            return _err(rid, 5013, str(e))
    if key == "profile":
        from hermes_constants import display_hermes_home
        return _ok(rid, {"home": str(_hermes_home), "display": display_hermes_home()})
    if key == "full":
        return _ok(rid, {"config": _load_cfg()})
    if key == "prompt":
        return _ok(rid, {"prompt": _load_cfg().get("custom_prompt", "")})
    if key == "skin":
        return _ok(rid, {"value": _load_cfg().get("display", {}).get("skin", "default")})
    if key == "personality":
        return _ok(rid, {"value": _load_cfg().get("display", {}).get("personality", "default")})
    if key == "reasoning":
        cfg = _load_cfg()
        effort = str(cfg.get("agent", {}).get("reasoning_effort", "medium") or "medium")
        display = "show" if bool(cfg.get("display", {}).get("show_reasoning", False)) else "hide"
        return _ok(rid, {"value": effort, "display": display})
    if key == "details_mode":
        allowed_dm = frozenset({"hidden", "collapsed", "expanded"})
        raw = str(_load_cfg().get("display", {}).get("details_mode", "collapsed") or "collapsed").strip().lower()
        nv = raw if raw in allowed_dm else "collapsed"
        return _ok(rid, {"value": nv})
    if key == "thinking_mode":
        allowed_tm = frozenset({"collapsed", "truncated", "full"})
        cfg = _load_cfg()
        raw = str(cfg.get("display", {}).get("thinking_mode", "") or "").strip().lower()
        if raw in allowed_tm:
            nv = raw
        else:
            dm = str(cfg.get("display", {}).get("details_mode", "collapsed") or "collapsed").strip().lower()
            nv = "full" if dm == "expanded" else "collapsed"
        return _ok(rid, {"value": nv})
    if key == "compact":
        on = bool(_load_cfg().get("display", {}).get("tui_compact", False))
        return _ok(rid, {"value": "on" if on else "off"})
    if key == "statusbar":
        on = bool(_load_cfg().get("display", {}).get("tui_statusbar", True))
        return _ok(rid, {"value": "on" if on else "off"})
    if key == "mtime":
        cfg_path = _hermes_home / "config.yaml"
        try:
            return _ok(rid, {"mtime": cfg_path.stat().st_mtime if cfg_path.exists() else 0})
        except Exception:
            return _ok(rid, {"mtime": 0})
    return _err(rid, 4002, f"unknown config key: {key}")


@method("setup.status")
def _(rid, params: dict) -> dict:
    try:
        from hermes_cli.main import _has_any_provider_configured
        return _ok(rid, {"provider_configured": bool(_has_any_provider_configured())})
    except Exception as e:
        return _err(rid, 5016, str(e))


# ── Methods: tools & system ──────────────────────────────────────────

@method("process.stop")
def _(rid, params: dict) -> dict:
    try:
        from tools.process_registry import process_registry
        return _ok(rid, {"killed": process_registry.kill_all()})
    except Exception as e:
        return _err(rid, 5010, str(e))


@method("reload.mcp")
def _(rid, params: dict) -> dict:
    session = _sessions.get(params.get("session_id", ""))
    try:
        from tools.mcp_tool import shutdown_mcp_servers, discover_mcp_tools
        shutdown_mcp_servers()
        discover_mcp_tools()
        if session:
            agent = session["agent"]
            if hasattr(agent, "refresh_tools"):
                agent.refresh_tools()
            _emit("session.info", params.get("session_id", ""), _session_info(agent))
        return _ok(rid, {"status": "reloaded"})
    except Exception as e:
        return _err(rid, 5015, str(e))


_TUI_HIDDEN: frozenset[str] = frozenset({
    "sethome", "set-home", "update", "commands", "status", "approve", "deny",
})

_TUI_EXTRA: list[tuple[str, str, str]] = [
    ("/compact", "Toggle compact display mode", "TUI"),
    ("/logs", "Show recent gateway log lines", "TUI"),
]

# Commands that queue messages onto _pending_input in the CLI.
# In the TUI the slash worker subprocess has no reader for that queue,
# so slash.exec rejects them → TUI falls through to command.dispatch.
_PENDING_INPUT_COMMANDS: frozenset[str] = frozenset({
    "retry", "queue", "q", "steer", "plan",
})


@method("commands.catalog")
def _(rid, params: dict) -> dict:
    """Registry-backed slash metadata for the TUI — categorized, no aliases."""
    try:
        from hermes_cli.commands import COMMAND_REGISTRY, SUBCOMMANDS, _build_description

        all_pairs: list[list[str]] = []
        canon: dict[str, str] = {}
        categories: list[dict] = []
        cat_map: dict[str, list[list[str]]] = {}
        cat_order: list[str] = []

        for cmd in COMMAND_REGISTRY:
            c = f"/{cmd.name}"
            canon[c.lower()] = c
            for a in cmd.aliases:
                canon[f"/{a}".lower()] = c

            if cmd.name in _TUI_HIDDEN:
                continue

            desc = _build_description(cmd)
            all_pairs.append([c, desc])

            cat = cmd.category
            if cat not in cat_map:
                cat_map[cat] = []
                cat_order.append(cat)
            cat_map[cat].append([c, desc])

        for name, desc, cat in _TUI_EXTRA:
            all_pairs.append([name, desc])
            if cat not in cat_map:
                cat_map[cat] = []
                cat_order.append(cat)
            cat_map[cat].append([name, desc])

        warning = ""
        try:
            qcmds = _load_cfg().get("quick_commands", {}) or {}
            if isinstance(qcmds, dict) and qcmds:
                bucket = "User commands"
                if bucket not in cat_map:
                    cat_map[bucket] = []
                    cat_order.append(bucket)
                for qname, qc in sorted(qcmds.items()):
                    if not isinstance(qc, dict):
                        continue
                    key = f"/{qname}"
                    canon[key.lower()] = key
                    qtype = qc.get("type", "")
                    if qtype == "exec":
                        default_desc = f"exec: {qc.get('command', '')}"
                    elif qtype == "alias":
                        default_desc = f"alias → {qc.get('target', '')}"
                    else:
                        default_desc = qtype or "quick command"
                    qdesc = str(qc.get("description") or default_desc)
                    qdesc = qdesc[:120] + ("…" if len(qdesc) > 120 else "")
                    all_pairs.append([key, qdesc])
                    cat_map[bucket].append([key, qdesc])
        except Exception as e:
            if not warning:
                warning = f"quick_commands discovery unavailable: {e}"

        skill_count = 0
        try:
            from agent.skill_commands import scan_skill_commands
            for k, info in sorted(scan_skill_commands().items()):
                d = str(info.get("description", "Skill"))
                all_pairs.append([k, d[:120] + ("…" if len(d) > 120 else "")])
                skill_count += 1
        except Exception as e:
            warning = f"skill discovery unavailable: {e}"

        for cat in cat_order:
            categories.append({"name": cat, "pairs": cat_map[cat]})

        sub = {k: v[:] for k, v in SUBCOMMANDS.items()}
        return _ok(rid, {
            "pairs": all_pairs,
            "sub": sub,
            "canon": canon,
            "categories": categories,
            "skill_count": skill_count,
            "warning": warning,
        })
    except Exception as e:
        return _err(rid, 5020, str(e))


def _cli_exec_blocked(argv: list[str]) -> str | None:
    """Return user hint if this argv must not run headless in the gateway process."""
    if not argv:
        return "bare `hermes` is interactive — use `/hermes chat -q …` or run `hermes` in another terminal"
    a0 = argv[0].lower()
    if a0 == "setup":
        return "`hermes setup` needs a full terminal — run it outside the TUI"
    if a0 == "gateway":
        return "`hermes gateway` is long-running — run it in another terminal"
    if a0 == "sessions" and len(argv) > 1 and argv[1].lower() == "browse":
        return "`hermes sessions browse` is interactive — use /resume here, or run browse in another terminal"
    if a0 == "config" and len(argv) > 1 and argv[1].lower() == "edit":
        return "`hermes config edit` needs $EDITOR in a real terminal"
    return None


@method("cli.exec")
def _(rid, params: dict) -> dict:
    """Run `python -m hermes_cli.main` with argv; capture stdout/stderr (non-interactive only)."""
    argv = params.get("argv", [])
    if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
        return _err(rid, 4003, "argv must be list[str]")
    hint = _cli_exec_blocked(argv)
    if hint:
        return _ok(rid, {"blocked": True, "hint": hint, "code": -1, "output": ""})
    try:
        r = subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", *argv],
            capture_output=True,
            text=True,
            timeout=min(int(params.get("timeout", 240)), 600),
            cwd=os.getcwd(),
            env=os.environ.copy(),
        )
        parts = [r.stdout or "", r.stderr or ""]
        out = "\n".join(p for p in parts if p).strip() or "(no output)"
        return _ok(rid, {"blocked": False, "code": r.returncode, "output": out[:48_000]})
    except subprocess.TimeoutExpired:
        return _err(rid, 5016, "cli.exec: timeout")
    except Exception as e:
        return _err(rid, 5017, str(e))


@method("command.resolve")
def _(rid, params: dict) -> dict:
    try:
        from hermes_cli.commands import resolve_command
        r = resolve_command(params.get("name", ""))
        if r:
            return _ok(rid, {"canonical": r.name, "description": r.description, "category": r.category})
        return _err(rid, 4011, f"unknown command: {params.get('name')}")
    except Exception as e:
        return _err(rid, 5012, str(e))


def _resolve_name(name: str) -> str:
    try:
        from hermes_cli.commands import resolve_command
        r = resolve_command(name)
        return r.name if r else name
    except Exception:
        return name


@method("command.dispatch")
def _(rid, params: dict) -> dict:
    name, arg = params.get("name", "").lstrip("/"), params.get("arg", "")
    resolved = _resolve_name(name)
    if resolved != name:
        name = resolved
    session = _sessions.get(params.get("session_id", ""))

    qcmds = _load_cfg().get("quick_commands", {})
    if name in qcmds:
        qc = qcmds[name]
        if qc.get("type") == "exec":
            r = subprocess.run(qc.get("command", ""), shell=True, capture_output=True, text=True, timeout=30)
            output = ((r.stdout or "") + ("\n" if r.stdout and r.stderr else "") + (r.stderr or "")).strip()[:4000]
            if r.returncode != 0:
                return _err(rid, 4018, output or f"quick command failed with exit code {r.returncode}")
            return _ok(rid, {"type": "exec", "output": output})
        if qc.get("type") == "alias":
            return _ok(rid, {"type": "alias", "target": qc.get("target", "")})

    try:
        from hermes_cli.plugins import get_plugin_command_handler
        handler = get_plugin_command_handler(name)
        if handler:
            return _ok(rid, {"type": "plugin", "output": str(handler(arg) or "")})
    except Exception:
        pass

    try:
        from agent.skill_commands import scan_skill_commands, build_skill_invocation_message
        cmds = scan_skill_commands()
        key = f"/{name}"
        if key in cmds:
            msg = build_skill_invocation_message(key, arg, task_id=session.get("session_key", "") if session else "")
            if msg:
                return _ok(rid, {"type": "skill", "message": msg, "name": cmds[key].get("name", name)})
    except Exception:
        pass

    # ── Commands that queue messages onto _pending_input in the CLI ───
    # In the TUI the slash worker subprocess has no reader for that queue,
    # so we handle them here and return a structured payload.

    if name in ("queue", "q"):
        if not arg:
            return _err(rid, 4004, "usage: /queue <prompt>")
        return _ok(rid, {"type": "send", "message": arg})

    if name == "retry":
        if not session:
            return _err(rid, 4001, "no active session to retry")
        if session.get("running"):
            return _err(rid, 4009, "session busy — /interrupt the current turn before /retry")
        history = session.get("history", [])
        if not history:
            return _err(rid, 4018, "no previous user message to retry")
        # Walk backwards to find the last user message
        last_user_idx = None
        for i in range(len(history) - 1, -1, -1):
            if history[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is None:
            return _err(rid, 4018, "no previous user message to retry")
        content = history[last_user_idx].get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
            )
        if not content:
            return _err(rid, 4018, "last user message is empty")
        # Truncate history: remove everything from the last user message onward
        # (mirrors CLI retry_last() which strips the failed exchange)
        with session["history_lock"]:
            session["history"] = history[:last_user_idx]
            session["history_version"] = int(session.get("history_version", 0)) + 1
        return _ok(rid, {"type": "send", "message": content})

    if name == "steer":
        if not arg:
            return _err(rid, 4004, "usage: /steer <prompt>")
        agent = session.get("agent") if session else None
        if agent and hasattr(agent, "steer"):
            try:
                accepted = agent.steer(arg)
                if accepted:
                    return _ok(rid, {"type": "exec", "output": f"⏩ Steer queued — arrives after the next tool call: {arg[:80]}{'...' if len(arg) > 80 else ''}"})
            except Exception:
                pass
        # Fallback: no active run, treat as next-turn message
        return _ok(rid, {"type": "send", "message": arg})

    if name == "plan":
        try:
            from agent.skill_commands import build_skill_invocation_message as _bsim, build_plan_path
            user_instruction = arg or ""
            plan_path = build_plan_path(user_instruction)
            msg = _bsim(
                "/plan", user_instruction,
                task_id=session.get("session_key", "") if session else "",
                runtime_note=(
                    "Save the markdown plan with write_file to this exact relative path "
                    f"inside the active workspace/backend cwd: {plan_path}"
                ),
            )
            if msg:
                return _ok(rid, {"type": "send", "message": msg})
        except Exception as e:
            return _err(rid, 5030, f"plan skill failed: {e}")

    return _err(rid, 4018, f"not a quick/plugin/skill command: {name}")


# ── Methods: paste ────────────────────────────────────────────────────

_paste_counter = 0

@method("paste.collapse")
def _(rid, params: dict) -> dict:
    global _paste_counter
    text = params.get("text", "")
    if not text:
        return _err(rid, 4004, "empty paste")

    _paste_counter += 1
    line_count = text.count('\n') + 1
    paste_dir = _hermes_home / "pastes"
    paste_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    paste_file = paste_dir / f"paste_{_paste_counter}_{datetime.now().strftime('%H%M%S')}.txt"
    paste_file.write_text(text, encoding="utf-8")

    placeholder = f"[Pasted text #{_paste_counter}: {line_count} lines \u2192 {paste_file}]"
    return _ok(rid, {"placeholder": placeholder, "path": str(paste_file), "lines": line_count})


# ── Methods: complete ─────────────────────────────────────────────────

@method("complete.path")
def _(rid, params: dict) -> dict:
    word = params.get("word", "")
    if not word:
        return _ok(rid, {"items": []})

    items: list[dict] = []
    try:
        is_context = word.startswith("@")
        query = word[1:] if is_context else word

        if is_context and not query:
            items = [
                {"text": "@diff", "display": "@diff", "meta": "git diff"},
                {"text": "@staged", "display": "@staged", "meta": "staged diff"},
                {"text": "@file:", "display": "@file:", "meta": "attach file"},
                {"text": "@folder:", "display": "@folder:", "meta": "attach folder"},
                {"text": "@url:", "display": "@url:", "meta": "fetch url"},
                {"text": "@git:", "display": "@git:", "meta": "git log"},
            ]
            return _ok(rid, {"items": items})

        # Accept both `@folder:path` and the bare `@folder` form so the user
        # sees directory listings as soon as they finish typing the keyword,
        # without first accepting the static `@folder:` hint.
        if is_context and query in ("file", "folder"):
            prefix_tag, path_part = query, ""
        elif is_context and query.startswith(("file:", "folder:")):
            prefix_tag, _, tail = query.partition(":")
            path_part = tail
        else:
            prefix_tag = ""
            path_part = query if is_context else query

        expanded = _normalize_completion_path(path_part) if path_part else "."
        if expanded == "." or not expanded:
            search_dir, match = ".", ""
        elif expanded.endswith("/"):
            search_dir, match = expanded, ""
        else:
            search_dir = os.path.dirname(expanded) or "."
            match = os.path.basename(expanded)

        if not os.path.isdir(search_dir):
            return _ok(rid, {"items": []})

        want_dir = prefix_tag == "folder"
        match_lower = match.lower()
        for entry in sorted(os.listdir(search_dir)):
            if match and not entry.lower().startswith(match_lower):
                continue
            if is_context and not prefix_tag and entry.startswith("."):
                continue
            full = os.path.join(search_dir, entry)
            is_dir = os.path.isdir(full)
            # Explicit `@folder:` / `@file:` — honour the user's filter.  Skip
            # the opposite kind instead of auto-rewriting the completion tag,
            # which used to defeat the prefix and let `@folder:` list files.
            if prefix_tag and want_dir != is_dir:
                continue
            rel = os.path.relpath(full)
            suffix = "/" if is_dir else ""

            if is_context and prefix_tag:
                text = f"@{prefix_tag}:{rel}{suffix}"
            elif is_context:
                kind = "folder" if is_dir else "file"
                text = f"@{kind}:{rel}{suffix}"
            elif word.startswith("~"):
                text = "~/" + os.path.relpath(full, os.path.expanduser("~")) + suffix
            elif word.startswith("./"):
                text = "./" + rel + suffix
            else:
                text = rel + suffix

            items.append({"text": text, "display": entry + suffix, "meta": "dir" if is_dir else ""})
            if len(items) >= 30:
                break
    except Exception as e:
        return _err(rid, 5021, str(e))

    return _ok(rid, {"items": items})


@method("complete.slash")
def _(rid, params: dict) -> dict:
    text = params.get("text", "")
    if not text.startswith("/"):
        return _ok(rid, {"items": []})

    try:
        from hermes_cli.commands import SlashCommandCompleter
        from prompt_toolkit.document import Document
        from prompt_toolkit.formatted_text import to_plain_text

        from agent.skill_commands import get_skill_commands

        completer = SlashCommandCompleter(skill_commands_provider=lambda: get_skill_commands())
        doc = Document(text, len(text))
        items = [
            {"text": c.text, "display": c.display or c.text,
             "meta": to_plain_text(c.display_meta) if c.display_meta else ""}
            for c in completer.get_completions(doc, None)
        ][:30]
        text_lower = text.lower()
        extras = [
            {"text": "/compact", "display": "/compact", "meta": "Toggle compact display mode"},
            {"text": "/logs", "display": "/logs", "meta": "Show recent gateway log lines"},
        ]
        for extra in extras:
            if extra["text"].startswith(text_lower) and not any(item["text"] == extra["text"] for item in items):
                items.append(extra)
        return _ok(rid, {"items": items, "replace_from": text.rfind(" ") + 1 if " " in text else 1})
    except Exception as e:
        return _err(rid, 5020, str(e))


@method("model.options")
def _(rid, params: dict) -> dict:
    try:
        from hermes_cli.model_switch import list_authenticated_providers

        session = _sessions.get(params.get("session_id", ""))
        agent = session.get("agent") if session else None
        cfg = _load_cfg()
        current_provider = getattr(agent, "provider", "") or ""
        current_model = getattr(agent, "model", "") or _resolve_model()
        # list_authenticated_providers already populates each provider's
        # "models" with the curated list (same source as `hermes model` and
        # classic CLI's /model picker). Do NOT overwrite with live
        # provider_model_ids() — that bypasses curation and pulls in
        # non-agentic models (e.g. Nous /models returns ~400 IDs including
        # TTS, embeddings, rerankers, image/video generators).
        providers = list_authenticated_providers(
            current_provider=current_provider,
            user_providers=cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {},
            custom_providers=cfg.get("custom_providers") if isinstance(cfg.get("custom_providers"), list) else [],
            max_models=50,
        )
        return _ok(rid, {"providers": providers, "model": current_model, "provider": current_provider})
    except Exception as e:
        return _err(rid, 5033, str(e))


# ── Methods: slash.exec ──────────────────────────────────────────────


def _mirror_slash_side_effects(sid: str, session: dict, command: str) -> str:
    """Apply side effects that must also hit the gateway's live agent."""
    parts = command.lstrip("/").split(None, 1)
    if not parts:
        return ""
    name, arg, agent = parts[0], (parts[1].strip() if len(parts) > 1 else ""), session.get("agent")

    # Reject agent-mutating commands during an in-flight turn.  These
    # all do read-then-mutate on live agent/session state that the
    # worker thread running agent.run_conversation is using.  Parity
    # with the session.compress / session.undo guards and the gateway
    # runner's running-agent /model guard.
    _MUTATES_WHILE_RUNNING = {"model", "personality", "prompt", "compress"}
    if name in _MUTATES_WHILE_RUNNING and session.get("running"):
        return (
            f"session busy — /interrupt the current turn before running /{name}"
        )

    try:
        if name == "model" and arg and agent:
            result = _apply_model_switch(sid, session, arg)
            return result.get("warning", "")
        elif name == "personality" and arg and agent:
            _, new_prompt = _validate_personality(arg, _load_cfg())
            _apply_personality_to_session(sid, session, new_prompt)
        elif name == "prompt" and agent:
            cfg = _load_cfg()
            new_prompt = cfg.get("agent", {}).get("system_prompt", "") or ""
            agent.ephemeral_system_prompt = new_prompt or None
            agent._cached_system_prompt = None
        elif name == "compress" and agent:
            with session["history_lock"]:
                _compress_session_history(session, arg)
            _emit("session.info", sid, _session_info(agent))
        elif name == "fast" and agent:
            mode = arg.lower()
            if mode in {"fast", "on"}:
                agent.service_tier = "priority"
            elif mode in {"normal", "off"}:
                agent.service_tier = None
            _emit("session.info", sid, _session_info(agent))
        elif name == "reload-mcp" and agent and hasattr(agent, "reload_mcp_tools"):
            agent.reload_mcp_tools()
        elif name == "stop":
            from tools.process_registry import process_registry
            process_registry.kill_all()
    except Exception as e:
        return f"live session sync failed: {e}"
    return ""


@method("slash.exec")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err

    cmd = params.get("command", "").strip()
    if not cmd:
        return _err(rid, 4004, "empty command")

    # Skill slash commands and _pending_input commands must NOT go through the
    # slash worker — see _PENDING_INPUT_COMMANDS definition above.
    # (/browser connect/disconnect also uses _pending_input for context
    # notes, but the actual browser operations need the slash worker's
    # env-var side effects, so they stay in slash.exec — only the context
    # note to the model is lost, which is low-severity.)
    _cmd_parts = cmd.split() if not cmd.startswith("/") else cmd.lstrip("/").split()
    _cmd_base = _cmd_parts[0] if _cmd_parts else ""

    if _cmd_base in _PENDING_INPUT_COMMANDS:
        return _err(rid, 4018, f"pending-input command: use command.dispatch for /{_cmd_base}")

    try:
        from agent.skill_commands import get_skill_commands
        _cmd_key = f"/{_cmd_base}"
        if _cmd_key in get_skill_commands():
            return _err(rid, 4018, f"skill command: use command.dispatch for {_cmd_key}")
    except Exception:
        pass

    worker = session.get("slash_worker")
    if not worker:
        try:
            worker = _SlashWorker(session["session_key"], getattr(session.get("agent"), "model", _resolve_model()))
            session["slash_worker"] = worker
        except Exception as e:
            return _err(rid, 5030, f"slash worker start failed: {e}")

    try:
        output = worker.run(cmd)
        warning = _mirror_slash_side_effects(params.get("session_id", ""), session, cmd)
        payload = {"output": output or "(no output)"}
        if warning:
            payload["warning"] = warning
        return _ok(rid, payload)
    except Exception as e:
        try:
            worker.close()
        except Exception:
            pass
        session["slash_worker"] = None
        return _err(rid, 5030, str(e))


# ── Methods: voice ───────────────────────────────────────────────────

@method("voice.toggle")
def _(rid, params: dict) -> dict:
    action = params.get("action", "status")
    if action == "status":
        env = os.environ.get("HERMES_VOICE", "").strip()
        if env in {"0", "1"}:
            return _ok(rid, {"enabled": env == "1"})
        return _ok(rid, {"enabled": bool(_load_cfg().get("display", {}).get("voice_enabled", False))})
    if action in ("on", "off"):
        enabled = action == "on"
        os.environ["HERMES_VOICE"] = "1" if enabled else "0"
        _write_config_key("display.voice_enabled", enabled)
        return _ok(rid, {"enabled": action == "on"})
    return _err(rid, 4013, f"unknown voice action: {action}")


@method("voice.record")
def _(rid, params: dict) -> dict:
    action = params.get("action", "start")
    try:
        if action == "start":
            from hermes_cli.voice import start_recording
            start_recording()
            return _ok(rid, {"status": "recording"})
        if action == "stop":
            from hermes_cli.voice import stop_and_transcribe
            return _ok(rid, {"text": stop_and_transcribe() or ""})
        return _err(rid, 4019, f"unknown voice action: {action}")
    except ImportError:
        return _err(rid, 5025, "voice module not available — install audio dependencies")
    except Exception as e:
        return _err(rid, 5025, str(e))


@method("voice.tts")
def _(rid, params: dict) -> dict:
    text = params.get("text", "")
    if not text:
        return _err(rid, 4020, "text required")
    try:
        from hermes_cli.voice import speak_text
        threading.Thread(target=speak_text, args=(text,), daemon=True).start()
        return _ok(rid, {"status": "speaking"})
    except ImportError:
        return _err(rid, 5026, "voice module not available")
    except Exception as e:
        return _err(rid, 5026, str(e))


# ── Methods: insights ────────────────────────────────────────────────

@method("insights.get")
def _(rid, params: dict) -> dict:
    days = params.get("days", 30)
    try:
        cutoff = time.time() - days * 86400
        rows = [s for s in _get_db().list_sessions_rich(limit=500) if (s.get("started_at") or 0) >= cutoff]
        return _ok(rid, {"days": days, "sessions": len(rows), "messages": sum(s.get("message_count", 0) for s in rows)})
    except Exception as e:
        return _err(rid, 5017, str(e))


# ── Methods: rollback ────────────────────────────────────────────────

@method("rollback.list")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    try:
        def go(mgr, cwd):
            if not mgr.enabled:
                return _ok(rid, {"enabled": False, "checkpoints": []})
            return _ok(rid, {"enabled": True, "checkpoints": [
                {"hash": c.get("hash", ""), "timestamp": c.get("timestamp", ""), "message": c.get("message", "")}
                for c in mgr.list_checkpoints(cwd)]})
        return _with_checkpoints(session, go)
    except Exception as e:
        return _err(rid, 5020, str(e))


@method("rollback.restore")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    target = params.get("hash", "")
    file_path = params.get("file_path", "")
    if not target:
        return _err(rid, 4014, "hash required")
    # Full-history rollback mutates session history.  Rejecting during
    # an in-flight turn prevents prompt.submit from silently dropping
    # the agent's output (version mismatch path) or clobbering the
    # rollback (version-matches path).  A file-scoped rollback only
    # touches disk, so we allow it.
    if not file_path and session.get("running"):
        return _err(rid, 4009, "session busy — /interrupt the current turn before full rollback.restore")
    try:
        def go(mgr, cwd):
            resolved = _resolve_checkpoint_hash(mgr, cwd, target)
            result = mgr.restore(cwd, resolved, file_path=file_path or None)
            if result.get("success") and not file_path:
                removed = 0
                with session["history_lock"]:
                    history = session.get("history", [])
                    while history and history[-1].get("role") in ("assistant", "tool"):
                        history.pop()
                        removed += 1
                    if history and history[-1].get("role") == "user":
                        history.pop()
                        removed += 1
                    if removed:
                        session["history_version"] = int(session.get("history_version", 0)) + 1
                result["history_removed"] = removed
            return result

        return _ok(rid, _with_checkpoints(session, go))
    except Exception as e:
        return _err(rid, 5021, str(e))


@method("rollback.diff")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    target = params.get("hash", "")
    if not target:
        return _err(rid, 4014, "hash required")
    try:
        r = _with_checkpoints(session, lambda mgr, cwd: mgr.diff(cwd, _resolve_checkpoint_hash(mgr, cwd, target)))
        raw = r.get("diff", "")[:4000]
        payload = {"stat": r.get("stat", ""), "diff": raw}
        rendered = render_diff(raw, session.get("cols", 80))
        if rendered:
            payload["rendered"] = rendered
        return _ok(rid, payload)
    except Exception as e:
        return _err(rid, 5022, str(e))


# ── Methods: browser / plugins / cron / skills ───────────────────────

@method("browser.manage")
def _(rid, params: dict) -> dict:
    action = params.get("action", "status")
    if action == "status":
        url = os.environ.get("BROWSER_CDP_URL", "")
        return _ok(rid, {"connected": bool(url), "url": url})
    if action == "connect":
        url = params.get("url", "http://localhost:9222")
        try:
            import urllib.request
            from urllib.parse import urlparse
            from tools.browser_tool import cleanup_all_browsers

            parsed = urlparse(url if "://" in url else f"http://{url}")
            if parsed.scheme not in {"http", "https", "ws", "wss"}:
                return _err(rid, 4015, f"unsupported browser url: {url}")
            probe_root = (
                f"{'https' if parsed.scheme == 'wss' else 'http' if parsed.scheme == 'ws' else parsed.scheme}://{parsed.netloc}"
            )
            probe_urls = [f"{probe_root.rstrip('/')}/json/version", f"{probe_root.rstrip('/')}/json"]
            ok = False
            for probe in probe_urls:
                try:
                    with urllib.request.urlopen(probe, timeout=2.0) as resp:
                        if 200 <= getattr(resp, "status", 200) < 300:
                            ok = True
                            break
                except Exception:
                    continue
            if not ok:
                return _err(rid, 5031, f"could not reach browser CDP at {url}")

            os.environ["BROWSER_CDP_URL"] = url
            cleanup_all_browsers()
        except Exception as e:
            return _err(rid, 5031, str(e))
        return _ok(rid, {"connected": True, "url": url})
    if action == "disconnect":
        os.environ.pop("BROWSER_CDP_URL", None)
        try:
            from tools.browser_tool import cleanup_all_browsers
            cleanup_all_browsers()
        except Exception:
            pass
        return _ok(rid, {"connected": False})
    return _err(rid, 4015, f"unknown action: {action}")


@method("plugins.list")
def _(rid, params: dict) -> dict:
    try:
        from hermes_cli.plugins import get_plugin_manager
        return _ok(rid, {"plugins": [
            {"name": n, "version": getattr(i, "version", "?"), "enabled": getattr(i, "enabled", True)}
            for n, i in get_plugin_manager()._plugins.items()]})
    except Exception as e:
        return _err(rid, 5032, str(e))


@method("config.show")
def _(rid, params: dict) -> dict:
    try:
        cfg = _load_cfg()
        model = _resolve_model()
        api_key = os.environ.get("HERMES_API_KEY", "") or cfg.get("api_key", "")
        masked = f"****{api_key[-4:]}" if len(api_key) > 4 else "(not set)"
        base_url = os.environ.get("HERMES_BASE_URL", "") or cfg.get("base_url", "")

        sections = [{
            "title": "Model",
            "rows": [
                ["Model", model],
                ["Base URL", base_url or "(default)"],
                ["API Key", masked],
            ]
        }, {
            "title": "Agent",
            "rows": [
                ["Max Turns", str(cfg.get("max_turns", 25))],
                ["Toolsets", ", ".join(cfg.get("enabled_toolsets", [])) or "all"],
                ["Verbose", str(cfg.get("verbose", False))],
            ]
        }, {
            "title": "Environment",
            "rows": [
                ["Working Dir", os.getcwd()],
                ["Config File", str(_hermes_home / "config.yaml")],
            ]
        }]
        return _ok(rid, {"sections": sections})
    except Exception as e:
        return _err(rid, 5030, str(e))


@method("tools.list")
def _(rid, params: dict) -> dict:
    try:
        from toolsets import get_all_toolsets, get_toolset_info
        session = _sessions.get(params.get("session_id", ""))
        enabled = set(getattr(session["agent"], "enabled_toolsets", []) or []) if session else set(_load_enabled_toolsets() or [])

        items = []
        for name in sorted(get_all_toolsets().keys()):
            info = get_toolset_info(name)
            if not info:
                continue
            items.append({
                "name": name,
                "description": info["description"],
                "tool_count": info["tool_count"],
                "enabled": name in enabled if enabled else True,
                "tools": info["resolved_tools"],
            })
        return _ok(rid, {"toolsets": items})
    except Exception as e:
        return _err(rid, 5031, str(e))


@method("tools.show")
def _(rid, params: dict) -> dict:
    try:
        from model_tools import get_toolset_for_tool, get_tool_definitions

        session = _sessions.get(params.get("session_id", ""))
        enabled = getattr(session["agent"], "enabled_toolsets", None) if session else _load_enabled_toolsets()
        tools = get_tool_definitions(enabled_toolsets=enabled, quiet_mode=True)
        sections = {}

        for tool in sorted(tools, key=lambda t: t["function"]["name"]):
            name = tool["function"]["name"]
            desc = str(tool["function"].get("description", "") or "").split("\n")[0]
            if ". " in desc:
                desc = desc[:desc.index(". ") + 1]
            sections.setdefault(get_toolset_for_tool(name) or "unknown", []).append({
                "name": name,
                "description": desc,
            })

        return _ok(rid, {
            "sections": [{"name": name, "tools": rows} for name, rows in sorted(sections.items())],
            "total": len(tools),
        })
    except Exception as e:
        return _err(rid, 5034, str(e))


@method("tools.configure")
def _(rid, params: dict) -> dict:
    action = str(params.get("action", "") or "").strip().lower()
    targets = [str(name).strip() for name in params.get("names", []) or [] if str(name).strip()]
    if action not in {"disable", "enable"}:
        return _err(rid, 4017, f"unknown tools action: {action}")
    if not targets:
        return _err(rid, 4018, "names required")

    try:
        from hermes_cli.config import load_config, save_config
        from hermes_cli.tools_config import (
            CONFIGURABLE_TOOLSETS,
            _apply_mcp_change,
            _apply_toolset_change,
            _get_platform_tools,
            _get_plugin_toolset_keys,
        )

        cfg = load_config()
        valid_toolsets = {ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS} | _get_plugin_toolset_keys()
        toolset_targets = [name for name in targets if ":" not in name]
        mcp_targets = [name for name in targets if ":" in name]
        unknown = [name for name in toolset_targets if name not in valid_toolsets]
        toolset_targets = [name for name in toolset_targets if name in valid_toolsets]

        if toolset_targets:
            _apply_toolset_change(cfg, "cli", toolset_targets, action)

        missing_servers = _apply_mcp_change(cfg, mcp_targets, action) if mcp_targets else set()
        save_config(cfg)

        session = _sessions.get(params.get("session_id", ""))
        info = _reset_session_agent(params.get("session_id", ""), session) if session else None
        enabled = sorted(_get_platform_tools(load_config(), "cli", include_default_mcp_servers=False))
        changed = [
            name for name in targets
            if name not in unknown and (":" not in name or name.split(":", 1)[0] not in missing_servers)
        ]

        return _ok(rid, {
            "changed": changed,
            "enabled_toolsets": enabled,
            "info": info,
            "missing_servers": sorted(missing_servers),
            "reset": bool(session),
            "unknown": unknown,
        })
    except Exception as e:
        return _err(rid, 5035, str(e))


@method("toolsets.list")
def _(rid, params: dict) -> dict:
    try:
        from toolsets import get_all_toolsets, get_toolset_info
        session = _sessions.get(params.get("session_id", ""))
        enabled = set(getattr(session["agent"], "enabled_toolsets", []) or []) if session else set(_load_enabled_toolsets() or [])

        items = []
        for name in sorted(get_all_toolsets().keys()):
            info = get_toolset_info(name)
            if not info:
                continue
            items.append({
                "name": name,
                "description": info["description"],
                "tool_count": info["tool_count"],
                "enabled": name in enabled if enabled else True,
            })
        return _ok(rid, {"toolsets": items})
    except Exception as e:
        return _err(rid, 5032, str(e))


@method("agents.list")
def _(rid, params: dict) -> dict:
    try:
        from tools.process_registry import process_registry
        procs = process_registry.list_sessions()
        return _ok(rid, {
            "processes": [{
                "session_id": p["session_id"],
                "command": p["command"][:80],
                "status": p["status"],
                "uptime": p["uptime_seconds"],
            } for p in procs]
        })
    except Exception as e:
        return _err(rid, 5033, str(e))


@method("cron.manage")
def _(rid, params: dict) -> dict:
    action, jid = params.get("action", "list"), params.get("name", "")
    try:
        from tools.cronjob_tools import cronjob
        if action == "list":
            return _ok(rid, json.loads(cronjob(action="list")))
        if action == "add":
            return _ok(rid, json.loads(cronjob(action="create", name=jid,
                                               schedule=params.get("schedule", ""), prompt=params.get("prompt", ""))))
        if action in ("remove", "pause", "resume"):
            return _ok(rid, json.loads(cronjob(action=action, job_id=jid)))
        return _err(rid, 4016, f"unknown cron action: {action}")
    except Exception as e:
        return _err(rid, 5023, str(e))


@method("skills.manage")
def _(rid, params: dict) -> dict:
    action, query = params.get("action", "list"), params.get("query", "")
    try:
        if action == "list":
            from hermes_cli.banner import get_available_skills
            return _ok(rid, {"skills": get_available_skills()})
        if action == "search":
            from hermes_cli.skills_hub import unified_search, GitHubAuth, create_source_router
            raw = unified_search(query, create_source_router(GitHubAuth()), source_filter="all", limit=20) or []
            return _ok(rid, {"results": [{"name": r.name, "description": r.description} for r in raw]})
        if action == "install":
            from hermes_cli.skills_hub import do_install
            class _Q:
                def print(self, *a, **k): pass
            do_install(query, skip_confirm=True, console=_Q())
            return _ok(rid, {"installed": True, "name": query})
        if action == "browse":
            from hermes_cli.skills_hub import browse_skills
            pg = int(params.get("page", 0) or 0) or (int(query) if query.isdigit() else 1)
            return _ok(rid, browse_skills(page=pg, page_size=int(params.get("page_size", 20))))
        if action == "inspect":
            from hermes_cli.skills_hub import inspect_skill
            return _ok(rid, {"info": inspect_skill(query) or {}})
        return _err(rid, 4017, f"unknown skills action: {action}")
    except Exception as e:
        return _err(rid, 5024, str(e))


# ── Methods: shell ───────────────────────────────────────────────────

@method("shell.exec")
def _(rid, params: dict) -> dict:
    cmd = params.get("command", "")
    if not cmd:
        return _err(rid, 4004, "empty command")
    try:
        from tools.approval import detect_dangerous_command
        is_dangerous, _, desc = detect_dangerous_command(cmd)
        if is_dangerous:
            return _err(rid, 4005, f"blocked: {desc}. Use the agent for dangerous commands.")
    except ImportError:
        pass
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=os.getcwd())
        return _ok(rid, {"stdout": r.stdout[-4000:], "stderr": r.stderr[-2000:], "code": r.returncode})
    except subprocess.TimeoutExpired:
        return _err(rid, 5002, "command timed out (30s)")
    except Exception as e:
        return _err(rid, 5003, str(e))
