import atexit
import json
import os
import subprocess
import sys
import threading
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
_pending: dict[str, threading.Event] = {}
_answers: dict[str, str] = {}
_db = None
_stdout_lock = threading.Lock()

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

        argv = [sys.executable, "-m", "tui_gateway.slash_worker", "--session-key", session_key]
        if model:
            argv += ["--model", model]

        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, cwd=os.getcwd(), env=os.environ.copy(),
        )
        threading.Thread(target=self._drain_stderr, daemon=True).start()

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

            for line in self.proc.stdout:
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
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


def _sess(params, rid):
    s = _sessions.get(params.get("session_id", ""))
    return (s, None) if s else (None, _err(rid, 4001, "session not found"))


# ── Config I/O ────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    try:
        import yaml
        p = _hermes_home / "config.yaml"
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


def _save_cfg(cfg: dict):
    import yaml
    with open(_hermes_home / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)


# ── Blocking prompt factory ──────────────────────────────────────────

def _block(event: str, sid: str, payload: dict, timeout: int = 300) -> str:
    rid = uuid.uuid4().hex[:8]
    ev = threading.Event()
    _pending[rid] = ev
    payload["request_id"] = rid
    _emit(event, sid, payload)
    ev.wait(timeout=timeout)
    _pending.pop(rid, None)
    return _answers.pop(rid, "")


def _clear_pending():
    for rid, ev in list(_pending.items()):
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


def _agent_cbs(sid: str) -> dict:
    return dict(
        tool_start_callback=lambda tc_id, name, args: _emit("tool.start", sid, {"tool_id": tc_id, "name": name, "context": _tool_ctx(name, args)}),
        tool_complete_callback=lambda tc_id, name, args, result: _emit("tool.complete", sid, {"tool_id": tc_id, "name": name}),
        tool_progress_callback=lambda name, preview, args: _emit("tool.progress", sid, {"name": name, "preview": preview}),
        tool_gen_callback=lambda name: _emit("tool.generating", sid, {"name": name}),
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


def _make_agent(sid: str, key: str, session_id: str | None = None):
    from run_agent import AIAgent
    cfg = _load_cfg()
    system_prompt = cfg.get("agent", {}).get("system_prompt", "") or ""
    return AIAgent(
        model=_resolve_model(), quiet_mode=True, platform="tui",
        session_id=session_id or key, session_db=_get_db(),
        ephemeral_system_prompt=system_prompt or None,
        **_agent_cbs(sid),
    )


def _init_session(sid: str, key: str, agent, history: list, cols: int = 80):
    _sessions[sid] = {
        "agent": agent,
        "session_key": key,
        "history": history,
        "attached_images": [],
        "image_counter": 0,
        "cols": cols,
        "slash_worker": None,
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


# ── Methods: session ─────────────────────────────────────────────────

@method("session.create")
def _(rid, params: dict) -> dict:
    sid = uuid.uuid4().hex[:8]
    key = _new_session_key()
    os.environ["HERMES_SESSION_KEY"] = key
    os.environ["HERMES_INTERACTIVE"] = "1"
    try:
        agent = _make_agent(sid, key)
        _get_db().create_session(key, source="tui", model=_resolve_model())
        _init_session(sid, key, agent, [], cols=int(params.get("cols", 80)))
    except Exception as e:
        return _err(rid, 5000, f"agent init failed: {e}")
    return _ok(rid, {"session_id": sid, "info": _session_info(agent)})


@method("session.list")
def _(rid, params: dict) -> dict:
    try:
        db = _get_db()
        # Show both TUI and CLI sessions — TUI is the successor to the CLI,
        # so users expect to resume their old CLI sessions here too.
        tui = db.list_sessions_rich(source="tui", limit=params.get("limit", 20))
        cli = db.list_sessions_rich(source="cli", limit=params.get("limit", 20))
        rows = sorted(tui + cli, key=lambda s: s.get("started_at") or 0, reverse=True)[:params.get("limit", 20)]
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
    os.environ["HERMES_SESSION_KEY"] = target
    os.environ["HERMES_INTERACTIVE"] = "1"
    try:
        db.reopen_session(target)
        messages = [
            {"role": m["role"], "text": m["content"] or ""}
            for m in db.get_messages(target)
            if m.get("role") in ("user", "assistant", "tool", "system")
            and isinstance(m.get("content"), str)
            and (m.get("content") or "").strip()
        ]
        history = [{"role": m["role"], "content": m["text"]} for m in messages]
        agent = _make_agent(sid, target, session_id=target)
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
    return err or _ok(rid, {"count": len(session.get("history", []))})


@method("session.undo")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    history, removed = session.get("history", []), 0
    while history and history[-1].get("role") in ("assistant", "tool"):
        history.pop(); removed += 1
    if history and history[-1].get("role") == "user":
        history.pop(); removed += 1
    return _ok(rid, {"removed": removed})


@method("session.compress")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    agent = session["agent"]
    try:
        if hasattr(agent, "compress_context"):
            agent.compress_context()
        return _ok(rid, {"status": "compressed", "usage": _get_usage(agent)})
    except Exception as e:
        return _err(rid, 5005, str(e))


@method("session.save")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    import time as _time
    filename = f"hermes_conversation_{_time.strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(filename, "w") as f:
            json.dump({"model": getattr(session["agent"], "model", ""), "messages": session.get("history", [])},
                      f, indent=2, ensure_ascii=False)
        return _ok(rid, {"file": filename})
    except Exception as e:
        return _err(rid, 5011, str(e))


@method("session.branch")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    db = _get_db()
    old_key = session["session_key"]
    history = session.get("history", [])
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
    os.environ["HERMES_SESSION_KEY"] = new_key
    try:
        agent = _make_agent(new_sid, new_key, session_id=new_key)
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
    _clear_pending()
    try:
        from tools.approval import resolve_gateway_approval
        resolve_gateway_approval(session["session_key"], "deny", resolve_all=True)
    except Exception:
        pass
    return _ok(rid, {"status": "interrupted"})


@method("terminal.resize")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    session["cols"] = int(params.get("cols", 80))
    return _ok(rid, {"cols": session["cols"]})


# ── Methods: prompt ──────────────────────────────────────────────────

@method("prompt.submit")
def _(rid, params: dict) -> dict:
    sid, text = params.get("session_id", ""), params.get("text", "")
    session = _sessions.get(sid)
    if not session:
        return _err(rid, 4001, "session not found")
    agent, history = session["agent"], session["history"]
    _emit("message.start", sid)

    def run():
        try:
            cols = session.get("cols", 80)
            streamer = make_stream_renderer(cols)
            images = session.pop("attached_images", [])
            prompt = _enrich_with_attached_images(text, images) if images else text

            def _stream(delta):
                payload = {"text": delta}
                if streamer and (r := streamer.feed(delta)) is not None:
                    payload["rendered"] = r
                _emit("message.delta", sid, payload)

            result = agent.run_conversation(
                prompt, conversation_history=list(history),
                stream_callback=_stream,
            )

            if isinstance(result, dict):
                if isinstance(result.get("messages"), list):
                    session["history"] = result["messages"]
                raw = result.get("final_response", "")
                status = "interrupted" if result.get("interrupted") else "error" if result.get("error") else "complete"
            else:
                raw = str(result)
                status = "complete"

            payload = {"text": raw, "usage": _get_usage(agent), "status": status}
            rendered = render_message(raw, cols)
            if rendered:
                payload["rendered"] = rendered
            _emit("message.complete", sid, payload)
        except Exception as e:
            _emit("error", sid, {"message": str(e)})

    threading.Thread(target=run, daemon=True).start()
    return _ok(rid, {"status": "streaming"})


@method("clipboard.paste")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    try:
        from datetime import datetime
        from hermes_cli.clipboard import has_clipboard_image, save_clipboard_image
    except Exception as e:
        return _err(rid, 5027, f"clipboard unavailable: {e}")

    session["image_counter"] = session.get("image_counter", 0) + 1
    img_dir = _hermes_home / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / f"clip_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session['image_counter']}.png"

    # Save-first: mirrors CLI keybinding path; more robust than has_image() precheck
    if not save_clipboard_image(img_path):
        msg = "Clipboard has image but extraction failed" if has_clipboard_image() else "No image found in clipboard"
        return _ok(rid, {"attached": False, "message": msg})

    session.setdefault("attached_images", []).append(str(img_path))
    return _ok(rid, {"attached": True, "path": str(img_path), "count": len(session["attached_images"])})


@method("prompt.background")
def _(rid, params: dict) -> dict:
    text, parent = params.get("text", ""), params.get("session_id", "")
    if not text:
        return _err(rid, 4012, "text required")
    task_id = f"bg_{uuid.uuid4().hex[:6]}"

    def run():
        try:
            from run_agent import AIAgent
            result = AIAgent(model=_resolve_model(), quiet_mode=True, platform="tui",
                             session_id=task_id, max_iterations=30).run_conversation(text)
            _emit("background.complete", parent, {"task_id": task_id,
                  "text": result.get("final_response", str(result)) if isinstance(result, dict) else str(result)})
        except Exception as e:
            _emit("background.complete", parent, {"task_id": task_id, "text": f"error: {e}"})

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
        try:
            from run_agent import AIAgent
            result = AIAgent(model=_resolve_model(), quiet_mode=True, platform="tui",
                             max_iterations=8, enabled_toolsets=[]).run_conversation(text, conversation_history=snapshot)
            _emit("btw.complete", sid, {"text": result.get("final_response", str(result)) if isinstance(result, dict) else str(result)})
        except Exception as e:
            _emit("btw.complete", sid, {"text": f"error: {e}"})

    threading.Thread(target=run, daemon=True).start()
    return _ok(rid, {"status": "running"})


# ── Methods: respond ─────────────────────────────────────────────────

def _respond(rid, params, key):
    r = params.get("request_id", "")
    ev = _pending.get(r)
    if not ev:
        return _err(rid, 4009, f"no pending {key} request")
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

    if key == "model":
        os.environ["HERMES_MODEL"] = value
        return _ok(rid, {"key": key, "value": value})

    if key == "verbose":
        cycle = ["off", "new", "all", "verbose"]
        if value and value != "cycle":
            os.environ["HERMES_VERBOSE"] = value
            return _ok(rid, {"key": key, "value": value})
        cur = os.environ.get("HERMES_VERBOSE", "all")
        try:
            idx = cycle.index(cur)
        except ValueError:
            idx = 2
        nv = cycle[(idx + 1) % len(cycle)]
        os.environ["HERMES_VERBOSE"] = nv
        return _ok(rid, {"key": key, "value": nv})

    if key == "yolo":
        nv = "0" if os.environ.get("HERMES_YOLO", "0") == "1" else "1"
        os.environ["HERMES_YOLO"] = nv
        return _ok(rid, {"key": key, "value": nv})

    if key == "reasoning":
        if value in ("show", "on"):
            os.environ["HERMES_SHOW_REASONING"] = "1"
            return _ok(rid, {"key": key, "value": "show"})
        if value in ("hide", "off"):
            os.environ.pop("HERMES_SHOW_REASONING", None)
            return _ok(rid, {"key": key, "value": "hide"})
        os.environ["HERMES_REASONING"] = value
        return _ok(rid, {"key": key, "value": value})

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
            elif key == "personality":
                cfg.setdefault("display", {})["personality"] = value if value not in ("none", "default", "neutral") else ""
                nv = value
            else:
                cfg.setdefault("display", {})[key] = value
                nv = value
            _save_cfg(cfg)
            if key == "skin":
                _emit("skin.changed", "", resolve_skin())
            return _ok(rid, {"key": key, "value": nv})
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
    return _err(rid, 4002, f"unknown config key: {key}")


# ── Methods: tools & system ──────────────────────────────────────────

@method("process.stop")
def _(rid, params: dict) -> dict:
    try:
        from tools.process_registry import ProcessRegistry
        return _ok(rid, {"killed": ProcessRegistry().kill_all()})
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

        skill_count = 0
        try:
            from agent.skill_commands import scan_skill_commands
            for k, info in sorted(scan_skill_commands().items()):
                d = str(info.get("description", "Skill"))
                all_pairs.append([k, d[:120] + ("…" if len(d) > 120 else "")])
                skill_count += 1
        except Exception:
            pass

        for cat in cat_order:
            categories.append({"name": cat, "pairs": cat_map[cat]})

        sub = {k: v[:] for k, v in SUBCOMMANDS.items()}
        return _ok(rid, {
            "pairs": all_pairs,
            "sub": sub,
            "canon": canon,
            "categories": categories,
            "skill_count": skill_count,
        })
    except Exception as e:
        return _err(rid, 5020, str(e))


def _cli_exec_blocked(argv: list[str]) -> str | None:
    """Return user hint if this argv must not run headless in the gateway process."""
    if not argv:
        return "bare `hermes` is interactive — use `/hermes chat -q …` or run `hermes` in another terminal"
    a0 = argv[0].lower()
    if a0 == "setup":
        return "`hermes setup` needs a full terminal — run it outside the Ink UI"
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
            return _ok(rid, {"type": "exec", "output": (r.stdout or r.stderr)[:4000]})
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

        if is_context and query.startswith(("file:", "folder:")):
            prefix_tag = query.split(":", 1)[0]
            path_part = query.split(":", 1)[1] or "."
        else:
            prefix_tag = ""
            path_part = query if not is_context else query

        expanded = os.path.expanduser(path_part)
        if expanded.endswith("/"):
            search_dir, match = expanded, ""
        else:
            search_dir = os.path.dirname(expanded) or "."
            match = os.path.basename(expanded)

        match_lower = match.lower()
        for entry in sorted(os.listdir(search_dir))[:200]:
            if match and not entry.lower().startswith(match_lower):
                continue
            if is_context and not prefix_tag and entry.startswith("."):
                continue
            full = os.path.join(search_dir, entry)
            is_dir = os.path.isdir(full)
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
    except Exception:
        pass

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

        completer = SlashCommandCompleter()
        doc = Document(text, len(text))
        items = [
            {"text": c.text, "display": c.display or c.text,
             "meta": to_plain_text(c.display_meta) if c.display_meta else ""}
            for c in completer.get_completions(doc, None)
        ][:30]
        return _ok(rid, {"items": items, "replace_from": text.rfind(" ") + 1 if " " in text else 1})
    except Exception:
        return _ok(rid, {"items": []})


# ── Methods: slash.exec ──────────────────────────────────────────────


def _mirror_slash_side_effects(sid: str, session: dict, command: str):
    """Apply side effects that must also hit the gateway's live agent."""
    parts = command.lstrip("/").split(None, 1)
    if not parts:
        return
    name, arg, agent = parts[0], (parts[1].strip() if len(parts) > 1 else ""), session.get("agent")

    try:
        if name == "model" and arg and agent:
            from hermes_cli.model_switch import switch_model
            result = switch_model(
                raw_input=arg,
                current_provider=getattr(agent, "provider", "") or "",
                current_model=getattr(agent, "model", "") or "",
                current_base_url=getattr(agent, "base_url", "") or "",
                current_api_key=getattr(agent, "api_key", "") or "",
            )
            if result.success:
                agent.switch_model(
                    new_model=result.new_model,
                    new_provider=result.target_provider,
                    api_key=result.api_key,
                    base_url=result.base_url,
                    api_mode=result.api_mode,
                )
                _emit("session.info", sid, _session_info(agent))
        elif name in ("personality", "prompt") and agent:
            cfg = _load_cfg()
            new_prompt = cfg.get("agent", {}).get("system_prompt", "") or ""
            agent.ephemeral_system_prompt = new_prompt or None
            agent._cached_system_prompt = None
        elif name == "compress" and agent:
            (getattr(agent, "compress_context", None) or getattr(agent, "context_compressor", agent).compress)()
        elif name == "reload-mcp" and agent and hasattr(agent, "reload_mcp_tools"):
            agent.reload_mcp_tools()
        elif name == "stop":
            from tools.process_registry import ProcessRegistry
            ProcessRegistry().kill_all()
    except Exception:
        pass


@method("slash.exec")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err

    cmd = params.get("command", "").strip()
    if not cmd:
        return _err(rid, 4004, "empty command")

    worker = session.get("slash_worker")
    if not worker:
        try:
            worker = _SlashWorker(session["session_key"], getattr(session.get("agent"), "model", _resolve_model()))
            session["slash_worker"] = worker
        except Exception as e:
            return _err(rid, 5030, f"slash worker start failed: {e}")

    try:
        output = worker.run(cmd)
        _mirror_slash_side_effects(params.get("session_id", ""), session, cmd)
        return _ok(rid, {"output": output or "(no output)"})
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
        return _ok(rid, {"enabled": os.environ.get("HERMES_VOICE", "0") == "1"})
    if action in ("on", "off"):
        os.environ["HERMES_VOICE"] = "1" if action == "on" else "0"
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
        import time
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
    if not target:
        return _err(rid, 4014, "hash required")
    try:
        return _ok(rid, _with_checkpoints(session, lambda mgr, cwd: mgr.restore(cwd, target)))
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
        r = _with_checkpoints(session, lambda mgr, cwd: mgr.diff(cwd, target))
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
        os.environ["BROWSER_CDP_URL"] = url
        try:
            from tools.browser_tool import cleanup_all_browsers
            cleanup_all_browsers()
        except Exception:
            pass
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
    except Exception:
        return _ok(rid, {"plugins": []})


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
