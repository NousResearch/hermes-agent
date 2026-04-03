import json
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from hermes_constants import get_hermes_home
from hermes_cli.env_loader import load_hermes_dotenv

_hermes_home = get_hermes_home()
load_hermes_dotenv(hermes_home=_hermes_home, project_env=Path(__file__).parent.parent / ".env")

_sessions: dict[str, dict] = {}
_methods:  dict[str, callable] = {}
_clarify_pending: dict[str, threading.Event] = {}
_clarify_answers: dict[str, str] = {}


# ── Wire ─────────────────────────────────────────────────────────────

def _emit(event_type: str, sid: str, payload: dict | None = None):
    params = {"type": event_type, "session_id": sid}
    if payload:
        params["payload"] = payload
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "method": "event", "params": params}) + "\n")
    sys.stdout.flush()


def _ok(req_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _err(req_id, code: int, msg: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": msg}}


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


# ── Helpers ──────────────────────────────────────────────────────────

def resolve_skin() -> dict:
    try:
        import yaml
        from hermes_cli.skin_engine import init_skin_from_config, get_active_skin
        cfg_path = _hermes_home / "config.yaml"
        cfg = {}
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
        init_skin_from_config(cfg)
        skin = get_active_skin()
        return {"name": skin.name, "colors": skin.colors, "branding": skin.branding}
    except Exception:
        return {}


def _resolve_model() -> str:
    env = os.environ.get("HERMES_MODEL", "")
    if env:
        return env
    try:
        import yaml
        cfg_path = _hermes_home / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                m = (yaml.safe_load(f) or {}).get("model", "")
            if isinstance(m, dict):
                return m.get("default", "")
            if isinstance(m, str):
                return m
    except Exception:
        pass
    return "anthropic/claude-sonnet-4"


def _get_usage(agent) -> dict:
    ga = lambda k, fb=None: getattr(agent, k, 0) or (getattr(agent, fb, 0) if fb else 0)
    return {
        "input":  ga("session_input_tokens", "session_prompt_tokens"),
        "output": ga("session_output_tokens", "session_completion_tokens"),
        "total":  ga("session_total_tokens"),
        "calls":  ga("session_api_calls"),
    }


def _collect_session_info(agent) -> dict:
    info: dict = {"model": getattr(agent, "model", ""), "tools": {}, "skills": {}}
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
    return info


def _make_clarify_cb(sid: str):
    def cb(question: str, choices: list | None) -> str:
        rid = uuid.uuid4().hex[:8]
        ev = threading.Event()
        _clarify_pending[rid] = ev
        _emit("clarify.request", sid, {"request_id": rid, "question": question, "choices": choices})
        ev.wait(timeout=300)
        _clarify_pending.pop(rid, None)
        return _clarify_answers.pop(rid, "")
    return cb


def _register_approval_notify(sid: str, session_key: str):
    try:
        from tools.approval import register_gateway_notify
        register_gateway_notify(session_key, lambda data: _emit("approval.request", sid, data))
    except Exception:
        pass


# ── Methods ──────────────────────────────────────────────────────────

@method("session.create")
def _(req_id, params: dict) -> dict:
    sid = uuid.uuid4().hex[:8]
    session_key = f"tui-{sid}"

    os.environ["HERMES_SESSION_KEY"] = session_key
    os.environ["HERMES_INTERACTIVE"] = "1"

    try:
        from run_agent import AIAgent
        agent = AIAgent(
            model=_resolve_model(),
            quiet_mode=True,
            platform="tui",
            tool_start_callback=lambda tc_id, name, args: _emit("tool.start", sid, {"tool_id": tc_id, "name": name}),
            tool_complete_callback=lambda tc_id, name, args, result: _emit("tool.complete", sid, {"tool_id": tc_id, "name": name}),
            tool_progress_callback=lambda name, preview, args: _emit("tool.progress", sid, {"name": name, "preview": preview}),
            tool_gen_callback=lambda name: _emit("tool.generating", sid, {"name": name}),
            thinking_callback=lambda text: _emit("thinking.delta", sid, {"text": text}),
            reasoning_callback=lambda text: _emit("reasoning.delta", sid, {"text": text}),
            status_callback=lambda text: _emit("status.update", sid, {"text": text}),
            clarify_callback=_make_clarify_cb(sid),
        )
        _sessions[sid] = {"agent": agent, "session_key": session_key, "history": []}
    except Exception as e:
        return _err(req_id, 5000, f"agent init failed: {e}")

    _register_approval_notify(sid, session_key)

    from tools.approval import load_permanent_allowlist
    load_permanent_allowlist()

    _emit("session.info", sid, _collect_session_info(agent))
    return _ok(req_id, {"session_id": sid})


@method("prompt.submit")
def _(req_id, params: dict) -> dict:
    sid, text = params.get("session_id", ""), params.get("text", "")
    session = _sessions.get(sid)
    if not session:
        return _err(req_id, 4001, "session not found")

    agent = session["agent"]
    history = session["history"]
    _emit("message.start", sid)

    def run():
        try:
            result = agent.run_conversation(
                text,
                conversation_history=list(history),
                stream_callback=lambda delta: _emit("message.delta", sid, {"text": delta}),
            )

            if isinstance(result, dict):
                returned_msgs = result.get("messages")
                if isinstance(returned_msgs, list):
                    session["history"] = returned_msgs
                final = result.get("final_response", "")
                status = "interrupted" if result.get("interrupted") else "error" if result.get("error") else "complete"
                _emit("message.complete", sid, {
                    "text": final or "",
                    "usage": _get_usage(agent),
                    "status": status,
                })
            else:
                _emit("message.complete", sid, {"text": str(result), "usage": _get_usage(agent), "status": "complete"})

        except Exception as e:
            _emit("error", sid, {"message": str(e)})

    threading.Thread(target=run, daemon=True).start()
    return _ok(req_id, {"status": "streaming"})


@method("clarify.respond")
def _(req_id, params: dict) -> dict:
    rid = params.get("request_id", "")
    ev = _clarify_pending.get(rid)
    if not ev:
        return _err(req_id, 4003, "no pending clarify request")
    _clarify_answers[rid] = params.get("answer", "")
    ev.set()
    return _ok(req_id, {"status": "ok"})


@method("approval.respond")
def _(req_id, params: dict) -> dict:
    sid = params.get("session_id", "")
    choice = params.get("choice", "deny")

    session = _sessions.get(sid)
    if not session:
        return _err(req_id, 4001, "session not found")

    try:
        from tools.approval import resolve_gateway_approval
        n = resolve_gateway_approval(session["session_key"], choice, resolve_all=params.get("all", False))
        return _ok(req_id, {"resolved": n})
    except Exception as e:
        return _err(req_id, 5004, str(e))


@method("session.usage")
def _(req_id, params: dict) -> dict:
    session = _sessions.get(params.get("session_id", ""))
    if not session:
        return _err(req_id, 4001, "session not found")
    return _ok(req_id, _get_usage(session["agent"]))


@method("session.history")
def _(req_id, params: dict) -> dict:
    session = _sessions.get(params.get("session_id", ""))
    if not session:
        return _err(req_id, 4001, "session not found")
    return _ok(req_id, {"count": len(session.get("history", []))})


@method("session.undo")
def _(req_id, params: dict) -> dict:
    session = _sessions.get(params.get("session_id", ""))
    if not session:
        return _err(req_id, 4001, "session not found")
    history = session.get("history", [])
    removed = 0
    while history and history[-1].get("role") in ("assistant", "tool"):
        history.pop(); removed += 1
    if history and history[-1].get("role") == "user":
        history.pop(); removed += 1
    return _ok(req_id, {"removed": removed})


@method("session.compress")
def _(req_id, params: dict) -> dict:
    session = _sessions.get(params.get("session_id", ""))
    if not session:
        return _err(req_id, 4001, "session not found")
    agent = session["agent"]
    try:
        if hasattr(agent, "compress_context"):
            agent.compress_context()
        return _ok(req_id, {"status": "compressed", "usage": _get_usage(agent)})
    except Exception as e:
        return _err(req_id, 5005, str(e))


@method("config.set")
def _(req_id, params: dict) -> dict:
    key, value = params.get("key", ""), params.get("value", "")

    if key == "model":
        os.environ["HERMES_MODEL"] = value
        return _ok(req_id, {"key": key, "value": value})

    if key == "skin":
        try:
            import yaml
            cfg_path = _hermes_home / "config.yaml"
            cfg = {}
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f) or {}
            cfg["skin"] = value
            with open(cfg_path, "w") as f:
                yaml.safe_dump(cfg, f)
            return _ok(req_id, {"key": key, "value": value})
        except Exception as e:
            return _err(req_id, 5001, str(e))

    return _err(req_id, 4002, f"unknown config key: {key}")


@method("session.interrupt")
def _(req_id, params: dict) -> dict:
    session = _sessions.get(params.get("session_id", ""))
    if not session:
        return _err(req_id, 4001, "session not found")

    if hasattr(session["agent"], "interrupt"):
        session["agent"].interrupt()

    for rid, ev in list(_clarify_pending.items()):
        _clarify_answers[rid] = ""
        ev.set()

    try:
        from tools.approval import resolve_gateway_approval
        resolve_gateway_approval(session["session_key"], "deny", resolve_all=True)
    except Exception:
        pass

    return _ok(req_id, {"status": "interrupted"})


@method("shell.exec")
def _(req_id, params: dict) -> dict:
    cmd = params.get("command", "")
    if not cmd:
        return _err(req_id, 4004, "empty command")

    try:
        from tools.approval import detect_dangerous_command
        is_dangerous, _, description = detect_dangerous_command(cmd)
        if is_dangerous:
            return _err(req_id, 4005, f"blocked: {description}. Use the agent for dangerous commands (it has approval flow).")
    except ImportError:
        pass

    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=os.getcwd())
        return _ok(req_id, {"stdout": r.stdout[-4000:], "stderr": r.stderr[-2000:], "code": r.returncode})
    except subprocess.TimeoutExpired:
        return _err(req_id, 5002, "command timed out (30s)")
    except Exception as e:
        return _err(req_id, 5003, str(e))
