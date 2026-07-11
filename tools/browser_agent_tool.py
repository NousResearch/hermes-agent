"""
Browser Agent Tool Module — AI-driven browser automation via CloakBrowser + browser-use.

Exposes two tools in the ``browser_agent`` toolset:

``browser_task``
    Spawns a browser-use worker subprocess that connects to the running
    CloakBrowser (managed by stealth-browser.service) via CDP.  The worker
    communicates over JSON-lines stdio.  Mid-flight user input (missing form
    data, choices) is relayed through the existing ``completion_queue`` event
    rail — the tool handler pushes an ``info_needed`` event, blocks on a
    ``threading.Event``, and the parent Hermes (which HAS ``clarify_callback``)
    asks the user then calls ``browser_respond`` to unblock.

``browser_respond``
    Delivers a user's answer to a pending ``info_request`` from a background
    browser task.  Sets the ``threading.Event`` that unblocks the waiting
    ``browser_task`` handler so the answer can be forwarded to the worker.
"""

import json
import logging
import os
import subprocess
import threading
from typing import Any, Dict, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state for mid-flight communication
# ---------------------------------------------------------------------------
# request_id → {"event": threading.Event, "answer": str | None}
_pending_responses: Dict[str, Dict[str, Any]] = {}
_pending_lock = threading.Lock()

# Active worker subprocesses — keyed by task_id for interrupt cleanup
_active_workers: Dict[str, subprocess.Popen] = {}
_workers_lock = threading.Lock()


def kill_all_workers() -> None:
    """Kill all active browser workers — called on /stop or shutdown."""
    import signal as _signal
    import threading
    with _workers_lock:
        workers = list(_active_workers.items())
    
    def _kill_proc(task_id, proc):
        # Try graceful cancellation via stdin command
        try:
            if proc.stdin:
                proc.stdin.write(json.dumps({"type": "cancel"}) + "\n")
                proc.stdin.flush()
        except Exception:
            pass
        
        # Wait up to 3s for graceful exit
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            # Escalating to SIGINT (KeyboardInterrupt) to trigger worker finally blocks
            try:
                os.killpg(os.getpgid(proc.pid), _signal.SIGINT)
                proc.wait(timeout=2)
            except Exception:
                # Escalating to SIGKILL
                if proc.poll() is None:
                    try:
                        os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)
                        proc.wait(timeout=1)
                    except Exception:
                        pass
        except Exception:
            pass
            
        with _workers_lock:
            _active_workers.pop(task_id, None)

    # Spawn daemon threads to avoid blocking Hermes's main thread during /stop
    for task_id, proc in workers:
        t = threading.Thread(target=_kill_proc, args=(task_id, proc), daemon=True)
        t.start()


def _cleanup_stale_workers() -> None:
    """Clean up references to terminated subprocesses to prevent leaks."""
    with _workers_lock:
        stale = [tid for tid, proc in _active_workers.items() if proc.poll() is not None]
        for tid in stale:
            _active_workers.pop(tid, None)

# ---------------------------------------------------------------------------
# Worker path — isolated uv env with browser-use + cloakbrowser
# ---------------------------------------------------------------------------
WORKER_DIR = os.path.expanduser("~/tool/browser-use-cloak-stealth")
WORKER_TIMEOUT = 900  # seconds — wall-clock limit for the subprocess


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_session_key() -> str:
    """Return the current session key for completion_queue routing."""
    try:
        from tools.approval import get_current_session_key
        return get_current_session_key(default="") or ""
    except Exception:
        return ""


def _push_info_needed(request_id: str, question: str, choices, goal: str) -> None:
    """Push an info_needed event to the completion_queue."""
    from tools.process_registry import process_registry
    process_registry.completion_queue.put({
        "type": "async_delegation",
        "status": "info_needed",
        "request_id": request_id,
        "session_key": _get_session_key(),
        "question": question,
        "choices": choices,
        "goal": goal,
    })


def _push_escalation(request_id: str, reason: str, tunnel_url: str,
                     timeout: int, goal: str) -> None:
    """Push an escalation event to the completion_queue."""
    from tools.process_registry import process_registry
    process_registry.completion_queue.put({
        "type": "async_delegation",
        "status": "escalation",
        "request_id": request_id,
        "session_key": _get_session_key(),
        "reason": reason,
        "tunnel_url": tunnel_url,
        "timeout": timeout,
        "goal": goal,
    })


# ---------------------------------------------------------------------------
# browser_task handler
# ---------------------------------------------------------------------------

def handle_browser_task(args: Dict[str, Any], **kw) -> str:
    """Spawn the browser-use worker and relay events until completion."""
    import logging
    _logger = logging.getLogger(__name__)
    
    # Debug: log what kwargs we received
    _logger.info(f"browser_task called with kw keys: {list(kw.keys())}")
    
    _toolsets = kw.get("enabled_toolsets") or []
    # Auto-delegate check: if explicitly delegated (has subagent metadata) or running inside a child agent (which has restricted toolsets)
    is_delegated = (
        any(k in kw for k in ("task_index", "subagent_id", "depth"))
        or ("browser_agent" in _toolsets and len(_toolsets) <= 2)
    )
    _logger.info(f"is_delegated={is_delegated} (keys={list(kw.keys())}, toolsets={_toolsets})")
    
    if not is_delegated:
        # Direct call detected — return instruction for the model to delegate
        goal_text = args.get("task", "") or args.get("goal", "")
        constraints = args.get("constraints", "")
        timeout = args.get("timeout", "")
        
        # Build the full goal for the child agent
        child_goal = f"Use the browser_task tool to: {goal_text}"
        if constraints:
            child_goal += f"\nConstraints: {constraints}"
        if timeout:
            child_goal += f"\nTimeout: {timeout}s"
        
        return json.dumps({
            "status": "redirect",
            "message": (
                f"To run this browser task without blocking, call delegate_task with:\n"
                f"  goal: '{child_goal}'\n"
                f"  background: true\n"
                f"  toolsets: ['browser_agent']\n\n"
                f"IMPORTANT: The child agent MUST call browser_task(task='{goal_text}') to use the stealth browser. "
                f"Do NOT run browser-use directly via terminal/python — it won't connect to VNC."
            ),
        }, ensure_ascii=False)
    
    task_id = kw.get("task_id", "")
    goal = args.get("task", "") or args.get("goal", "")
    constraints = args.get("constraints", "")
    timeout = int(args.get("timeout", WORKER_TIMEOUT))

    if not goal:
        return json.dumps({"error": "task (goal) is required"}, ensure_ascii=False)
    
    # Clean up only terminated worker processes.
    _cleanup_stale_workers()
    
    # Resolve LLM config from Hermes config so the worker uses the same endpoint
    _llm_base_url = os.environ.get("BROWSER_LLM_BASE_URL", "")
    _llm_api_key = os.environ.get("BROWSER_LLM_API_KEY", "")
    _llm_model = ""
    try:
        from hermes_cli.config import load_config
        _cfg = load_config()
        _model_cfg = _cfg.get("model", {})
        if not _llm_base_url and _model_cfg.get("base_url"):
            _llm_base_url = _model_cfg["base_url"]
        if _model_cfg.get("default"):
            _llm_model = _model_cfg["default"]
        _key_env = _model_cfg.get("key_env", "")
        if _key_env:
            _llm_api_key = os.environ.get(_key_env, _llm_api_key or "local")
    except Exception:
        pass
    if not _llm_api_key:
        _llm_api_key = "local"

    task_spec = json.dumps({
        "goal": goal,
        "constraints": constraints,
        "timeout": timeout,
        "task_id": task_id,
        "llm_base_url": _llm_base_url,
        "llm_api_key": _llm_api_key,
        "llm_model": _llm_model,
    })

    # Spawn worker subprocess in the isolated uv env
    try:
        proc = subprocess.Popen(
            ["uv", "run", "python", "browser_worker.py", task_spec],
            cwd=WORKER_DIR,
            env={**os.environ, "DISPLAY": ":99"},
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
            start_new_session=True,  # new process group for killpg cleanup
        )
    except Exception as exc:
        return json.dumps(
            {"error": f"Failed to spawn browser worker: {exc}"},
            ensure_ascii=False,
        )

    # Start a thread to capture stderr (for crash diagnostics)
    stderr_lines = []
    def _read_stderr():
        try:
            for line in proc.stderr:
                stderr_lines.append(line.rstrip())
        except Exception:
            pass
    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    stderr_thread.start()

    # Register for interrupt cleanup — /stop can find and kill us
    with _workers_lock:
        _active_workers[task_id] = proc

    status_messages = []

    try:
        # Non-blocking stdout read loop — checks for interrupt every 2s
        import selectors
        sel = selectors.DefaultSelector()
        sel.register(proc.stdout, selectors.EVENT_READ)

        while True:
            # Check if the process was killed externally (e.g. /stop)
            if proc.poll() is not None:
                break

            # Wait for data with 2s timeout (allows interrupt checking)
            ready = sel.select(timeout=2)
            if not ready:
                continue

            line = proc.stdout.readline()
            if not line:  # EOF — process exited
                break
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue

            evt_type = evt.get("type")

            if evt_type == "status":
                msg = evt.get("message", "")
                status_messages.append(msg)
                logger.info("[browser_task %s] %s", task_id, msg)

            elif evt_type == "progress":
                step = evt.get("step", 0)
                max_steps = evt.get("max_steps", 0)
                pct = evt.get("percent", 0)
                status_messages.append(f"Step {step}/{max_steps} ({pct}%)")

            elif evt_type == "info_request":
                request_id = evt.get("request_id") or uuid4().hex[:8]
                wait_event = threading.Event()
                with _pending_lock:
                    _pending_responses[request_id] = {
                        "event": wait_event,
                        "answer": None,
                    }
                # Push to completion_queue — drain injects as new turn for parent
                _push_info_needed(
                    request_id=request_id,
                    question=evt.get("question", ""),
                    choices=evt.get("choices"),
                    goal=goal,
                )
                # Block until browser_respond sets the Event (interrupt-aware)
                interrupted = False
                while not wait_event.wait(timeout=2):
                    if proc.poll() is not None:
                        interrupted = True
                        break
                if not interrupted:
                    with _pending_lock:
                        answer = _pending_responses.pop(request_id, {}).get("answer")
                    if answer is not None and proc.stdin:
                        try:
                            proc.stdin.write(json.dumps({
                                "type": "clarify_response",
                                "request_id": request_id,
                                "answer": answer,
                            }) + "\n")
                            proc.stdin.flush()
                        except (BrokenPipeError, OSError):
                            pass
                else:
                    with _pending_lock:
                        _pending_responses.pop(request_id, None)

            elif evt_type == "escalation_request":
                request_id = evt.get("request_id") or uuid4().hex[:8]
                wait_event = threading.Event()
                with _pending_lock:
                    _pending_responses[request_id] = {
                        "event": wait_event,
                        "answer": None,
                    }
                _push_escalation(
                    request_id=request_id,
                    reason=evt.get("reason", ""),
                    tunnel_url=evt.get("tunnel_url", ""),
                    timeout=evt.get("timeout", 120),
                    goal=goal,
                )
                # Block until browser_respond sets the Event (interrupt-aware)
                interrupted = False
                while not wait_event.wait(timeout=2):
                    if proc.poll() is not None:
                        interrupted = True
                        break
                if not interrupted:
                    with _pending_lock:
                        answer = _pending_responses.pop(request_id, {}).get("answer")
                    if answer is not None and proc.stdin:
                        try:
                            proc.stdin.write(json.dumps({
                                "type": "clarify_response",
                                "request_id": request_id,
                                "answer": answer,
                            }) + "\n")
                            proc.stdin.flush()
                        except (BrokenPipeError, OSError):
                            pass
                else:
                    with _pending_lock:
                        _pending_responses.pop(request_id, None)

            elif evt_type == "result":
                # Worker finished — return its result
                summary = evt.get("summary", "")
                if status_messages:
                    summary = "\n".join(status_messages[-5:]) + "\n\n" + summary
                return json.dumps(evt, ensure_ascii=False)

            elif evt_type == "error":
                return json.dumps(evt, ensure_ascii=False)

            elif evt_type == "heartbeat":
                pass  # worker is alive

            elif evt_type == "ready":
                status_messages.append("Worker ready")

    except Exception as exc:
        return json.dumps(
            {"error": f"Browser task failed: {exc}"},
            ensure_ascii=False,
        )
    finally:
        # Unregister from active workers
        with _workers_lock:
            _active_workers.pop(task_id, None)
        # Always terminate the subprocess — escalate to SIGKILL if needed
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=3)
            except Exception:
                pass
        # Kill the entire process group (catches any grandchildren)
        try:
            import signal as _signal
            os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)
        except Exception:
            pass

    # Include stderr in the error message if the worker crashed
    stderr_output = "\n".join(stderr_lines[-20:]) if stderr_lines else ""
    if stderr_output:
        logger.error("[browser_task %s] Worker stderr:\n%s", task_id, stderr_output)
    return json.dumps(
        {"error": "Worker exited without result", "status_log": status_messages[-10:], "stderr": stderr_output[-2000:]},
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# browser_respond handler
# ---------------------------------------------------------------------------

def handle_browser_respond(args: Dict[str, Any], **kw) -> str:
    """Deliver a user's answer to a pending browser info_request."""
    request_id = args.get("request_id", "")
    answer = args.get("answer", "")

    if not request_id:
        return json.dumps({"error": "request_id is required"}, ensure_ascii=False)

    with _pending_lock:
        entry = _pending_responses.get(request_id)

    if not entry:
        return json.dumps(
            {"error": f"No pending request with id '{request_id}'"},
            ensure_ascii=False,
        )

    entry["answer"] = answer
    entry["event"].set()
    return json.dumps(
        {"status": "delivered", "request_id": request_id},
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def check_browser_agent_requirements() -> bool:
    """Browser agent requires the stealth-browser.service and worker env."""
    return os.path.isdir(WORKER_DIR)


# =============================================================================
# OpenAI Function-Calling Schemas
# =============================================================================

BROWSER_TASK_SCHEMA = {
    "name": "browser_task",
    "description": (
        "Run an AI-driven browser automation task using CloakBrowser + browser-use. "
        "The worker connects to a stealth browser instance and performs the given "
        "task autonomously (navigate, fill forms, click buttons, extract data, etc.).\n\n"
        "IMPORTANT: This tool is designed to be called via delegate_task(background=True) "
        "so it runs in the background without blocking the conversation. "
        "Example: delegate_task(goal='browser_task: Fill the form on example.com', "
        "background=True, toolsets=['browser_agent']).\n\n"
        "If the worker needs information mid-flight (e.g. missing form data), "
        "you will receive a [BROWSER TASK NEEDS INPUT] message. Respond by calling "
        "clarify() to ask the user, then browser_respond() to unblock the worker."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The browser task goal (e.g. 'Fill out the contact form on example.com').",
            },
            "constraints": {
                "type": "string",
                "description": "Optional constraints or instructions for the task.",
            },
            "timeout": {
                "type": "integer",
                "description": "Maximum seconds for the task (default: 300).",
                "minimum": 30,
                "maximum": 1800,
            },
        },
        "required": ["task"],
    },
}

BROWSER_RESPOND_SCHEMA = {
    "name": "browser_respond",
    "description": (
        "Deliver a user's answer to a background browser task that requested input. "
        "Call this after asking the user via the clarify tool. The request_id comes "
        "from the [BROWSER TASK NEEDS INPUT] message."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "request_id": {
                "type": "string",
                "description": "The request_id from the BROWSER TASK NEEDS INPUT message.",
            },
            "answer": {
                "type": "string",
                "description": "The user's answer to the question.",
            },
        },
        "required": ["request_id", "answer"],
    },
}


# =============================================================================
# Registry
# =============================================================================
from tools.registry import registry, tool_error

registry.register(
    name="browser_task",
    toolset="browser_agent",
    schema=BROWSER_TASK_SCHEMA,
    handler=lambda args, **kw: handle_browser_task(args, **kw),
    check_fn=check_browser_agent_requirements,
    emoji="🌐",
)

registry.register(
    name="browser_respond",
    toolset="browser_agent",
    schema=BROWSER_RESPOND_SCHEMA,
    handler=lambda args, **kw: handle_browser_respond(args, **kw),
    check_fn=lambda: True,
    emoji="💬",
)
