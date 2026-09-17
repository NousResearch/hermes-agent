"""Persistent external-agent delegation sessions.

`delegate_task` is intentionally a Hermes->Hermes child-agent primitive.  This
module provides the complementary session primitive for external agents whose
native protocol already owns conversation state.  Backends are configurable
via the ``backend`` argument (``"pi"`` default | ``"opencode"``) or the
``HERMES_DELEGATE_SESSION_BACKEND`` environment variable: Hermes keeps one
native process/session alive per record and can send follow-up turns, steer an
active turn, answer backend questions autonomously, inspect messages, and
stop/resume the native session without wrapping it in a child AIAgent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from agent.opencode_client import OpenCodeClient
from agent.pi_rpc_client import PiRPCClient, pending_question_for_owner
from agent.runtime_cwd import resolve_agent_cwd
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

_SESSION_LOCK = threading.RLock()
_SESSION_CONDITION = threading.Condition(_SESSION_LOCK)
_SESSIONS: Dict[str, Dict[str, Any]] = {}
_MAX_TEXT = 12_000
_MAX_DURABLE_SESSIONS = 500

_KNOWN_BACKENDS = ("pi", "opencode")

# Owners without a conversation id fall back to a process-local handle. Such
# handles are meaningless in a later process, so they must never authorize
# durable metadata (an id() collision would otherwise grant access).
_PROCESS_LOCAL_OWNER_PREFIX = "agent:"


def _default_backend() -> str:
    raw = os.environ.get("HERMES_DELEGATE_SESSION_BACKEND", "").strip().lower()
    return raw if raw in _KNOWN_BACKENDS else "pi"


def _resolve_backend(requested: Any, saved: Any = None) -> Optional[str]:
    """Resolve the effective backend; explicit argument wins over metadata."""
    raw = str(requested or "").strip().lower()
    if raw:
        if raw not in _KNOWN_BACKENDS:
            return None
        return raw
    if isinstance(saved, str) and saved.strip().lower() in _KNOWN_BACKENDS:
        return saved.strip().lower()
    return _default_backend()


def _backend_client_class(backend: str):
    if backend == "opencode":
        return OpenCodeClient
    return PiRPCClient


def _session_store_root() -> Path:
    """Profile-scoped durable metadata store (IDs/cwd only; never prompt text)."""
    try:
        from hermes_constants import get_hermes_home

        root = Path(get_hermes_home())
    except Exception:
        root = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    return root / "cache" / "delegate-sessions"


def _metadata_path(session_id: str) -> Path:
    digest = hashlib.sha256(session_id.encode("utf-8", errors="replace")).hexdigest()
    return _session_store_root() / f"{digest}.json"


def _metadata_snapshot(record: Dict[str, Any]) -> dict[str, Any]:
    return {
        "version": 3,
        "backend": record.get("backend") or "pi",
        "session_id": record.get("session_id"),
        "native_session_id": record.get("native_session_id")
        or record.get("session_id"),
        "pi_session_id": record.get("native_session_id")
        or record.get("session_id"),  # v1 read-alias
        "owner": record.get("owner"),
        "owner_scope": record.get("owner_scope"),
        "cwd": record.get("cwd"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
    }


def _metadata_files_newest(root: Path) -> list[Path]:
    """Return readable metadata files newest-first, tolerating concurrent churn."""
    ranked: list[tuple[float, Path]] = []
    try:
        candidates = list(root.glob("*.json"))
    except OSError:
        return []
    for candidate in candidates:
        try:
            ranked.append((candidate.stat().st_mtime, candidate))
        except OSError:
            continue
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [path for _mtime, path in ranked]


def _prune_durable_metadata(root: Path) -> None:
    """Bound Hermes' delegate-session metadata cache without touching Pi history."""
    for stale in _metadata_files_newest(root)[_MAX_DURABLE_SESSIONS:]:
        try:
            stale.unlink()
        except OSError:
            logger.debug(
                "Could not prune stale delegate-session metadata %s",
                stale,
                exc_info=True,
            )


def _persist_metadata(record: Dict[str, Any]) -> None:
    """Persist enough metadata to reopen the native Pi session after restart."""
    session_id = str(record.get("session_id") or "").strip()
    if not session_id:
        return
    path = _metadata_path(session_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            path.parent.chmod(0o700)
        except OSError:
            pass
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(
            json.dumps(_metadata_snapshot(record), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        try:
            tmp.chmod(0o600)
        except OSError:
            pass
        tmp.replace(path)
        _prune_durable_metadata(path.parent)
    except OSError:
        logger.debug(
            "Could not persist delegate-session metadata for %s",
            session_id,
            exc_info=True,
        )


def _load_metadata(session_id: str) -> dict[str, Any] | None:
    try:
        data = json.loads(_metadata_path(session_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("session_id") != session_id:
        return None
    return data


def _scope_for_workspace(cwd: str | Path | None = None) -> str | None:
    """Stable profile+workspace authorization scope, independent of conversation id."""
    try:
        workspace = Path(cwd).expanduser() if cwd is not None else resolve_agent_cwd()
        workspace = workspace.resolve()
        if not workspace.is_dir():
            return None
        profile_root = _session_store_root().resolve().parent
    except (OSError, RuntimeError, ValueError):
        return None
    raw = f"{profile_root}\0{workspace}".encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(raw).hexdigest()


def _metadata_authorized(
    data: dict[str, Any], parent_agent: Any, caller_scope: str | None = None
) -> bool:
    """Authorize durable metadata: conversation owner or profile/workspace scope."""
    owner = str(data.get("owner") or "").strip()
    if owner and not owner.startswith(_PROCESS_LOCAL_OWNER_PREFIX):
        # The conversation that started the session keeps durable control even
        # if its resolved cwd later moved. Process-local "agent:<id>" fallbacks
        # are not stable across restarts and never authorize durable metadata.
        if owner == _owner_key(parent_agent):
            return True
    if caller_scope is None:
        caller_scope = _scope_for_workspace()
    if not caller_scope:
        return False
    saved_scope = str(data.get("owner_scope") or "").strip()
    if saved_scope:
        return saved_scope == caller_scope
    # v1/v2 metadata predates owner_scope. Migrate safely from its durable cwd;
    # the legacy conversation owner alone is not enough to cross workspaces.
    saved_cwd = str(data.get("cwd") or "").strip()
    return bool(saved_cwd and _scope_for_workspace(saved_cwd) == caller_scope)


def _durable_rows_for_caller(
    parent_agent: Any, caller_scope: str | None = None
) -> list[dict[str, Any]]:
    root = _session_store_root()
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in _metadata_files_newest(root)[:_MAX_DURABLE_SESSIONS]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            not isinstance(data, dict)
            or not data.get("session_id")
            or not _metadata_authorized(data, parent_agent, caller_scope)
        ):
            continue
        rows.append(data)
    return rows


def check_delegate_session_requirements() -> bool:
    pi_available = bool(
        shutil.which("pi") or Path.home().joinpath(".local", "bin", "pi").is_file()
    )
    opencode_available = bool(
        os.environ.get("HERMES_OPENCODE_SERVER_URL", "").strip()
        or shutil.which("opencode")
        or Path.home().joinpath(".local", "bin", "opencode").is_file()
    )
    return pi_available or opencode_available


def _owner_key(parent_agent: Any) -> str:
    durable = str(getattr(parent_agent, "session_id", "") or "").strip()
    return durable or f"{_PROCESS_LOCAL_OWNER_PREFIX}{id(parent_agent)}"


def _bounded(value: Any, maximum: int = _MAX_TEXT) -> str:
    text = str(value or "")
    return text if len(text) <= maximum else text[: maximum - 3] + "..."


def _bounded_edges(value: Any, maximum: int = _MAX_TEXT) -> str:
    """Bound text while preserving structured trailers at the end.

    Delegate turns commonly put machine-readable completion envelopes after a
    human-readable explanation. Prefix-only truncation can therefore erase the
    only phase_result even though the delegate emitted it correctly.
    """
    text = str(value or "")
    if len(text) <= maximum:
        return text
    marker = "\n...[middle truncated]...\n"
    available = max(0, maximum - len(marker))
    head = available // 3
    tail = available - head
    return text[:head] + marker + text[-tail:]


def _message_text(value: Any) -> str:
    """Best-effort text extraction for recent supervising-agent context."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_message_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("text", "content", "output", "result"):
            if key in value:
                text = _message_text(value.get(key))
                if text:
                    return text
    return ""


def _parent_context_excerpt(parent_agent: Any, maximum: int = 24_000) -> str:
    """Return bounded recent conversation/tool context for Hermes auto-answers."""
    history = getattr(parent_agent, "_session_messages", None)
    if not isinstance(history, list):
        return ""
    chunks: list[str] = []
    used = 0
    for message in reversed(history[-60:]):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "context").strip().lower()
        if role == "system":
            continue
        text = _message_text(message.get("content"))
        if not text and role == "tool":
            text = _message_text(message)
        text = text.strip()
        if not text:
            continue
        chunk = f"{role.upper()}: {_bounded(text, 6000)}"
        if used + len(chunk) + 2 > maximum:
            remaining = maximum - used
            if remaining > 200:
                chunks.append(chunk[:remaining])
            break
        chunks.append(chunk)
        used += len(chunk) + 2
    return "\n\n".join(reversed(chunks))


def _parent_main_runtime(parent_agent: Any) -> dict[str, Any] | None:
    getter = getattr(parent_agent, "_current_main_runtime", None)
    if callable(getter):
        try:
            runtime = getter()
            if isinstance(runtime, dict):
                return runtime
        except Exception:
            logger.debug(
                "Could not read parent runtime for Pi question answer", exc_info=True
            )
    runtime = {
        key: getattr(parent_agent, key, "") or ""
        for key in ("model", "provider", "base_url", "api_key", "api_mode", "auth_mode")
    }
    return runtime if any(runtime.values()) else None



def _pi_model_for_parent(parent_agent: Any) -> str:
    """Best-effort pi ``--model`` argument derived from the parent runtime.

    Returns "" when nothing can be derived (pi keeps its own default).  The
    mapping is advisory: an unknown provider simply yields no argument.
    """
    runtime = _parent_main_runtime(parent_agent) or {}
    model = str(runtime.get("model") or "").strip()
    provider = str(runtime.get("provider") or "").strip()
    if not model:
        # Last-resort: the deployment-wide auxiliary model (same provider the
        # delegate-question answerer uses), so a bare parent object still gets
        # a working model instead of pi's unauthorized default.
        model = os.getenv("HERMES_ASSIST_MODEL", "").strip()
        if not model:
            return ""
        if not provider:
            provider = os.getenv("HERMES_ASSIST_PROVIDER", "").strip()
    # pi provider ids strip the "custom:" prefix Hermes uses for custom
    # providers; the models.json provider key is the bare name.
    provider_id = provider.split(":", 1)[-1] if provider else ""
    if provider_id and not _pi_provider_exists(provider_id):
        # "custom" (and similar Hermes-internal prefixes) are not pi provider
        # ids; passing e.g. "custom/glm-4.7" makes pi's binary wrapper reject
        # the model. Resolve against pi's own provider registry instead.
        provider_id = ""
    if not provider_id:
        # The runtime said nothing about the provider.  Resolve the model id
        # against pi's own provider registry so a bare model name (which pi
        # cannot route) still lands on the one provider that serves it.
        provider_id = _pi_provider_serving_model(model)
    if provider_id and provider_id.lower() not in {"anthropic", "openai"}:
        return f"{provider_id}/{model}"
    return model


def _pi_provider_exists(provider_id: str) -> bool:
    """True when ``provider_id`` is a known pi provider in models(.store).json."""
    import json as _json

    for candidates in (
        Path.home() / ".pi" / "agent" / "models.json",
        Path.home() / ".pi" / "agent" / "models-store.json",
    ):
        try:
            data = _json.loads(candidates.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        providers = data.get("providers") if isinstance(data, dict) else None
        if isinstance(providers, dict) and provider_id in providers:
            return True
    return provider_id in ("anthropic", "openai")


def _pi_provider_serving_model(model: str) -> str:
    """Find the pi provider whose catalog contains ``model`` ("" if none/many)."""
    import json as _json

    for candidates in (
        Path.home() / ".pi" / "agent" / "models.json",
        Path.home() / ".pi" / "agent" / "models-store.json",
    ):
        try:
            data = _json.loads(candidates.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        providers = data.get("providers") if isinstance(data, dict) else None
        if not isinstance(providers, dict):
            continue
        serving = [
            pid
            for pid, cfg in providers.items()
            if any(
                str(m.get("id")) == model
                for m in (cfg.get("models") or [])
                if isinstance(m, dict)
            )
        ]
        if len(serving) == 1:
            return str(serving[0])
        if serving:  # ambiguous: prefer a non-first-party provider
            external = [pid for pid in serving if pid not in ("anthropic", "openai")]
            return str(external[0]) if len(external) == 1 else ""
    return ""


def _auto_answer_pi_question(
    parent_agent: Any,
    method: str,
    question: str,
    options: list[str],
) -> str | None:
    return _auto_answer_delegate_question(parent_agent, "pi", method, question, options)


def _auto_answer_delegate_question(
    parent_agent: Any,
    backend_name: str,
    method: str,
    question: str,
    options: list[str],
) -> str | None:
    """Have supervising Hermes answer a delegate question without involving the user."""
    from agent.oneshot import run_oneshot

    context = _parent_context_excerpt(parent_agent)
    option_block = "\n".join(f"- {item}" for item in options[:50]) or "(none)"
    if method == "confirm":
        format_rule = "Answer exactly yes or no."
    elif method == "select" and options:
        format_rule = (
            "Answer with exactly one of the listed options, with no explanation."
        )
    else:
        format_rule = "Answer directly and concisely. Return only the answer the delegate should receive."

    instructions = (
        f"You are Hermes supervising a persistent {backend_name} coding delegate. The delegate has asked "
        "a question during delegated work. Answer it yourself from the available "
        "conversation/project context. Never ask the user, never request clarification, "
        "and never defer the decision back to the user. If context is incomplete, make "
        "the safest reasonable reversible choice that best advances the user's stated "
        "goal. Do not mention that you are an auxiliary model. " + format_rule
    )
    user_input = (
        f"Delegate question type: {method}\n"
        f"Delegate question: {question}\n"
        f"Options:\n{option_block}\n\n"
        "Recent supervising Hermes context:\n"
        f"{context or '(no additional context available)'}"
    )
    answer = run_oneshot(
        instructions=instructions,
        user_input=user_input,
        task="delegate_session_question",
        max_tokens=256,
        temperature=0.0,
        timeout=60.0,
        main_runtime=_parent_main_runtime(parent_agent),
    ).strip()
    if not answer:
        return None
    if method == "select" and options:
        low = answer.casefold().strip(" \"'`.,!\t\n")
        for option in options:
            if option.casefold().strip(" \"'`.,!\t\n") == low:
                return option
        if low.isdigit():
            index = int(low)
            if 1 <= index <= len(options):
                return options[index - 1]
        # A select response must be one of the offered values. Returning None
        # deliberately activates the conservative supervised fallback instead
        # of sending an invalid free-form selection over the native protocol.
        return None
    if method == "confirm":
        low = answer.casefold().strip(" .!\t\n")
        if low.startswith(("yes", "true", "approve", "confirm", "proceed")):
            return "yes"
        if low.startswith(("no", "false", "deny", "reject", "stop")):
            return "no"
        return None
    return _bounded(answer, 2000)


def _pending_payload(record: Dict[str, Any]) -> dict[str, Any] | None:
    client = record["client"]
    backend = record.get("backend") or "pi"
    if backend == "opencode":
        payload: Any = None
        getter = getattr(client, "pending_question_payload", None)
        if callable(getter):
            payload = getter()
        if not isinstance(payload, dict):
            return None
        return {
            "method": payload.get("method"),
            "question": _bounded(payload.get("question"), 2000),
            "options": list(payload.get("options") or [])[:50],
            "created_at": payload.get("created_at"),
        }
    question = pending_question_for_owner(client)
    if question is None:
        return None
    return {
        "method": question.method,
        "question": _bounded(question.title, 2000),
        "options": list(question.options)[:50],
        "created_at": question.created_at,
    }


def _summary(record: Dict[str, Any], *, include_result: bool = True) -> dict[str, Any]:
    out = {
        "session_id": record["session_id"],
        "backend": record.get("backend") or "pi",
        "native_session_id": record.get("native_session_id") or record["session_id"],
        "pi_session_id": record.get("native_session_id")
        or record["session_id"],  # kept for model-callers
        "status": record.get("status", "unknown"),
        "cwd": record.get("cwd"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "pending_question": _pending_payload(record),
        "error": record.get("error") or None,
    }
    if include_result and record.get("last_result"):
        result = record["last_result"]
        out["last_result"] = {
            "text": _bounded_edges(result.get("text")),
            "duration_s": result.get("duration_s"),
        }
    return out


def _record_authorized(
    record: Dict[str, Any], parent_agent: Any, caller_scope: str | None = None
) -> bool:
    """Authorize a live record: conversation owner or profile/workspace scope.

    The conversation that started the session keeps control even if its
    resolved workspace later moved (e.g. the caller cd'd elsewhere); otherwise a
    matching stable scope lets a replacement supervisor in the same workspace
    and profile recover the session.
    """
    if record.get("owner") == _owner_key(parent_agent):
        return True
    if caller_scope is None:
        caller_scope = _scope_for_workspace()
    saved_scope = str(record.get("owner_scope") or "").strip()
    # Process-local legacy records may lack owner_scope before lazy migration;
    # those fall back to the owner match above only.
    return bool(caller_scope and saved_scope and saved_scope == caller_scope)


# Returned with offline-durable payloads so the model knows how to recover.
_OFFLINE_RESUME_NOTE = (
    "Delegate session is offline: durable metadata exists but the native client "
    "is not loaded in this process. Use action='resume' with this session_id to "
    "reopen the native session and recover full control."
)


def _durable_summary(
    meta: dict[str, Any], *, note: Optional[str] = None
) -> dict[str, Any]:
    """Offline summary rebuilt from durable metadata (no client loaded)."""
    sid = str(meta.get("session_id") or "")
    native = str(meta.get("native_session_id") or meta.get("pi_session_id") or sid)
    out: dict[str, Any] = {
        "session_id": sid,
        "backend": meta.get("backend") or "pi",
        "native_session_id": native,
        "pi_session_id": native,  # kept for model-callers
        "status": "offline",
        "cwd": meta.get("cwd"),
        "created_at": meta.get("created_at"),
        "updated_at": meta.get("updated_at"),
        "pending_question": None,
        "error": None,
    }
    if note:
        out["note"] = note
    return out


def _authorized_durable_meta(
    session_id: str, parent_agent: Any, caller_scope: str | None = None
) -> Optional[dict[str, Any]]:
    """Load durable metadata for session_id when this supervisor may see it."""
    data = _load_metadata(session_id)
    if data is None or not _metadata_authorized(data, parent_agent, caller_scope):
        return None
    return data


def _dead_client_error(record: Dict[str, Any], backend: str) -> Optional[str]:
    """Mark a live record whose native process died; return an actionable error.

    Without this, a supervisor reading a stale in-memory record gets either a
    silent "running" state or a raw RPC failure instead of a resume hint.
    """
    client = record.get("client")
    proc = getattr(client, "_proc", None)
    dead = proc is not None and proc.poll() is not None
    is_dead = getattr(client, "is_dead", None)
    if callable(is_dead) and is_dead():
        dead = True
    if not dead or record.get("status") in {"closed", "error"}:
        return None
    exit_code = getattr(proc, "returncode", None) if proc is not None else None
    with _SESSION_LOCK:
        record["status"] = "error"
        record["error"] = f"{backend} delegate process exited with code {exit_code}"
        record["updated_at"] = time.time()
    _persist_metadata(record)
    return (
        f"{backend} delegate process exited with code {exit_code}. "
        "Use action='resume' to reopen the native session."
    )


def _lookup(session_id: str, parent_agent: Any) -> Dict[str, Any] | None:
    with _SESSION_LOCK:
        record = _SESSIONS.get(session_id)
        if record is None or not _record_authorized(record, parent_agent):
            return None
        return record


def _notify_state_locked(record: Dict[str, Any]) -> None:
    record["state_generation"] = int(record.get("state_generation") or 0) + 1
    _SESSION_CONDITION.notify_all()


def _transition_status_locked(record: Dict[str, Any], new_status: str) -> bool:
    if record.get("status") == new_status:
        return False
    record["status"] = new_status
    record["updated_at"] = time.time()
    _notify_state_locked(record)
    return True


def _delegate_dead(client: Any) -> bool:
    proc = getattr(client, "_proc", None)
    if proc is not None and proc.poll() is not None:
        return True
    is_dead = getattr(client, "is_dead", None)
    return callable(is_dead) and bool(is_dead())


def _mark_dead_delegate(record: Dict[str, Any]) -> bool:
    client = record.get("client")
    if client is None or record.get("status") in {"closed", "error"}:
        return False
    if not _delegate_dead(client):
        return False
    proc = getattr(client, "_proc", None)
    exit_code = getattr(proc, "returncode", None) if proc is not None else None
    backend = record.get("backend") or "pi"
    with _SESSION_CONDITION:
        record["error"] = f"{backend} delegate process exited with code {exit_code}"
        changed = _transition_status_locked(record, "error")
    if changed:
        _persist_metadata(record)
    return changed


def _run_turn(record: Dict[str, Any], message: str, timeout: float) -> None:
    client = record["client"]
    with _SESSION_CONDITION:
        if record.get("status") == "closed":
            return
        record["error"] = ""
        _transition_status_locked(record, "running")
    try:
        result = client.run_session_prompt(message, timeout_seconds=timeout)
        state = result.get("state") if isinstance(result, dict) else {}
        with _SESSION_CONDITION:
            record["last_result"] = result
            record["native_session_id"] = (
                (state.get("sessionId") if isinstance(state, dict) else None)
                or record.get("native_session_id")
                or record["session_id"]
            )
            if record.get("status") != "closed":
                _transition_status_locked(record, "idle")
            else:
                record["updated_at"] = time.time()
        _persist_metadata(record)
    except Exception as exc:  # noqa: BLE001 - surfaced as bounded session state
        logger.exception("Delegate session %s turn failed", record.get("session_id"))
        with _SESSION_CONDITION:
            record["error"] = _bounded(exc, 2000)
            if record.get("status") != "closed":
                _transition_status_locked(record, "error")
            else:
                record["updated_at"] = time.time()
        _persist_metadata(record)


def _dispatch_turn(record: Dict[str, Any], message: str, timeout: float) -> None:
    thread = threading.Thread(
        target=_run_turn,
        args=(record, message, timeout),
        name=f"delegate-{record['session_id'][:8]}",
        # Keep the interpreter alive until the active delegated turn reaches a
        # terminal state.  A daemon thread can be torn down as soon as the
        # supervising Hermes/cron process returns, which abandons the live Pi
        # or OpenCode turn even though durable metadata remains recoverable.
        daemon=False,
    )
    with _SESSION_CONDITION:
        record["thread"] = thread
        _transition_status_locked(record, "running")
    thread.start()


def _initial_prompt(goal: str, context: str | None) -> str:
    policy = (
        "[Delegation policy] You are a coding delegate operating in a persistent "
        "session owned by Hermes. Work directly in the current working tree. Do "
        "not commit or push unless Hermes explicitly asks you to. Ask questions "
        "when a decision is genuinely required; the supervising Hermes agent will "
        "answer them automatically through the same delegate session."
    )
    parts = [policy]
    if context and context.strip():
        parts.append("Context from Hermes:\n" + context.strip())
    if goal and goal.strip():
        parts.append("Task:\n" + goal.strip())
    return "\n\n".join(parts)


def delegate_session(
    *,
    action: str = "start",
    session_id: Optional[str] = None,
    goal: Optional[str] = None,
    context: Optional[str] = None,
    message: Optional[str] = None,
    timeout: Optional[int] = None,
    wait_seconds: Optional[float] = None,
    backend: Optional[str] = None,
    parent_agent: Any = None,
) -> str:
    """Create/control a persistent native delegation session (Pi or OpenCode)."""
    if parent_agent is None:
        return tool_error("delegate_session requires a parent agent context.")

    normalized = (action or "start").strip().lower()
    if normalized not in {
        "start",
        "resume",
        "send",
        "steer",
        "status",
        "wait",
        "messages",
        "list",
        "stop",
    }:
        return tool_error(
            "Unknown action. Use start, resume, send, steer, status, wait, messages, list, or stop."
        )
    if backend is not None and str(backend).strip().lower() not in _KNOWN_BACKENDS:
        return tool_error(f"Unknown backend {backend!r}. Use 'pi' or 'opencode'.")
    effective_timeout = float(max(10, min(int(timeout or 900), 3600)))
    try:
        effective_wait = max(
            0.0, min(float(120 if wait_seconds is None else wait_seconds), 3600.0)
        )
    except (TypeError, ValueError):
        return tool_error("wait_seconds must be a number between 0 and 3600.")
    owner = _owner_key(parent_agent)

    if normalized == "list":
        caller_scope = _scope_for_workspace()
        with _SESSION_LOCK:
            live_records = [
                record
                for record in _SESSIONS.values()
                if _record_authorized(record, parent_agent, caller_scope)
            ]
            live_ids = {str(record.get("session_id") or "") for record in live_records}
            rows = [_summary(record, include_result=False) for record in live_records]
        for meta in _durable_rows_for_caller(parent_agent, caller_scope):
            sid = str(meta.get("session_id") or "")
            if not sid or sid in live_ids:
                continue
            rows.append(_durable_summary(meta))
        rows.sort(key=lambda row: float(row.get("updated_at") or 0), reverse=True)
        return json.dumps({"success": True, "sessions": rows}, ensure_ascii=False)

    if normalized in {"start", "resume"}:
        requested_id = (session_id or "").strip()
        if normalized == "resume" and not requested_id:
            return tool_error("action=resume requires session_id.")
        handle = requested_id or str(uuid.uuid4())
        saved = _load_metadata(handle) if requested_id else None
        if saved is not None and not _metadata_authorized(saved, parent_agent):
            return tool_error(
                "That delegate session belongs to another profile or workspace."
            )

        backend_name = _resolve_backend(backend, (saved or {}).get("backend"))
        if backend_name is None:
            return tool_error(f"Unknown backend {backend!r}. Use 'pi' or 'opencode'.")

        existing = None
        with _SESSION_LOCK:
            existing = _SESSIONS.get(handle)
            if existing is not None:
                if not _record_authorized(existing, parent_agent):
                    return tool_error(
                        "That delegate session belongs to another profile or workspace."
                    )
                if (
                    backend is not None
                    and (existing.get("backend") or "pi") != backend_name
                ):
                    return tool_error(
                        f"Session {handle} is a {existing.get('backend') or 'pi'} delegate session; "
                        f"it cannot be reopened as a {backend_name} session."
                    )
                client_obj = existing.get("client")
                proc = getattr(client_obj, "_proc", None)
                process_dead = proc is not None and proc.poll() is not None
                is_dead = getattr(client_obj, "is_dead", None)
                if callable(is_dead) and is_dead():
                    process_dead = True
                # R69 (2026-09-17): a dead client must reopen for ANY
                # lifecycle action, not only resume. The spool server drives
                # phases with action=start + session_id + goal (follow-up
                # turns); with reopen gated on resume only, a start-with-goal
                # on a dead pi client dispatched the turn into the CLOSED
                # client ("pi rpc client is closed") and every retry hit the
                # same dead handle — the phase could never recover without a
                # brand-new session id. Dead is dead: reopen, then dispatch.
                client_closed = bool(getattr(client_obj, "is_closed", False))
                reopen = existing.get("status") in {"closed", "error"} or client_closed or process_dead
                if not reopen:
                    if goal and goal.strip():
                        # Re-start on a live session is a FOLLOW-UP, not a
                        # no-op: silently dropping the goal made every later
                        # phase of a multi-turn delegation appear to succeed
                        # while no work ran (conductor v5/v6 cycles).
                        _dispatch_turn(
                            existing, _initial_prompt(goal, context), effective_timeout
                        )
                        return json.dumps(
                            {"success": True, "reused": True, "turn_dispatched": True, **_summary(existing)},
                            ensure_ascii=False,
                        )
                    return json.dumps(
                        {"success": True, "reused": True, **_summary(existing)},
                        ensure_ascii=False,
                    )
                _SESSIONS.pop(handle, None)
        if existing is not None:
            try:
                existing["client"].close()
            except Exception:
                logger.debug(
                    "Could not close stale delegate client before resume", exc_info=True
                )

        saved_cwd = str((saved or {}).get("cwd") or "").strip()
        cwd_path = Path(saved_cwd).expanduser() if saved_cwd else resolve_agent_cwd()
        if not cwd_path.is_dir():
            return tool_error(
                f"Cannot resume delegate session because its workspace no longer exists: {_bounded(cwd_path, 1000)}"
            )
        cwd = str(cwd_path.resolve())
        native_hint = str(
            (saved or {}).get("native_session_id")
            or (saved or {}).get("pi_session_id")
            or ""
        ).strip()
        client_class = _backend_client_class(backend_name)
        answer_backend = backend_name
        if answer_backend == "pi":

            def _answer(method: str, title: str, options: list[str]) -> Optional[str]:
                # Late-bound so tests can patch ds._auto_answer_pi_question.
                return _auto_answer_pi_question(parent_agent, method, title, options)

        else:

            def _answer(method: str, title: str, options: list[str]) -> Optional[str]:
                return _auto_answer_delegate_question(
                    parent_agent, answer_backend, method, title, options
                )

        client_kwargs: dict[str, Any] = {}
        if backend_name == "pi":
            # Without an explicit model pi falls back to its built-in
            # Anthropic model and dies with 401 on keyless installs.
            # Resolve, in order: explicit env override, then the parent
            # runtime's provider/model pair mapped onto a pi provider id.
            explicit_model = os.getenv("HERMES_PI_MODEL", "").strip()
            model_arg = explicit_model or _pi_model_for_parent(parent_agent)
            if model_arg:
                client_kwargs["args"] = ["--model", model_arg]
        client = client_class(
            persistent_session=True,
            session_id=native_hint or handle,
            session_name=f"Hermes {handle[:8]}",
            acp_cwd=cwd,
            question_answerer=_answer,
            **client_kwargs,
        )
        if native_hint and hasattr(client, "native_session_id"):
            client.native_session_id = native_hint
        try:
            state = client.start(timeout=min(30.0, effective_timeout))
        except Exception as exc:  # noqa: BLE001
            try:
                client.close()
            except Exception:
                logger.debug(
                    "Could not close failed %s delegate client",
                    backend_name,
                    exc_info=True,
                )
            return tool_error(
                f"Could not start {backend_name} delegate session: {_bounded(exc, 1000)}"
            )
        native_id = str(state.get("sessionId") or handle)
        now = time.time()
        record: Dict[str, Any] = {
            "session_id": handle,
            "backend": backend_name,
            "native_session_id": native_id,
            "owner": owner,
            "owner_scope": _scope_for_workspace(cwd),
            "cwd": cwd,
            "client": client,
            "status": "idle",
            "created_at": (saved or {}).get("created_at") or now,
            "updated_at": now,
            "last_result": None,
            "error": "",
            "thread": None,
        }
        with _SESSION_LOCK:
            _SESSIONS[handle] = record
        _persist_metadata(record)
        if goal and goal.strip():
            _dispatch_turn(record, _initial_prompt(goal, context), effective_timeout)
        return json.dumps(
            {"success": True, "created": True, **_summary(record)}, ensure_ascii=False
        )

    if not session_id or not session_id.strip():
        return tool_error(f"action='{normalized}' requires session_id.")
    record = _lookup(session_id.strip(), parent_agent)
    if record is None:
        # Durable metadata may outlive the in-memory registry (gateway restart,
        # supervisor replacement). Reads fall back to an offline summary;
        # control actions fail closed until the session is resumed.
        durable = _authorized_durable_meta(session_id.strip(), parent_agent)
        if durable is None:
            return tool_error(
                "Delegate session not found for this conversation. Use action='resume' to reopen a native session."
            )
        if normalized == "status":
            return json.dumps(
                {
                    "success": True,
                    **_durable_summary(durable, note=_OFFLINE_RESUME_NOTE),
                },
                ensure_ascii=False,
            )
        if normalized == "messages":
            # Durable metadata deliberately stores no prompt text; the native
            # backend holds the history once the session is resumed.
            return json.dumps(
                {
                    "success": True,
                    **_durable_summary(durable, note=_OFFLINE_RESUME_NOTE),
                    "messages_json": "[]",
                },
                ensure_ascii=False,
            )
        return tool_error(
            f"Delegate session {session_id.strip()} is offline (durable). "
            f"Use action='resume' to reopen it before action='{normalized}'."
        )
    session_backend = record.get("backend") or "pi"
    if backend is not None and str(backend).strip().lower() != session_backend:
        return tool_error(
            f"Session {record['session_id']} is a {session_backend} delegate session; pass backend='{session_backend}' or omit it."
        )
    client = record["client"]

    if normalized == "status":
        _mark_dead_delegate(record)
        return json.dumps({"success": True, **_summary(record)}, ensure_ascii=False)

    if normalized == "wait":
        started = time.monotonic()
        deadline = started + effective_wait
        entry_status = record.get("status")
        while True:
            _mark_dead_delegate(record)
            with _SESSION_CONDITION:
                if record.get("status") != "running":
                    return json.dumps(
                        {
                            "success": True,
                            "timed_out": False,
                            "state_changed": record.get("status") != entry_status,
                            "waited_s": round(time.monotonic() - started, 3),
                            **_summary(record),
                        },
                        ensure_ascii=False,
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return json.dumps(
                        {
                            "success": True,
                            "timed_out": True,
                            "state_changed": False,
                            "waited_s": round(time.monotonic() - started, 3),
                            **_summary(record),
                        },
                        ensure_ascii=False,
                    )
                _SESSION_CONDITION.wait(timeout=remaining)

    if normalized == "messages":
        dead_error = _dead_client_error(record, session_backend)
        if dead_error:
            return tool_error(dead_error)
        try:
            messages = client.get_messages(timeout=min(30.0, effective_timeout))
        except Exception as exc:  # noqa: BLE001
            return tool_error(
                f"Could not read {session_backend} session messages: {_bounded(exc, 1000)}"
            )
        # Keep the tool result bounded while preserving the newest conversational
        # state AND valid JSON. Never slice the serialized array mid-document:
        # consumers such as Conductor need to parse the result to recover a
        # structured phase_result from the newest assistant message.
        safe = list(messages[-40:])
        encoded = json.dumps(safe, ensure_ascii=False, default=str)
        while len(encoded) > 40_000 and len(safe) > 1:
            safe.pop(0)
            encoded = json.dumps(safe, ensure_ascii=False, default=str)
        if len(encoded) > 40_000 and safe:
            newest = safe[-1]
            row = newest.get("message") if isinstance(newest, dict) and isinstance(newest.get("message"), dict) else newest
            role = str(row.get("role") or "assistant") if isinstance(row, dict) else "assistant"
            content = row.get("content") if isinstance(row, dict) else row
            text = _bounded_edges(_message_text(content), 30_000)
            safe = [{"role": role, "content": [{"text": text}]}]
            encoded = json.dumps(safe, ensure_ascii=False, default=str)
        return json.dumps(
            {
                "success": True,
                "session_id": record["session_id"],
                "messages_json": encoded,
            },
            ensure_ascii=False,
        )

    if normalized == "send":
        text = (message or goal or "").strip()
        if not text:
            return tool_error("action='send' requires message.")
        dead_error = _dead_client_error(record, session_backend)
        if dead_error:
            return tool_error(dead_error)
        with _SESSION_LOCK:
            if record.get("status") == "running":
                return tool_error(
                    "Delegate session is currently running. Use action='steer' to redirect it, or wait for idle."
                )
            if record.get("status") == "closed":
                return tool_error(
                    "Delegate session is closed. Use action='resume' to reopen it."
                )
        _dispatch_turn(record, text, effective_timeout)
        return json.dumps(
            {
                "success": True,
                "accepted": True,
                **_summary(record, include_result=False),
            },
            ensure_ascii=False,
        )

    if normalized == "steer":
        text = (message or "").strip()
        if not text:
            return tool_error("action='steer' requires message.")
        dead_error = _dead_client_error(record, session_backend)
        if dead_error:
            return tool_error(dead_error)
        with _SESSION_LOCK:
            status = record.get("status")
            if status == "closed":
                return tool_error(
                    "Delegate session is closed. Use action='resume' to reopen it."
                )
            if status != "running":
                # Auto-degrade: the turn already ended (or the backend has no
                # live steer), so route the message through the send path so
                # the course-correction is not lost to a race window.
                _dispatch_turn(record, text, effective_timeout)
                return json.dumps(
                    {
                        "success": True,
                        "degraded_to_send": True,
                        "note": f"{session_backend} turn had already ended; message was delivered as a new follow-up turn.",
                        **_summary(record, include_result=False),
                    },
                    ensure_ascii=False,
                )
        try:
            response = client.steer(text, timeout=min(30.0, effective_timeout))
        except Exception as exc:  # noqa: BLE001
            return tool_error(
                f"Could not steer {session_backend} session: {_bounded(exc, 1000)}"
            )
        return json.dumps(
            {
                "success": True,
                "response": response,
                **_summary(record, include_result=False),
            },
            ensure_ascii=False,
            default=str,
        )

    if normalized == "stop":
        try:
            if record.get("status") == "running":
                client.abort(timeout=min(10.0, effective_timeout))
        except Exception:
            logger.debug("Delegate abort failed before close", exc_info=True)
        client.close()
        with _SESSION_LOCK:
            record["status"] = "closed"
            record["updated_at"] = time.time()
        _persist_metadata(record)
        return json.dumps(
            {"success": True, "closed": True, **_summary(record)}, ensure_ascii=False
        )

    return tool_error("Unhandled delegate_session action.")


DELEGATE_SESSION_SCHEMA = {
    "name": "delegate_session",
    "description": (
        "Delegate coding work to Pi or OpenCode through a persistent native "
        "session. Use this instead of delegate_task when the worker should be "
        "an external coding agent. The native conversation survives across "
        "turns: start a session, send follow-ups, steer a running turn, "
        "inspect status or messages, and stop/resume the same native session. "
        "Backend questions are answered automatically by the supervising "
        "Hermes agent rather than forwarded to the user. Pi is the default "
        "backend; pass backend='opencode' to delegate to OpenCode via its "
        "server API. delegate_task remains the Hermes child-agent primitive."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "start",
                    "resume",
                    "send",
                    "steer",
                    "status",
                    "wait",
                    "messages",
                    "list",
                    "stop",
                ],
                "description": "Session lifecycle/control action. Omit for start.",
            },
            "session_id": {
                "type": "string",
                "description": "Persistent delegation/session id returned by start. Required for all actions except start/list.",
            },
            "goal": {
                "type": "string",
                "description": "Initial coding objective for action='start'. The turn runs asynchronously in the persistent native session.",
            },
            "context": {
                "type": "string",
                "description": "Initial background/context passed with goal when starting the session.",
            },
            "message": {
                "type": "string",
                "description": "Follow-up for send, or live course correction for steer. Delegate questions are answered automatically by Hermes.",
            },
            "timeout": {
                "type": "integer",
                "minimum": 10,
                "maximum": 3600,
                "description": "Maximum seconds allowed for each delegate turn (default 900).",
            },
            "wait_seconds": {
                "type": "number",
                "minimum": 0,
                "maximum": 3600,
                "description": "Maximum seconds action='wait' blocks for a meaningful state change. Expiry is nonfatal and does not stop the delegated worker (default 120).",
            },
            "backend": {
                "type": "string",
                "enum": ["pi", "opencode"],
                "description": (
                    "External agent backend for the session: 'pi' (default) or "
                    "'opencode'. Applies at start/resume; control actions use the "
                    "stored backend."
                ),
            },
        },
        "required": [],
    },
}


def _resolve_parent_agent(parent_agent: Any) -> Any:
    """Fall back to the turn-bound active parent when not explicitly forwarded.

    The generic registry dispatch path (handle_function_call -> registry.dispatch)
    cannot forward the agent object, so a delegate_session call routed through it
    would otherwise always fail with "requires a parent agent context". The
    conversation loop binds the parent for the duration of each turn, which is
    exactly the context a live tool call executes in.
    """
    if parent_agent is not None:
        return parent_agent
    try:
        from agent.subagent_lifecycle import get_active_subagent_parent

        return get_active_subagent_parent()
    except Exception:
        return None


registry.register(
    name="delegate_session",
    toolset="delegation_session",
    schema=DELEGATE_SESSION_SCHEMA,
    handler=lambda args, **kw: delegate_session(
        action=args.get("action") or "start",
        session_id=args.get("session_id"),
        goal=args.get("goal"),
        context=args.get("context"),
        message=args.get("message"),
        timeout=args.get("timeout"),
        wait_seconds=args.get("wait_seconds"),
        backend=args.get("backend"),
        parent_agent=_resolve_parent_agent(kw.get("parent_agent")),
    ),
    check_fn=check_delegate_session_requirements,
    emoji="🔁",
)
