import os
import uuid
from functools import lru_cache
from typing import Any

from fastapi import HTTPException

from hermes_cli.config import load_config

try:
    from gateway.run import _resolve_model, _resolve_runtime_agent_kwargs
except ImportError:
    def _resolve_model() -> str:
        config = load_config()
        return config.get("model", os.getenv("HERMES_MODEL", "claude-sonnet-4-5"))

    def _resolve_runtime_agent_kwargs() -> dict:
        config = load_config()
        return {"provider": config.get("provider", os.getenv("HERMES_PROVIDER", "anthropic"))}
from hermes_state import SessionDB
from run_agent import AIAgent
from tools.memory_tool import MemoryStore


WEB_SOURCE = "web"


@lru_cache(maxsize=1)
def get_session_db() -> SessionDB:
    return SessionDB()


@lru_cache(maxsize=1)
def get_memory_store() -> MemoryStore:
    store = MemoryStore()
    store.load_from_disk()
    return store


def reload_memory_store() -> MemoryStore:
    store = get_memory_store()
    store.load_from_disk()
    return store


def get_config() -> dict[str, Any]:
    return load_config()


def get_runtime_model() -> str:
    return _resolve_model()


def get_runtime_agent_kwargs() -> dict[str, Any]:
    return _resolve_runtime_agent_kwargs()


def create_agent(
    *,
    session_id: str,
    session_db: SessionDB,
    model: str | None = None,
    ephemeral_system_prompt: str | None = None,
    enabled_toolsets: list[str] | None = None,
    disabled_toolsets: list[str] | None = None,
    skip_context_files: bool = False,
    skip_memory: bool = False,
    stream_callback=None,
    tool_progress_callback=None,
    thinking_callback=None,
    reasoning_callback=None,
    step_callback=None,
) -> AIAgent:
    runtime_kwargs = get_runtime_agent_kwargs()
    effective_model = model or get_runtime_model()
    max_iterations = int(os.getenv("HERMES_MAX_ITERATIONS", "90"))

    return AIAgent(
        model=effective_model,
        **runtime_kwargs,
        max_iterations=max_iterations,
        quiet_mode=True,
        verbose_logging=False,
        ephemeral_system_prompt=ephemeral_system_prompt,
        session_id=session_id,
        platform="webapi",
        session_db=session_db,
        enabled_toolsets=enabled_toolsets,
        disabled_toolsets=disabled_toolsets,
        skip_context_files=skip_context_files,
        skip_memory=skip_memory,
        tool_progress_callback=tool_progress_callback,
        thinking_callback=thinking_callback,
        reasoning_callback=reasoning_callback,
        step_callback=step_callback,
    )


def get_session_or_404(session_id: str, session_db: SessionDB | None = None) -> dict[str, Any]:
    db = session_db or get_session_db()
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session


def ensure_session_title(session_db: SessionDB, title: str | None) -> str | None:
    cleaned = session_db.sanitize_title(title)
    if cleaned:
        return cleaned
    return session_db.get_next_title_in_lineage("New Chat")


def new_session_id() -> str:
    return f"sess_{uuid.uuid4().hex}"
