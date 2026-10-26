"""Registry for background (async) delegate_task sessions."""
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DelegationSession:
    id: str                          # session_id returned to the model
    goal: str                        # first task goal (for display)
    n_tasks: int
    started_at: float = field(default_factory=time.monotonic)
    status: str = "running"          # running | completed | error | interrupted
    results: Optional[List[Dict]] = None
    error: Optional[str] = None


_registry: Dict[str, DelegationSession] = {}
_lock = threading.Lock()


def register(session: DelegationSession) -> None:
    with _lock:
        _registry[session.id] = session


def get(session_id: str) -> Optional[DelegationSession]:
    with _lock:
        return _registry.get(session_id)


def update(session_id: str, **kwargs) -> None:
    with _lock:
        s = _registry.get(session_id)
        if s:
            for k, v in kwargs.items():
                setattr(s, k, v)
