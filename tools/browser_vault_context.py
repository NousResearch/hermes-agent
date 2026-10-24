"""Backend-specific trusted fill lifetime; credential exceptions never leave it."""
import json
from contextlib import nullcontext
from functools import wraps


def with_backend_fill_context(fill):
    @wraps(fill)
    def wrapped(handle, task_id=None):
        from agent.vault_backends import backend_for_handle
        try:
            backend = backend_for_handle(handle)
            context = backend.fill_context() if backend is not None and hasattr(backend, "fill_context") else nullcontext()
            with context:
                return fill(handle, task_id, _backend=backend)
        except Exception:  # health: allow BLE001 -- backend failures may contain credentials; never log exception text
            return json.dumps({"success": False, "error_type": "backend_unavailable",
                               "error": "Credential backend unavailable or browser profile busy."})
    return wrapped
