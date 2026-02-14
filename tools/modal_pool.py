"""Modal sandbox pooling backend for terminal_tool.

This module provides an OPTIONAL pooled Modal backend that is compatible with
Hermes-Agent's existing terminal_tool interface.

Goals:
- Keep the default Modal path unchanged.
- Allow switching to pooled behavior with minimal friction:
    TERMINAL_ENV=modal
    TERMINAL_MODAL_MODE=pool

Design:
- Pool stores warm `_ModalEnvironment` instances (each wraps a live ModalDeployment).
- Each task acquires one environment exclusively, uses a task-specific working dir,
  then releases it back to the pool.
- Release attempts to remove the task working directory to reduce cross-task leakage.

NOTE: This is intentionally conservative and self-contained. It does not change
any tool schemas or model-facing behavior.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional


@dataclass
class _PooledEnv:
    env: object
    created_at: float


class ModalEnvPool:
    """Thread-safe pool of warm Modal environments."""

    def __init__(self, max_size: int = 4):
        self.max_size = max_size
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._idle: list[_PooledEnv] = []
        self._total = 0

    def acquire(self, create_fn, *, wait_s: int = 300):
        deadline = time.time() + wait_s
        with self._cond:
            while True:
                if self._idle:
                    return self._idle.pop().env

                if self._total < self.max_size:
                    self._total += 1
                    break

                remaining = deadline - time.time()
                if remaining <= 0:
                    # As a last resort, allow temporary oversubscription.
                    self._total += 1
                    break

                self._cond.wait(timeout=min(5, remaining))

        # Create outside lock (slow)
        try:
            return create_fn()
        except Exception:
            # Roll back count if create fails
            with self._cond:
                self._total -= 1
                self._cond.notify()
            raise

    def release(self, env: object):
        with self._cond:
            self._idle.append(_PooledEnv(env=env, created_at=time.time()))
            self._cond.notify()


# Global pool (process-level)
_global_pool: Optional[ModalEnvPool] = None


def get_global_pool() -> ModalEnvPool:
    global _global_pool
    if _global_pool is None:
        max_size = int(os.getenv("TERMINAL_MODAL_POOL_MAX", os.getenv("TERMINAL_MODAL_POOL_SIZE", "4")))
        _global_pool = ModalEnvPool(max_size=max_size)
    return _global_pool


class ModalPooledTaskEnvironment:
    """Per-task environment wrapper that leases a pooled Modal env."""

    def __init__(self, *, inner, base_cwd: str, timeout: int, task_id: str):
        self._inner = inner
        self.timeout = timeout
        self.task_id = task_id or str(uuid.uuid4())
        self.base_cwd = base_cwd.rstrip("/") or "/root"
        self.cwd = f"{self.base_cwd}/hermes_tasks/{self.task_id}"

        # Ensure workdir exists and is empty-ish
        self._inner.execute(f"mkdir -p {self.cwd} && rm -rf {self.cwd}/*", cwd="/", timeout=60)

    @classmethod
    def acquire(cls, *, image: str, base_cwd: str, timeout: int, task_id: str, create_modal_env_fn):
        pool = get_global_pool()
        inner = pool.acquire(create_modal_env_fn)
        return cls(inner=inner, base_cwd=base_cwd, timeout=timeout, task_id=task_id)

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict:
        # Always execute in the task workdir unless an explicit cwd is given.
        workdir = cwd or self.cwd
        return self._inner.execute(command, cwd=workdir, timeout=timeout or self.timeout)

    def cleanup(self):
        # Best-effort cleanup of task directory, then return to pool
        try:
            self._inner.execute(f"rm -rf {self.cwd}", cwd="/", timeout=60)
        except Exception:
            pass
        get_global_pool().release(self._inner)

    def stop(self):
        self.cleanup()
