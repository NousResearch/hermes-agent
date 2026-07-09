"""Hermes Native local bridge gating/runtime helper.

Adopted patterns from llm-bench-rig:
- Progress atomic write via `_tmp + rename` state machine.
- Hard gating before execution / inference.
- Local-only by default; no outbound inference fallback.
"""
import json
import os
import tempfile
import time
from pathlib import Path


class Progress:
    def __init__(self, path: str, model: str = "local"):
        self.path = Path(path)
        self.data = {
            "model": model,
            "step": "init",
            "pct": 0,
            "started_at": time.time(),
            "updated_at": time.time(),
            "partial": {},
        }
        self._write()

    def update(self, step: str, pct: int, partial: dict | None = None) -> None:
        self.data["step"] = step
        self.data["pct"] = int(pct)
        self.data["updated_at"] = time.time()
        if partial:
            self.data["partial"].update(partial)
        self._write()

    def done(self) -> None:
        self.data["step"] = "done"
        self.data["pct"] = 100
        self.data["updated_at"] = time.time()
        self._write()

    def fail(self, error: str) -> None:
        self.data["step"] = "error"
        self.data["error"] = str(error)
        self.data["updated_at"] = time.time()
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        tmp.replace(self.path)


def default_gate_check() -> tuple[bool, str]:
    """Local hard gate: return (pass, detail)."""
    gpu_available = os.environ.get("HERMES_GPU_AVAILABLE", "0") == "1"
    detail = "gate=local" if gpu_available else "gate=skip_local_gpu"
    return gpu_available, detail


def run_when_passed(progress_path: str, model: str = "local"):
    """Decorator-style gating wrapper around an action."""
    def decorator(action):
        def wrapper(*args, **kwargs):
            progress = Progress(progress_path, model=model)
            try:
                progress.update("gate_check", 5)
                passed, detail = default_gate_check()
                if not passed:
                    progress.fail(f"GATE_FAIL: {detail}")
                    return {"status": "skipped", "detail": detail}
                progress.update("running", 10)
                result = action(*args, **kwargs)
                progress.update("finalizing", 90)
                progress.done()
                return {"status": "done", "result": result, "detail": detail}
            except Exception as e:
                progress.fail(str(e))
                return {"status": "error", "error": str(e)}
        return wrapper
    return decorator
