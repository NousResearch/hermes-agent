"""Real-workload measurement runner for #112639 P0.

Drives REAL computer-use tasks through the real Hermes tool path on the local
machine (Xvfb + cua-driver + a real GUI app), records per-phase spans using the
#112778 taxonomy names, and prints a critical-path report. This is the P0 exit
gate made concrete: real benchmark wall time explained by named spans.

Spans are recorded by wrapping the real backend class methods and pre-seeding the
real session call lock with a timing lock -- ``handle_computer_use`` itself is
untouched, so every measured millisecond is the production code path. The run
also records the remaining P0 checkboxes as span dimensions: model/backend cold
starts, dispatch queue time, and a local CPU/GPU/VRAM snapshot per task. All
measurement stays in-process; nothing leaves the machine.

Input note: real input must reach apps through XTest fake_input; pynput-style
XSendEvent is dropped by toolkits, so direct-input prototypes must drive the
cua-driver path (as this runner does) instead.

Usage:
    DISPLAY=:99 HERMES_YOLO_MODE=1 python -m tools.computer_use.real_workload_runner \\
        --tasks perceive,act,dialog --session-id p0bench

Refuses to run without a display, without cua-driver, or without the yolo bypass
(an unattended approval gate would fail closed instead of measuring anything).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

# Phase names follow the #112778 taxonomy so this runner's output merges cleanly
# with the span pipeline once that PR lands. "task"/"suite" are runner-only.
INPUT_METHODS = ("click", "double_click", "middle_click", "right_click", "drag",
                 "scroll", "type_text", "key", "set_value", "focus_app",
                 "wait", "list_apps", "list_windows")

_NVIDIA_SMI_TIMEOUT_S = 2.0
_GPU_PROBE_TTL_S = 60.0
_gpu_probe_at = 0.0
_gpu_probe_result: Dict[str, Any] = {}


def sample_hardware() -> Dict[str, Any]:
    """Best-effort LOCAL hardware snapshot. Never raises; {} when unreadable."""
    snap: Dict[str, Any] = {}
    try:
        load1, _, _ = os.getloadavg()  # unix only
        snap["cpu_load_1m"] = round(float(load1), 2)
    except (OSError, AttributeError):
        pass
    try:
        mem = {}
        with open("/proc/meminfo") as f:  # linux only
            for line in f:
                k, _, v = line.partition(":")
                if k in ("MemTotal", "MemAvailable"):
                    mem[k] = int(v.split()[0])  # kB
        if "MemTotal" in mem:
            snap["mem_total_mb"] = mem["MemTotal"] // 1024
            if "MemAvailable" in mem:
                snap["mem_used_mb"] = (mem["MemTotal"] - mem["MemAvailable"]) // 1024
    except (OSError, ValueError):
        pass
    snap.update(_gpu_snapshot())
    return snap


def _gpu_snapshot() -> Dict[str, Any]:
    """nvidia-smi VRAM readout, probed at most once a minute. {} when absent."""
    global _gpu_probe_at, _gpu_probe_result
    now = time.monotonic()
    if now - _gpu_probe_at < _GPU_PROBE_TTL_S:
        return dict(_gpu_probe_result)
    _gpu_probe_at, _gpu_probe_result = now, {}
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            proc = subprocess.run(
                [smi, "--query-gpu=name,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=_NVIDIA_SMI_TIMEOUT_S)
            parts = [p.strip() for p in (proc.stdout or "").split(",")]
            if len(parts) == 3:
                _gpu_probe_result = {"gpu_name": parts[0],
                                     "vram_used_mb": int(parts[1]),
                                     "vram_total_mb": int(parts[2])}
        except Exception:
            pass
    return dict(_gpu_probe_result)


# Container spans nest the leaf work; the report attributes only leaves.
_CONTAINER_PHASES = frozenset({"task", "suite", "total"})


@dataclass
class Span:
    name: str
    duration_ms: float
    dims: Dict[str, Any] = field(default_factory=dict)


class SpanRecorder:
    """Minimal in-process span recorder speaking the #112778 phase taxonomy."""

    def __init__(self) -> None:
        self.spans: List[Span] = []
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def span(self, name: str, **dims: Any) -> Iterator[None]:
        start = time.monotonic()
        try:
            yield
        finally:
            ms = (time.monotonic() - start) * 1000.0
            with self._lock:
                self.spans.append(Span(name, ms, dict(dims)))

    def totals(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for s in self.spans:
            out[s.name] = out.get(s.name, 0.0) + s.duration_ms
        return out

    def cold_starts(self) -> int:
        return sum(1 for s in self.spans if s.dims.get("cold_start") is True)

    def queue_time_ms(self) -> float:
        return sum(float(s.dims.get("queue_time_ms", 0.0)) for s in self.spans
                   if s.name == "dispatch_lock_wait")

    def report(self, wall_ms: float) -> Dict[str, Any]:
        totals = self.totals()
        leaf = {k: v for k, v in totals.items() if k not in _CONTAINER_PHASES}
        attributed = sum(leaf.values())
        critical = max(leaf.items(), key=lambda kv: kv[1])[0] if leaf else "none"
        return {"wall_ms": round(wall_ms, 1),
                "phases": {k: round(v, 1) for k, v in sorted(totals.items())},
                "explained_pct": round(100.0 * attributed / wall_ms, 1) if wall_ms > 0 else 0.0,
                "critical_phase": critical,
                "cold_starts": self.cold_starts(),
                "dispatch_queue_ms": round(self.queue_time_ms(), 1)}


class _RecorderRef:
    """Mutable holder so per-task recorders can be swapped under the wrappers."""

    def __init__(self) -> None:
        self.recorder = SpanRecorder()


class _TimedRLock:
    """Session call lock that records real queue wait as dispatch_lock_wait.

    Composition over ``threading.RLock`` (a factory function on some builds,
    not subclassable) exposing the lock protocol ``handle_computer_use`` uses.
    """

    def __init__(self, ref: _RecorderRef) -> None:
        self._ref = ref
        self._lock = threading.RLock()

    def acquire(self, *args: Any, **kwargs: Any) -> bool:
        start = time.monotonic()
        try:
            return self._lock.acquire(*args, **kwargs)
        finally:
            ms = (time.monotonic() - start) * 1000.0
            with self._ref.recorder.span("dispatch_lock_wait",
                                         queue_time_ms=round(ms, 2)):
                pass

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> "_TimedRLock":
        self.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


def _wrap_backend_class(ref: _RecorderRef) -> List[str]:
    """Time CuaDriverBackend.start/capture/input at class level (covers the cold
    start inside _get_backend too). Returns wrapped method names. Idempotent.

    Backend methods call each other (capture -> list_windows, focus_app ->
    list_apps); a thread-local depth guard attributes nested calls to the outer
    span instead of double-counting them.
    """
    from tools.computer_use.cua_backend import CuaDriverBackend

    wrapped: List[str] = []
    cold = {"backend": True}
    depth = threading.local()

    def make_wrapper(name: str, phase: str, original: Callable) -> Callable:
        def timed(self: Any, *args: Any, **kwargs: Any) -> Any:
            if getattr(depth, "level", 0) > 0:
                return original(self, *args, **kwargs)
            dims: Dict[str, Any] = {}
            if phase == "backend_start":
                dims["cold_start"] = cold["backend"]
                cold["backend"] = False
            depth.level = getattr(depth, "level", 0) + 1
            try:
                with ref.recorder.span(phase, **dims):
                    return original(self, *args, **kwargs)
            finally:
                depth.level -= 1
        timed.__name__ = f"_timed_{name}"
        return timed

    for name, phase in [("start", "backend_start"), ("capture", "capture"),
                        *((m, "input") for m in INPUT_METHODS)]:
        original = getattr(CuaDriverBackend, name, None)
        if not callable(original) or getattr(original, "__name__", "").startswith("_timed_"):
            continue
        setattr(CuaDriverBackend, name, make_wrapper(name, phase, original))
        wrapped.append(name)
    return wrapped


@dataclass
class TaskResult:
    name: str
    ok: bool
    detail: str
    report: Dict[str, Any]
    hardware_before: Dict[str, Any]
    hardware_after: Dict[str, Any]


def _call(action: str, args: Dict[str, Any], session_id: str,
          recorder: SpanRecorder) -> Any:
    from tools.computer_use.tool import handle_computer_use
    with recorder.span("total", action=action):
        return handle_computer_use({"action": action, **args}, session_id=session_id)


def _ok(result: Any) -> bool:
    if isinstance(result, dict):
        return result.get("ok", True) is not False and "error" not in result
    if isinstance(result, str):
        try:
            payload = json.loads(result)
        except (TypeError, ValueError):
            return True
        return isinstance(payload, dict) and "error" not in payload
    return True


def _detail(result: Any) -> str:
    if isinstance(result, str):
        try:
            payload = json.loads(result)
            return str(payload.get("error", "ok"))[:120]
        except (TypeError, ValueError):
            return "non-json result"
    if isinstance(result, dict):
        return str(result.get("error", "ok"))[:120]
    return type(result).__name__


def task_perceive(session_id: str, recorder: SpanRecorder) -> TaskResult:
    """Perception loop: AX capture then SOM capture of the real desktop."""
    hw_before = sample_hardware()
    start = time.monotonic()
    with recorder.span("task", task_name="perceive"):
        r1 = _call("capture", {"mode": "ax"}, session_id, recorder)
        r2 = _call("capture", {"mode": "som"}, session_id, recorder)
    ok = _ok(r1) and _ok(r2)
    return TaskResult("perceive", ok, f"ax={_detail(r1)} som={_detail(r2)}",
                      recorder.report((time.monotonic() - start) * 1000.0),
                      hw_before, sample_hardware())


def _ensure_writer(session_id: str) -> bool:
    """Launch LibreOffice Writer under the current display; True if responsive.

    Setup calls use a throwaway recorder so app-launch polling never pollutes
    the measured task spans, and they reuse the runner's session so no second
    backend (and second cold start) is created.
    """
    setup_rec = SpanRecorder()

    def apps() -> str:
        try:
            return json.dumps(_call("list_apps", {}, session_id, setup_rec))
        except Exception:
            return ""

    if "soffice.bin" in apps():
        return True
    subprocess.Popen(["libreoffice", "--writer"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        time.sleep(1.0)
        if "soffice.bin" in apps():
            return True
    return False


def task_act(session_id: str, recorder: SpanRecorder) -> TaskResult:
    """Action loop: focus Writer, type real text, re-capture to observe."""
    hw_before = sample_hardware()
    start = time.monotonic()
    with recorder.span("task", task_name="act"):
        if not _ensure_writer(session_id):
            detail = "libreoffice writer did not appear"
            ok, r1 = False, {}
        else:
            r1 = _call("focus_app", {"app": "soffice.bin"}, session_id, recorder)
            r2 = _call("type", {"text": "hello hermes p0 benchmark"}, session_id, recorder)
            r3 = _call("capture", {"mode": "ax"}, session_id, recorder)
            ok = _ok(r1) and _ok(r2) and _ok(r3)
            detail = (f"focus={_detail(r1)} type={_detail(r2)} "
                      f"capture={_detail(r3)}")
    return TaskResult("act", ok, detail,
                      recorder.report((time.monotonic() - start) * 1000.0),
                      hw_before, sample_hardware())


def task_dialog(session_id: str, recorder: SpanRecorder) -> TaskResult:
    """Dialog loop: open Writer's file dialog, capture it, dismiss with Escape."""
    hw_before = sample_hardware()
    start = time.monotonic()
    with recorder.span("task", task_name="dialog"):
        if not _ensure_writer(session_id):
            ok, detail = False, "libreoffice writer did not appear"
        else:
            r1 = _call("focus_app", {"app": "soffice.bin"}, session_id, recorder)
            r2 = _call("key", {"keys": "ctrl+o"}, session_id, recorder)
            r3 = _call("wait", {"seconds": 1.0}, session_id, recorder)
            r4 = _call("capture", {"mode": "ax"}, session_id, recorder)
            r5 = _call("key", {"keys": "Escape"}, session_id, recorder)
            ok = all(_ok(r) for r in (r1, r2, r3, r4, r5))
            detail = f"open={_detail(r2)} dismiss={_detail(r5)}"
    return TaskResult("dialog", ok, detail,
                      recorder.report((time.monotonic() - start) * 1000.0),
                      hw_before, sample_hardware())


TASKS: Dict[str, Callable[[str, SpanRecorder], TaskResult]] = {
    "perceive": task_perceive,
    "act": task_act,
    "dialog": task_dialog,
}


def _preflight() -> None:
    """Fail fast with actionable errors instead of measuring nothing."""
    if sys.platform == "linux" and not os.environ.get("DISPLAY"):
        raise SystemExit("No DISPLAY: start Xvfb first (Xvfb :99 &) and export DISPLAY=:99.")
    from tools.computer_use.cua_backend_driver import cua_driver_binary_available
    if not cua_driver_binary_available():
        raise SystemExit("cua-driver not installed: run `hermes computer-use install`.")
    from tools.approval import _yolo_active
    if not _yolo_active():
        raise SystemExit("Set HERMES_YOLO_MODE=1: unattended approvals fail closed.")


def run_suite(task_names: List[str], session_id: str) -> Dict[str, Any]:
    """Drive the tasks through the real tool path; return the suite report."""
    _preflight()
    from tools.computer_use import tool as tool_mod

    ref = _RecorderRef()
    wrapped = _wrap_backend_class(ref)
    logger.info("real_workload_runner: wrapped backend methods: %s", wrapped)
    # The cold start happens inside _get_backend, so resolve it under the first
    # task's recorder: the backend_start span is attributed to task one.
    recorders = [SpanRecorder() for _ in task_names]
    ref.recorder = recorders[0]
    suite_start = time.monotonic()
    tool_mod._get_backend(session_id=session_id)
    # App launch is setup, not measured task time: park the recorder on a
    # throwaway while the writer is ensured.
    ref.recorder = SpanRecorder()
    if {"act", "dialog"} & set(task_names):
        _ensure_writer(session_id)
    ref.recorder = recorders[0]
    tool_mod._backend_call_locks[tool_mod._scoped_sid(session_id)] = _TimedRLock(ref)

    results: List[TaskResult] = []
    # The first task's wall includes backend resolution: a real first call pays
    # the cold start, so the backend_start span stays inside its wall clock.
    first = True
    for name, task_recorder in zip(task_names, recorders):
        ref.recorder = task_recorder
        t0 = suite_start if first else time.monotonic()
        first = False
        result = TASKS[name](session_id, task_recorder)
        result.report = task_recorder.report((time.monotonic() - t0) * 1000.0)
        results.append(result)
    wall_ms = (time.monotonic() - suite_start) * 1000.0
    return {"wall_ms": round(wall_ms, 1),
            "wrapped_backend_methods": wrapped,
            "tasks": [{"name": r.name, "ok": r.ok, "detail": r.detail,
                       "report": r.report, "hardware_before": r.hardware_before,
                       "hardware_after": r.hardware_after} for r in results]}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Real-workload P0 measurement runner.")
    parser.add_argument("--tasks", default="perceive,act,dialog",
                        help="comma-separated subset of: perceive,act,dialog")
    parser.add_argument("--session-id", default="p0bench")
    parser.add_argument("--json-out", default="", help="write the full report JSON here")
    args = parser.parse_args(argv)

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in task_names if t not in TASKS]
    if unknown:
        raise SystemExit(f"unknown tasks: {unknown} (choose from {sorted(TASKS)})")

    suite = run_suite(task_names, args.session_id)
    print(json.dumps(suite, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(suite, f, indent=2)

    failed = [t["name"] for t in suite["tasks"] if not t["ok"]]
    if failed:
        print(f"TASKS FAILED: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
