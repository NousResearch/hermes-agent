"""Real computer-use workloads on a live desktop (RFC #112639, P0 slice).

Drives REAL perception -> reasoning -> action -> observation loops through a
live X session (Xvfb in CI/dev): real screenshots via mss, real input events
via pynput, real tkinter apps as the task targets, and a deterministic scripted
policy standing in for the VLM (recorded honestly as policy=scripted on the
`model` phase). Spans use the same shape as the critical-path report (#113225)
so traces feed it directly.

The synthetic module stays as the CI regression fixture; this is the
measurement headline: real phase latencies on real OS operations.

Usage: DISPLAY=:99 python -m tools.computer_use.real_workloads --out traces.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from tools.computer_use.synth_workloads import Span

# Phases here are a subset of the report vocabulary: only phases we measure
# for real on the live path. Reasoning is a scripted policy, not a VLM call.
PHASES = (
    "backend_start", "capture", "element_processing", "model", "input",
    "backend_call", "capture_persist", "total",
)


class SpanRecorder:
    """Wall-clock span recorder; one trace per task."""

    def __init__(self, task_id: str, session_id: str = "") -> None:
        self.task_id = task_id
        self.session_id = session_id
        self.spans: List[Span] = []
        self._seq = 0

    def span(self, phase: str, fn: Callable[[], object],
             attrs: Optional[Dict[str, str]] = None) -> object:
        self._seq += 1
        start = time.perf_counter() * 1000.0
        try:
            return fn()
        finally:
            end = time.perf_counter() * 1000.0
            self.spans.append(Span(
                task_id=self.task_id, phase=phase, start_ms=start, end_ms=end,
                tool_call_id=f"tc-{self._seq:03d}", session_id=self.session_id,
                attrs=dict(attrs or {})))

    def phase_ms(self) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for s in self.spans:
            totals[s.phase] = totals.get(s.phase, 0.0) + s.duration_ms
        return totals


@dataclass
class Trace:
    task_id: str
    task: str
    success: bool
    wall_ms: float
    spans: List[Span]
    phase_ms: Dict[str, float]

    def to_json(self) -> str:
        return json.dumps({
            "task_id": self.task_id, "task": self.task, "success": self.success,
            "wall_ms": self.wall_ms, "phase_ms": self.phase_ms,
            "spans": [{"phase": s.phase, "start_ms": s.start_ms,
                       "end_ms": s.end_ms, "tool_call_id": s.tool_call_id,
                       "session_id": s.session_id, "attrs": s.attrs}
                      for s in self.spans]})


class LiveBackend:
    """Real capture + input against a live X session. Lazy imports: mss and
    pynput are only needed to RUN real tasks, never to import this module."""

    def __init__(self, rec: SpanRecorder) -> None:
        self.rec = rec
        from mss import MSS  # noqa: PLC0415  (needs a live X server)
        from pynput.mouse import Controller as Mouse  # noqa: PLC0415
        self._shot = MSS()
        self._mouse = Mouse()

    def capture(self):
        """Real screenshot of the full display; returns a PIL image."""
        def _grab():
            from PIL import Image  # noqa: PLC0415
            raw = self._shot.grab(self._shot.monitors[0])
            return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        return self.rec.span("capture", _grab)

    def persist(self, img, path: str) -> None:
        self.rec.span("capture_persist", lambda: img.save(path),
                       attrs={"path": path})

    def locate_color(self, img, rgb: Tuple[int, int, int], tol: int = 40
                     ) -> Tuple[int, int]:
        """Real pixel grounding: centroid of the color blob near rgb."""
        def _find():
            px = img.load()
            w, h = img.size
            xs: List[int] = []
            ys: List[int] = []
            r0, g0, b0 = rgb
            for y in range(0, h, 2):  # stride 2 keeps full-HD scans cheap
                for x in range(0, w, 2):
                    r, g, b = px[x, y][:3]
                    if abs(r - r0) <= tol and abs(g - g0) <= tol and abs(b - b0) <= tol:
                        xs.append(x)
                        ys.append(y)
            if not xs:
                raise RuntimeError(f"color {rgb} not found on screen")
            # Densest 40x40 cell wins: robust when the color also appears in
            # scattered UI chrome. Single pass over the matched points.
            from collections import Counter  # noqa: PLC0415
            cells = Counter((x // 40, y // 40) for x, y in zip(xs, ys))
            (ccx, ccy), _n = cells.most_common(1)[0]
            return (ccx * 40 + 20, ccy * 40 + 20)
        return self.rec.span("element_processing", _find,
                             attrs={"method": "color-blob", "rgb": str(rgb)})

    def ax_rect(self, widget) -> Tuple[int, int, int, int]:
        """Widget geometry from the live windowing system (the AX-tree stand-in:
        what the OS accessibility layer hands the real backend)."""
        def _geom():
            widget.update_idletasks()
            return (widget.winfo_rootx(), widget.winfo_rooty(),
                    widget.winfo_width(), widget.winfo_height())
        return self.rec.span("element_processing", _geom,
                             attrs={"method": "ax-geom"})

    def click(self, x: int, y: int) -> None:
        self.rec.span("input", lambda: self._click_at(x, y),
                      attrs={"x": str(x), "y": str(y)})

    def _click_at(self, x: int, y: int) -> None:
        from pynput.mouse import Button  # noqa: PLC0415
        self._mouse.position = (x, y)
        time.sleep(0.05)
        self._mouse.press(Button.left)
        time.sleep(0.03)
        self._mouse.release(Button.left)

    def type_text(self, text: str) -> None:
        # XTest all the way down (like the mouse path): pynput's keyboard
        # controller falls back to XSendEvent for regular keys, which the
        # toolkit drops; fake_input is what real automation backends use.
        def _type():
            from Xlib import X, XK  # noqa: PLC0415
            from Xlib.display import Display  # noqa: PLC0415
            from Xlib.ext import xtest  # noqa: PLC0415
            d = Display()
            shift_kc = d.keysym_to_keycode(XK.string_to_keysym("Shift_L"))
            _names = {" ": "space", ".": "period", ",": "comma",
                      "@": "at", "-": "minus", "_": "underscore"}
            # US layout: these keysyms live on the shifted layer.
            _shifted = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ@_")
            try:
                for ch in text:
                    ks = XK.string_to_keysym(_names.get(ch, ch))
                    if ks == 0:
                        raise ValueError(f"no keysym for {ch!r}")
                    kc = d.keysym_to_keycode(ks)
                    shifted = ch in _shifted
                    if shifted:
                        xtest.fake_input(d, X.KeyPress, shift_kc)
                        d.sync()
                    xtest.fake_input(d, X.KeyPress, kc)
                    d.sync()
                    xtest.fake_input(d, X.KeyRelease, kc)
                    d.sync()
                    if shifted:
                        xtest.fake_input(d, X.KeyRelease, shift_kc)
                        d.sync()
            finally:
                d.close()
        self.rec.span("input", _type, attrs={"chars": str(len(text))})

    def backend_call(self, name: str, fn: Callable[[], object]) -> object:
        return self.rec.span("backend_call", fn, attrs={"action": name})

    def close(self) -> None:
        self._shot.close()


def _scripted_policy(task: str, observation: dict) -> List[Tuple[str, tuple]]:
    """Deterministic policy: observation -> action list. Timed as `model`."""
    if task == "form_fill":
        o = observation
        return [("click", o["name"]), ("type", ("Ada Lovelace",)),
                ("click", o["email"]), ("type", ("ada@example.com",)),
                ("click", o["submit"])]
    if task == "dialog_dismiss":
        return [("click", observation["ok"])]
    if task == "text_entry":
        return [("click", observation["field"]),
                ("type", ("hello hermes",))]
    raise ValueError(f"unknown task {task}")


def _tk() -> object:
    import tkinter as tk  # noqa: PLC0415  (needs a live X server)
    return tk


def run_form_fill(backend_factory) -> Trace:
    """Fill a real form window: click two entries, type, submit."""
    task_id = "real-form-fill"
    rec = SpanRecorder(task_id, session_id="real-session")
    t0 = time.perf_counter() * 1000.0
    tk = _tk()
    state = {"submitted": None}
    root = None
    try:
        root = rec.span("backend_start", lambda: _build_form(tk, state))
        be = backend_factory(rec)
        img = be.capture()
        name = be.ax_rect(root.nametowidget(".name"))
        email = be.ax_rect(root.nametowidget(".email"))
        submit_rect = be.ax_rect(root.nametowidget(".submit"))
        blob = be.locate_color(img, (46, 204, 113))  # the green button
        obs = {"name": _center(name), "email": _center(email),
               "submit": _snap_to_rect(blob, submit_rect)}
        actions = rec.span("model", lambda: _scripted_policy("form_fill", obs),
                           attrs={"policy": "scripted"})
        for kind, arg in actions:
            if kind == "click":
                be.backend_call("click", lambda a=arg: be.click(*a))
            else:
                be.backend_call("type", lambda a=arg: be.type_text(a[0]))
            root.update()  # pump the app's event loop so inputs take effect
        time.sleep(0.3)
        root.update()
        be.capture()  # real observation after acting
        success = state["submitted"] == ("Ada Lovelace", "ada@example.com")
    finally:
        if root is not None:
            root.destroy()
    wall = time.perf_counter() * 1000.0 - t0
    rec.spans.append(Span(task_id=task_id, phase="total",
                          start_ms=t0, end_ms=t0 + wall))
    return Trace(task_id, "form_fill", success, wall, rec.spans, rec.phase_ms())


def _build_form(tk, state):
    root = tk.Tk()
    root.geometry("520x320+200+150")
    root.title("CU benchmark form")
    tk.Label(root, text="Name:").pack(pady=(20, 0))
    name = tk.Entry(root, width=30, name="name")
    name.pack()
    tk.Label(root, text="Email:").pack(pady=(10, 0))
    email = tk.Entry(root, width=30, name="email")
    email.pack()
    btn = tk.Button(root, text="Submit", bg="#2ecc71", activebackground="#2ecc71",
                    width=12, name="submit", command=lambda: state.update(
                        submitted=(name.get(), email.get())))
    btn.pack(pady=20)
    root.update()
    time.sleep(0.4)  # let the X server map the window
    return root


def _center(rect: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x, y, w, h = rect
    return (x + w // 2, y + h // 2)


def _snap_to_rect(pt: Tuple[int, int], rect: Tuple[int, int, int, int]
                  ) -> Tuple[int, int]:
    """Use the pixel-grounded point when it lands inside the AX rect (pixels
    agree with the tree); otherwise the AX center. Both are real signals."""
    x, y, w, h = rect
    px, py = pt
    if x <= px <= x + w and y <= py <= y + h:
        return pt
    return _center(rect)


def run_dialog_dismiss(backend_factory) -> Trace:
    """Dismiss a real modal dialog by clicking its OK button."""
    task_id = "real-dialog-dismiss"
    rec = SpanRecorder(task_id, session_id="real-session")
    t0 = time.perf_counter() * 1000.0
    tk = _tk()
    root = tk.Tk()
    root.withdraw()
    dlg = None
    try:
        def _build():
            d = tk.Toplevel(root)
            d.geometry("360x180+300+250")
            d.title("Confirm")
            tk.Label(d, text="Are you sure?").pack(pady=20)
            ok_btn = tk.Button(d, text="OK", bg="#3498db",
                               activebackground="#3498db",
                               width=10, command=d.destroy, name="ok")
            ok_btn.pack()
            d.update()
            time.sleep(0.4)
            return d, ok_btn
        dlg, ok_btn = rec.span("backend_start", _build)
        be = backend_factory(rec)
        img = be.capture()
        ok_rect = be.ax_rect(ok_btn)
        blob = be.locate_color(img, (52, 152, 219))  # the blue OK button
        ok = _snap_to_rect(blob, ok_rect)  # pixels agree with the AX tree?
        actions = rec.span("model",
                           lambda: _scripted_policy("dialog_dismiss", {"ok": ok}),
                           attrs={"policy": "scripted"})
        for kind, arg in actions:
            be.backend_call("click", lambda a=arg: be.click(*a))
            root.update()  # pump the app's event loop so the click lands
        time.sleep(0.3)
        root.update()
        be.capture()
        success = not dlg.winfo_exists()
    finally:
        root.destroy()
    wall = time.perf_counter() * 1000.0 - t0
    rec.spans.append(Span(task_id=task_id, phase="total",
                          start_ms=t0, end_ms=t0 + wall))
    return Trace(task_id, "dialog_dismiss", success, wall, rec.spans,
                 rec.phase_ms())


def run_text_entry(backend_factory) -> Trace:
    """Click a real text widget and type a sentence into it."""
    task_id = "real-text-entry"
    rec = SpanRecorder(task_id, session_id="real-session")
    t0 = time.perf_counter() * 1000.0
    tk = _tk()
    root = None
    try:
        def _build():
            r = tk.Tk()
            r.geometry("480x260+200+150")
            r.title("Notes")
            t = r.text = tk.Text(r, width=40, height=8, name="field")
            t.pack(padx=20, pady=20)
            r.update()
            time.sleep(0.4)
            return r
        root = rec.span("backend_start", _build)
        be = backend_factory(rec)
        be.capture()
        field = be.ax_rect(root.nametowidget(".field"))
        actions = rec.span("model",
                           lambda: _scripted_policy(
                               "text_entry", {"field": _center(field)}),
                           attrs={"policy": "scripted"})
        for kind, arg in actions:
            if kind == "click":
                be.backend_call("click", lambda a=arg: be.click(*a))
            else:
                be.backend_call("type", lambda a=arg: be.type_text(a[0]))
            root.update()  # pump the app's event loop so inputs take effect
        time.sleep(0.3)
        root.update()
        be.capture()
        got = root.text.get("1.0", "end").strip()
        success = got == "hello hermes"
    finally:
        if root is not None:
            root.destroy()
    wall = time.perf_counter() * 1000.0 - t0
    rec.spans.append(Span(task_id=task_id, phase="total",
                          start_ms=t0, end_ms=t0 + wall))
    return Trace(task_id, "text_entry", success, wall, rec.spans, rec.phase_ms())


TASKS: Dict[str, Callable] = {
    "form_fill": run_form_fill,
    "dialog_dismiss": run_dialog_dismiss,
    "text_entry": run_text_entry,
}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run real CU workloads on Xvfb.")
    ap.add_argument("--tasks", nargs="*", default=sorted(TASKS),
                    choices=sorted(TASKS))
    ap.add_argument("--out", default="",
                    help="Write one JSON trace per line to this file.")
    args = ap.parse_args(argv)
    out = open(args.out, "w", encoding="utf-8") if args.out else None
    try:
        for name in args.tasks:
            trace = TASKS[name](LiveBackend)
            line = trace.to_json()
            print(f"{trace.task_id}: success={trace.success} "
                  f"wall={trace.wall_ms:.0f}ms "
                  + " ".join(f"{p}={ms:.0f}" for p, ms in
                              sorted(trace.phase_ms.items()) if p != "total"))
            if out:
                out.write(line + "\n")
            if not trace.success:
                print(f"TASK FAILED: {name}", file=sys.stderr)
                return 1
    finally:
        if out:
            out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
