"""On-screen overlay for Windows computer_use — the visible "PC use mode".

Spawned as a subprocess by windows_backend. A fullscreen, transparent,
click-through, always-on-top tkinter window spanning the whole virtual
desktop. It shows:

  * a persistent banner pill while desktop control is active,
  * the numbered SOM element boxes after each capture (what Hermes sees),
  * click ripples / drag lines where actions land,
  * short action flashes ("typing…", "key ctrl+s").

The window is excluded from screen capture via SetWindowDisplayAffinity
(WDA_EXCLUDEFROMCAPTURE), so Hermes' own screenshots never contain it —
the user sees the overlay, the model does not.

IPC: JSON datagrams over localhost UDP. On startup the process binds an
ephemeral port and prints ``PORT <n>`` on stdout; the parent reads that
line. The process exits when it receives {"cmd": "bye"} or when its stdin
closes (parent process died).

Messages:
  {"cmd": "banner", "text": str, "state": "active"|"acting"}
  {"cmd": "elements", "items": [{"index": int, "bounds": [x,y,w,h]}], "ttl": float}
  {"cmd": "click", "x": int, "y": int}
  {"cmd": "drag", "from": [x,y], "to": [x,y]}
  {"cmd": "flash", "text": str, "ttl": float}
  {"cmd": "clear"}
  {"cmd": "bye"}
"""

from __future__ import annotations

import ctypes
import json
import queue
import socket
import sys
import threading
import time
import tkinter as tk

# Any pixel painted in this exact color becomes fully transparent AND
# click-through (tk colorkey transparency). Obscure color to avoid clashes.
_TRANS = "#010203"

_GWL_EXSTYLE = -20
_WS_EX_TRANSPARENT = 0x00000020
_WS_EX_TOOLWINDOW = 0x00000080
_WS_EX_NOACTIVATE = 0x08000000
_WDA_EXCLUDEFROMCAPTURE = 0x00000011

_TICK_MS = 50


def _set_dpi_awareness() -> None:
    user32 = ctypes.windll.user32
    try:
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass


class OverlayApp:
    def __init__(self) -> None:
        user32 = ctypes.windll.user32
        self.vx = user32.GetSystemMetrics(76)
        self.vy = user32.GetSystemMetrics(77)
        self.vw = user32.GetSystemMetrics(78)
        self.vh = user32.GetSystemMetrics(79)
        prev_fg = user32.GetForegroundWindow()

        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.geometry(f"{self.vw}x{self.vh}+{self.vx}+{self.vy}")
        self.root.attributes("-topmost", True)
        self.root.attributes("-transparentcolor", _TRANS)
        self.root.configure(bg=_TRANS)
        self.canvas = tk.Canvas(self.root, bg=_TRANS, highlightthickness=0,
                                width=self.vw, height=self.vh)
        self.canvas.pack(fill="both", expand=True)
        self.root.update_idletasks()
        self._apply_window_styles()
        # Mapping the window can steal foreground before WS_EX_NOACTIVATE
        # lands — hand focus back to whoever had it.
        try:
            if prev_fg and user32.GetForegroundWindow() == self._hwnd():
                user32.SetForegroundWindow(prev_fg)
        except Exception:
            pass

        self.msgs: "queue.Queue[dict]" = queue.Queue()
        # Renderer state.
        self.banner_text = "HERMES — DESKTOP CONTROL"
        self.banner_state = "active"
        self.banner_until = 0.0          # acting-state pulse expiry
        self.elements: list = []         # [{"index", "bounds"}]
        self.elements_until = 0.0
        self.ripples: list = []          # [(x, y, t0)]
        self.drags: list = []            # [(x1, y1, x2, y2, t0)]
        self.flash_text = ""
        self.flash_until = 0.0
        self._last_topmost = 0.0

        self.port = self._start_udp_listener()
        threading.Thread(target=self._watch_stdin, daemon=True).start()

    # ── window plumbing ─────────────────────────────────────────────
    def _hwnd(self) -> int:
        # GA_ROOT resolves the real OS top-level window. GetParent() of the
        # canvas only reaches tk's inner frame — display affinity and
        # click-through styles silently fail on child windows.
        return ctypes.windll.user32.GetAncestor(self.canvas.winfo_id(), 2)

    def _apply_window_styles(self) -> None:
        user32 = ctypes.windll.user32
        hwnd = self._hwnd()
        style = user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
        style |= _WS_EX_TRANSPARENT | _WS_EX_TOOLWINDOW | _WS_EX_NOACTIVATE
        user32.SetWindowLongW(hwnd, _GWL_EXSTYLE, style)
        # Hide from Hermes' own screenshots. Win10 2004+; on failure the
        # backend's post-capture element sends still keep captures clean,
        # but the banner would be visible to the model — log and continue.
        if not user32.SetWindowDisplayAffinity(hwnd, _WDA_EXCLUDEFROMCAPTURE):
            print("WARN display affinity failed; overlay may appear in captures",
                  flush=True)

    # ── IPC ─────────────────────────────────────────────────────────
    def _start_udp_listener(self) -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

        def loop() -> None:
            while True:
                try:
                    data, _addr = sock.recvfrom(1 << 20)
                    self.msgs.put(json.loads(data.decode("utf-8")))
                except Exception:
                    continue

        threading.Thread(target=loop, daemon=True).start()
        return port

    def _watch_stdin(self) -> None:
        """Exit when the parent process dies (stdin EOF)."""
        try:
            sys.stdin.buffer.read()
        except Exception:
            pass
        self.msgs.put({"cmd": "bye"})

    # ── message handling ────────────────────────────────────────────
    def _drain(self) -> bool:
        alive = True
        while True:
            try:
                m = self.msgs.get_nowait()
            except queue.Empty:
                return alive
            cmd = m.get("cmd")
            now = time.monotonic()
            if cmd == "bye":
                alive = False
            elif cmd == "banner":
                self.banner_text = str(m.get("text") or self.banner_text)
                self.banner_state = str(m.get("state") or "active")
            elif cmd == "elements":
                self.elements = list(m.get("items") or [])
                self.elements_until = now + float(m.get("ttl", 4.0))
            elif cmd == "click":
                self.ripples.append((int(m["x"]), int(m["y"]), now))
            elif cmd == "drag":
                (x1, y1), (x2, y2) = m["from"], m["to"]
                self.drags.append((int(x1), int(y1), int(x2), int(y2), now))
            elif cmd == "flash":
                self.flash_text = str(m.get("text") or "")
                self.flash_until = now + float(m.get("ttl", 1.5))
                self.banner_until = now + 1.0
            elif cmd == "clear":
                self.elements = []
                self.ripples = []
                self.drags = []
                self.flash_text = ""

    # ── rendering ───────────────────────────────────────────────────
    def _draw(self) -> None:
        c = self.canvas
        c.delete("all")
        now = time.monotonic()

        # Expire transients.
        if now > self.elements_until:
            self.elements = []
        self.ripples = [r for r in self.ripples if now - r[2] < 0.9]
        self.drags = [d for d in self.drags if now - d[4] < 1.2]

        # Element boxes — mirror of what Hermes sees on her screenshot.
        for e in self.elements:
            try:
                x, y, w, h = e["bounds"]
            except Exception:
                continue
            x, y = x - self.vx, y - self.vy
            c.create_rectangle(x, y, x + w, y + h, outline="#ff2d2d", width=2)
            label = str(e.get("index", "?"))
            bw = 7 * len(label) + 8
            c.create_rectangle(x, y, x + bw, y + 16, fill="#ff2d2d", outline="")
            c.create_text(x + bw / 2, y + 8, text=label, fill="white",
                          font=("Segoe UI", 8, "bold"))

        # Click ripples — expanding rings.
        for (x, y, t0) in self.ripples:
            age = now - t0
            x, y = x - self.vx, y - self.vy
            for k in range(3):
                r = 6 + (age * 70) + k * 9
                c.create_oval(x - r, y - r, x + r, y + r,
                              outline="#ffb02d", width=max(1, 3 - k))

        # Drag lines.
        for (x1, y1, x2, y2, _t0) in self.drags:
            c.create_line(x1 - self.vx, y1 - self.vy, x2 - self.vx, y2 - self.vy,
                          fill="#ffb02d", width=3, arrow="last")

        # Banner pill, top-center of the PRIMARY monitor (origin 0,0).
        acting = now < self.banner_until
        dot = "#ffb02d" if acting else "#3ddc84"
        text = self.banner_text
        if self.flash_text and now < self.flash_until:
            text = f"{self.banner_text}   ·   {self.flash_text}"
        px = -self.vx + ctypes.windll.user32.GetSystemMetrics(0) // 2
        tw = max(220, 8 * len(text) + 50)
        x1, y1 = px - tw // 2, -self.vy + 8
        x2, y2 = px + tw // 2, -self.vy + 42
        c.create_rectangle(x1, y1, x2, y2, fill="#1b1d22", outline="#3a3d45")
        c.create_oval(x1 + 12, (y1 + y2) / 2 - 5, x1 + 22, (y1 + y2) / 2 + 5,
                      fill=dot, outline="")
        c.create_text((x1 + x2) / 2 + 8, (y1 + y2) / 2, text=text,
                      fill="#e8e9ec", font=("Segoe UI", 10, "bold"))

    def _tick(self) -> None:
        if not self._drain():
            self.root.destroy()
            return
        self._draw()
        now = time.monotonic()
        if now - self._last_topmost > 2.0:
            self.root.attributes("-topmost", True)
            self._last_topmost = now
        self.root.after(_TICK_MS, self._tick)

    def run(self) -> None:
        print(f"PORT {self.port}", flush=True)
        self.root.after(_TICK_MS, self._tick)
        self.root.mainloop()


def main() -> None:
    _set_dpi_awareness()
    OverlayApp().run()


if __name__ == "__main__":
    main()
