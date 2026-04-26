/**
 * ChatPage — embeds `hermes --tui` inside the dashboard.
 *
 *   <div host> (dashboard chrome)                                         .
 *     └─ <div wrapper> (rounded, dark bg, padded — the "terminal window"  .
 *         look that gives the page a distinct visual identity)            .
 *         └─ @xterm/xterm Terminal (WebGL renderer, Unicode 11 widths)    .
 *              │ onData      keystrokes → WebSocket → PTY master          .
 *              │ onResize    terminal resize → `\x1b[RESIZE:cols;rows]`   .
 *              │ write(data) PTY output bytes → VT100 parser              .
 *              ▼                                                          .
 *     WebSocket /api/pty?token=<session>                                  .
 *          ▼                                                              .
 *     FastAPI pty_ws  (hermes_cli/web_server.py)                          .
 *          ▼                                                              .
 *     POSIX PTY → `node ui-tui/dist/entry.js` → tui_gateway + AIAgent     .
 */

import { FitAddon } from "@xterm/addon-fit";
import { Unicode11Addon } from "@xterm/addon-unicode11";
import { WebLinksAddon } from "@xterm/addon-web-links";
import { WebglAddon } from "@xterm/addon-webgl";
import { Terminal } from "@xterm/xterm";
import "@xterm/xterm/css/xterm.css";
import { Typography } from "@nous-research/ui";
import { cn } from "@/lib/utils";
import { Copy, PanelRight, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useSearchParams } from "react-router-dom";

import { ChatSidebar } from "@/components/ChatSidebar";
import { usePageHeader } from "@/contexts/usePageHeader";
import { useI18n } from "@/i18n";
import { PluginSlot } from "@/plugins";

function buildWsUrl(
  token: string,
  resume: string | null,
  channel: string,
): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const qs = new URLSearchParams({ token, channel });
  if (resume) qs.set("resume", resume);
  return `${proto}//${window.location.host}/api/pty?${qs.toString()}`;
}

// Channel id ties this chat tab's PTY child (publisher) to its sidebar
// (subscriber).  Generated once per mount so a tab refresh starts a fresh
// channel — the previous PTY child terminates with the old WS, and its
// channel auto-evicts when no subscribers remain.
function generateChannelId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `chat-${Math.random().toString(36).slice(2)}-${Date.now().toString(36)}`;
}

// Colors for the terminal body.  Matches the dashboard's dark teal canvas
// with cream foreground — we intentionally don't pick monokai or a loud
// theme, because the TUI's skin engine already paints the content; the
// terminal chrome just needs to sit quietly inside the dashboard.
const TERMINAL_THEME = {
  background: "#0d2626",
  foreground: "#f0e6d2",
  cursor: "#f0e6d2",
  cursorAccent: "#0d2626",
  selectionBackground: "#f0e6d244",
};

/**
 * CSS width for xterm font tiers.
 *
 * Prefer the terminal host's `clientWidth` — Chrome DevTools device mode often
 * keeps `window.innerWidth` at the full desktop value while the *drawn* layout
 * is phone-sized, which made us pick desktop font sizes (~14px) and look huge.
 */
function terminalTierWidthPx(host: HTMLElement | null): number {
  if (typeof window === "undefined") return 1280;
  const fromHost = host?.clientWidth ?? 0;
  if (fromHost > 2) return Math.round(fromHost);
  const doc = document.documentElement?.clientWidth ?? 0;
  const vv = window.visualViewport;
  const inner = window.innerWidth;
  const vvw = vv?.width ?? inner;
  const layout = Math.min(inner, vvw, doc > 0 ? doc : inner);
  return Math.max(1, Math.round(layout));
}

function terminalFontSizeForWidth(layoutWidthPx: number): number {
  if (layoutWidthPx < 300) return 7;
  if (layoutWidthPx < 360) return 8;
  if (layoutWidthPx < 420) return 9;
  if (layoutWidthPx < 520) return 10;
  if (layoutWidthPx < 720) return 11;
  if (layoutWidthPx < 1024) return 12;
  return 14;
}

function terminalLineHeightForWidth(layoutWidthPx: number): number {
  return layoutWidthPx < 1024 ? 1.02 : 1.15;
}

export default function ChatPage() {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const termRef = useRef<Terminal | null>(null);
  const fitRef = useRef<FitAddon | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [searchParams] = useSearchParams();
  // Lazy-init: the missing-token check happens at construction so the effect
  // body doesn't have to setState (React 19's set-state-in-effect rule).
  const [banner, setBanner] = useState<string | null>(() =>
    typeof window !== "undefined" && !window.__HERMES_SESSION_TOKEN__
      ? "Session token unavailable. Open this page through `hermes dashboard`, not directly."
      : null,
  );
  const [copyState, setCopyState] = useState<"idle" | "copied">("idle");
  const copyResetRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [mobilePanelOpen, setMobilePanelOpen] = useState(false);
  const { setEnd } = usePageHeader();
  const { t } = useI18n();
  const closeMobilePanel = useCallback(() => setMobilePanelOpen(false), []);
  const modelToolsLabel = useMemo(
    () => `${t.app.modelToolsSheetTitle} ${t.app.modelToolsSheetSubtitle}`,
    [t.app.modelToolsSheetSubtitle, t.app.modelToolsSheetTitle],
  );
  const [portalRoot] = useState<HTMLElement | null>(() =>
    typeof document !== "undefined" ? document.body : null,
  );
  const [narrow, setNarrow] = useState(() =>
    typeof window !== "undefined"
      ? window.matchMedia("(max-width: 1023px)").matches
      : false,
  );

  const resumeRef = useRef<string | null>(searchParams.get("resume"));
  const channel = useMemo(() => generateChannelId(), []);

  useEffect(() => {
    const mql = window.matchMedia("(max-width: 1023px)");
    const sync = () => setNarrow(mql.matches);
    sync();
    mql.addEventListener("change", sync);
    return () => mql.removeEventListener("change", sync);
  }, []);

  useEffect(() => {
    if (!mobilePanelOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeMobilePanel();
    };
    document.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [mobilePanelOpen, closeMobilePanel]);

  useEffect(() => {
    const mql = window.matchMedia("(min-width: 1024px)");
    const onChange = (e: MediaQueryListEvent) => {
      if (e.matches) setMobilePanelOpen(false);
    };
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    if (!narrow) {
      setEnd(null);
      return;
    }
    setEnd(
      <button
        type="button"
        onClick={() => setMobilePanelOpen(true)}
        className={cn(
          "inline-flex items-center gap-1.5 rounded border border-current/20",
          "px-2 py-1 text-[0.65rem] font-medium tracking-wide normal-case",
          "text-midground/80 hover:text-midground hover:bg-midground/5",
          "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-midground",
          "shrink-0 cursor-pointer",
        )}
        aria-expanded={mobilePanelOpen}
        aria-controls="chat-side-panel"
      >
        <PanelRight className="h-3 w-3 shrink-0" />
        {modelToolsLabel}
      </button>,
    );
    return () => setEnd(null);
  }, [narrow, mobilePanelOpen, modelToolsLabel, setEnd]);

  const handleCopyLast = () => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    // Send the slash as a burst, wait long enough for Ink's tokenizer to
    // emit a keypress event for each character (not coalesce them into a
    // paste), then send Return as its own event.  The timing here is
    // empirical — 100ms is safely past Node's default stdin coalescing
    // window and well inside UI responsiveness.
    ws.send("/copy");
    setTimeout(() => {
      const s = wsRef.current;
      if (s && s.readyState === WebSocket.OPEN) s.send("\r");
    }, 100);
    setCopyState("copied");
    if (copyResetRef.current) clearTimeout(copyResetRef.current);
    copyResetRef.current = setTimeout(() => setCopyState("idle"), 1500);
    termRef.current?.focus();
  };

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const token = window.__HERMES_SESSION_TOKEN__;
    // Banner already initialised above; just bail before wiring xterm/WS.
    if (!token) {
      return;
    }

    const tierW0 = terminalTierWidthPx(host);
    const term = new Terminal({
      allowProposedApi: true,
      cursorBlink: true,
      fontFamily:
        "'JetBrains Mono', 'Cascadia Mono', 'Fira Code', 'MesloLGS NF', 'Source Code Pro', Menlo, Consolas, 'DejaVu Sans Mono', monospace",
      fontSize: terminalFontSizeForWidth(tierW0),
      lineHeight: terminalLineHeightForWidth(tierW0),
      letterSpacing: 0,
      fontWeight: "400",
      fontWeightBold: "700",
      macOptionIsMeta: true,
      scrollback: 0,
      theme: TERMINAL_THEME,
    });
    termRef.current = term;

    // --- Clipboard integration ---------------------------------------
    //
    // Three independent paths all route to the system clipboard:
    //
    //   1. **Selection → Ctrl+C (or Cmd+C on macOS).**  Ink's own handler
    //      in useInputHandlers.ts turns Ctrl+C into a copy when the
    //      terminal has a selection, then emits an OSC 52 escape.  Our
    //      OSC 52 handler below decodes that escape and writes to the
    //      browser clipboard — so the flow works just like it does in
    //      `hermes --tui`.
    //
    //   2. **Ctrl/Cmd+Shift+C.**  Belt-and-suspenders shortcut that
    //      operates directly on xterm's selection, useful if the TUI
    //      ever stops listening (e.g. overlays / pickers) or if the user
    //      has selected with the mouse outside of Ink's selection model.
    //
    //   3. **Ctrl/Cmd+Shift+V.**  Reads the system clipboard and feeds
    //      it to the terminal as keyboard input.  xterm's paste() wraps
    //      it with bracketed-paste if the host has that mode enabled.
    //
    // OSC 52 reads (terminal asking to read the clipboard) are not
    // supported — that would let any content the TUI renders exfiltrate
    // the user's clipboard.
    term.parser.registerOscHandler(52, (data) => {
      // Format: "<targets>;<base64 | '?'>"
      const semi = data.indexOf(";");
      if (semi < 0) return false;
      const payload = data.slice(semi + 1);
      if (payload === "?" || payload === "") return false; // read/clear — ignore
      try {
        // atob returns a binary string (one byte per char); we need UTF-8
        // decode so multi-byte codepoints (≥, →, emoji, CJK) round-trip
        // correctly.  Without this step, the three UTF-8 bytes of `≥`
        // would land in the clipboard as the three separate Latin-1
        // characters `â‰¥`.
        const binary = atob(payload);
        const bytes = Uint8Array.from(binary, (c) => c.charCodeAt(0));
        const text = new TextDecoder("utf-8").decode(bytes);
        navigator.clipboard.writeText(text).catch(() => {});
      } catch {
        // Malformed base64 — silently drop.
      }
      return true;
    });

    const isMac =
      typeof navigator !== "undefined" && /Mac/i.test(navigator.platform);

    term.attachCustomKeyEventHandler((ev) => {
      if (ev.type !== "keydown") return true;

      const copyModifier = isMac ? ev.metaKey : ev.ctrlKey && ev.shiftKey;
      const pasteModifier = isMac ? ev.metaKey : ev.ctrlKey && ev.shiftKey;

      if (copyModifier && ev.key.toLowerCase() === "c") {
        const sel = term.getSelection();
        if (sel) {
          navigator.clipboard.writeText(sel).catch(() => {});
          ev.preventDefault();
          return false;
        }
      }

      if (pasteModifier && ev.key.toLowerCase() === "v") {
        navigator.clipboard
          .readText()
          .then((text) => {
            if (text) term.paste(text);
          })
          .catch(() => {});
        ev.preventDefault();
        return false;
      }

      return true;
    });

    const fit = new FitAddon();
    fitRef.current = fit;
    term.loadAddon(fit);

    const unicode11 = new Unicode11Addon();
    term.loadAddon(unicode11);
    term.unicode.activeVersion = "11";

    term.loadAddon(new WebLinksAddon());

    term.open(host);

    // WebGL draws from a texture atlas sized with device pixels. On phones and
    // in DevTools device mode that often produces *visually* much larger cells
    // than `fontSize` suggests — users see "huge" text even at 7–9px settings.
    // The canvas/DOM renderer tracks `fontSize` faithfully; use it for narrow
    // hosts.  Wide layouts still get WebGL for crisp box-drawing.
    const useWebgl = terminalTierWidthPx(host) >= 768;
    if (useWebgl) {
      try {
        const webgl = new WebglAddon();
        webgl.onContextLoss(() => webgl.dispose());
        term.loadAddon(webgl);
      } catch (err) {
        console.warn(
          "[hermes-chat] WebGL renderer unavailable; falling back to default",
          err,
        );
      }
    }

    // Initial fit + resize observer.  fit.fit() reads the container's
    // current bounding box and resizes the terminal grid to match.
    //
    // The subtle bit: the dashboard has CSS transitions on the container
    // (backdrop fade-in, rounded corners settling as fonts load).  If we
    // call fit() at mount time, the bounding box we measure is often 1-2
    // cell widths off from the final size.  ResizeObserver *does* fire
    // when the container settles, but if the pixel delta happens to be
    // smaller than one cell's width, fit() computes the same integer
    // (cols, rows) as before and doesn't emit onResize — so the PTY
    // never learns the final size.  Users see truncated long lines until
    // they resize the browser window.
    //
    // We force one extra fit + explicit RESIZE send after two animation
    // frames.  rAF→rAF guarantees one layout commit between the two
    // callbacks, giving CSS transitions and font metrics time to finalize
    // before we take the authoritative measurement.
    let hostSyncRaf = 0;
    const scheduleHostSync = () => {
      if (hostSyncRaf) return;
      hostSyncRaf = requestAnimationFrame(() => {
        hostSyncRaf = 0;
        syncTerminalMetrics();
      });
    };

    let metricsDebounce: ReturnType<typeof setTimeout> | null = null;
    const syncTerminalMetrics = () => {
      const w = terminalTierWidthPx(host);
      const nextSize = terminalFontSizeForWidth(w);
      const nextLh = terminalLineHeightForWidth(w);
      const fontChanged =
        term.options.fontSize !== nextSize ||
        term.options.lineHeight !== nextLh;
      if (fontChanged) {
        term.options.fontSize = nextSize;
        term.options.lineHeight = nextLh;
      }
      try {
        fit.fit();
      } catch {
        return;
      }
      if (fontChanged && term.rows > 0) {
        try {
          term.refresh(0, term.rows - 1);
        } catch {
          /* ignore */
        }
      }
      if (
        fontChanged &&
        wsRef.current &&
        wsRef.current.readyState === WebSocket.OPEN
      ) {
        wsRef.current.send(`\x1b[RESIZE:${term.cols};${term.rows}]`);
      }
    };

    const scheduleSyncTerminalMetrics = () => {
      if (metricsDebounce) clearTimeout(metricsDebounce);
      metricsDebounce = setTimeout(() => {
        metricsDebounce = null;
        syncTerminalMetrics();
      }, 60);
    };

    const ro = new ResizeObserver(() => scheduleHostSync());
    ro.observe(host);

    window.addEventListener("resize", scheduleSyncTerminalMetrics);
    window.visualViewport?.addEventListener("resize", scheduleSyncTerminalMetrics);
    window.visualViewport?.addEventListener("scroll", scheduleSyncTerminalMetrics);
    scheduleHostSync();
    requestAnimationFrame(() => scheduleHostSync());

    // Double-rAF authoritative fit.  On the second frame the layout has
    // committed at least once since mount; fit.fit() then reads the
    // stable container size.  We always send a RESIZE escape afterwards
    // (even if fit's cols/rows didn't change, so the PTY has the same
    // dims registered as our JS state — prevents a drift where Ink
    // thinks the terminal is one col bigger than what's on screen).
    let settleRaf1 = 0;
    let settleRaf2 = 0;
    settleRaf1 = requestAnimationFrame(() => {
      settleRaf1 = 0;
      settleRaf2 = requestAnimationFrame(() => {
        settleRaf2 = 0;
        syncTerminalMetrics();
      });
    });

    // WebSocket
    const url = buildWsUrl(token, resumeRef.current, channel);
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;
    // Suppress banner/terminal side-effects when cleanup() calls `ws.close()`
    // (React StrictMode remount, route change) so we never write to a
    // disposed xterm or setState on an unmounted tree.
    let unmounting = false;

    ws.onopen = () => {
      setBanner(null);
      // Send the initial RESIZE immediately so Ink has *a* size to lay
      // out against on its first paint.  The double-rAF block above will
      // follow up with the authoritative measurement — at worst Ink
      // reflows once after the PTY boots, which is imperceptible.
      ws.send(`\x1b[RESIZE:${term.cols};${term.rows}]`);
    };

    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        term.write(ev.data);
      } else {
        term.write(new Uint8Array(ev.data as ArrayBuffer));
      }
    };

    ws.onclose = (ev) => {
      wsRef.current = null;
      if (unmounting) {
        return;
      }
      if (ev.code === 4401) {
        setBanner("Auth failed. Reload the page to refresh the session token.");
        return;
      }
      if (ev.code === 4403) {
        setBanner("Chat is only reachable from localhost.");
        return;
      }
      if (ev.code === 1011) {
        // Server already wrote an ANSI error frame.
        return;
      }
      term.write("\r\n\x1b[90m[session ended]\x1b[0m\r\n");
    };

    // Keystrokes + mouse events → PTY, with cell-level dedup for motion.
    //
    // Ink enables `\x1b[?1003h` (any-motion tracking), which asks the
    // terminal to report every mouse-move as an SGR mouse event even with
    // no button held.  xterm.js happily emits one report per pixel of
    // mouse motion; without deduping, a casual mouse-over floods Ink with
    // hundreds of redraw-triggering reports and the UI goes laggy
    // (scrolling stutters, clicks land on stale positions by the time
    // Ink finishes processing the motion backlog).
    //
    // We keep track of the last cell we reported a motion for.  Press,
    // release, and wheel events always pass through; motion events only
    // pass through if the cell changed.  Parsing is cheap — SGR reports
    // are short literal strings.
    // eslint-disable-next-line no-control-regex -- intentional ESC byte in xterm SGR mouse report parser
    const SGR_MOUSE_RE = /^\x1b\[<(\d+);(\d+);(\d+)([Mm])$/;
    let lastMotionCell = { col: -1, row: -1 };
    let lastMotionCb = -1;
    const onDataDisposable = term.onData((data) => {
      if (ws.readyState !== WebSocket.OPEN) return;

      const m = SGR_MOUSE_RE.exec(data);
      if (m) {
        const cb = parseInt(m[1], 10);
        const col = parseInt(m[2], 10);
        const row = parseInt(m[3], 10);
        const released = m[4] === "m";
        // Motion events have bit 0x20 (32) set in the button code.
        // Wheel events have bit 0x40 (64); always forward wheel.
        const isMotion = (cb & 0x20) !== 0 && (cb & 0x40) === 0;
        const isWheel = (cb & 0x40) !== 0;
        if (isMotion && !isWheel && !released) {
          if (
            col === lastMotionCell.col &&
            row === lastMotionCell.row &&
            cb === lastMotionCb
          ) {
            return; // same cell + same button state; skip redundant report
          }
          lastMotionCell = { col, row };
          lastMotionCb = cb;
        } else {
          // Non-motion event (press, release, wheel) — reset dedup state
          // so the next motion after this always reports.
          lastMotionCell = { col: -1, row: -1 };
          lastMotionCb = -1;
        }
      }

      ws.send(data);
    });

    const onResizeDisposable = term.onResize(({ cols, rows }) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(`\x1b[RESIZE:${cols};${rows}]`);
      }
    });

    term.focus();

    return () => {
      unmounting = true;
      onDataDisposable.dispose();
      onResizeDisposable.dispose();
      if (metricsDebounce) clearTimeout(metricsDebounce);
      window.removeEventListener("resize", scheduleSyncTerminalMetrics);
      window.visualViewport?.removeEventListener(
        "resize",
        scheduleSyncTerminalMetrics,
      );
      window.visualViewport?.removeEventListener(
        "scroll",
        scheduleSyncTerminalMetrics,
      );
      ro.disconnect();
      if (hostSyncRaf) cancelAnimationFrame(hostSyncRaf);
      if (settleRaf1) cancelAnimationFrame(settleRaf1);
      if (settleRaf2) cancelAnimationFrame(settleRaf2);
      ws.close();
      wsRef.current = null;
      term.dispose();
      termRef.current = null;
      fitRef.current = null;
      if (copyResetRef.current) {
        clearTimeout(copyResetRef.current);
        copyResetRef.current = null;
      }
    };
  }, [channel]);

  // Layout:
  //   outer flex column — sits inside the dashboard's content area
  //   row split — terminal pane (flex-1) + sidebar (fixed width, lg+)
  //   terminal wrapper — rounded, dark, padded — the "terminal window"
  //   floating copy button — bottom-right corner, transparent with a
  //     subtle border; stays out of the way until hovered.  Sends
  //     `/copy\n` to Ink, which emits OSC 52 → our clipboard handler.
  //   sidebar — ChatSidebar opens its own JSON-RPC sidecar; renders
  //     model badge, tool-call list, model picker. Best-effort: if the
  //     sidecar fails to connect the terminal pane keeps working.
  //
  // `normal-case` opts out of the dashboard's global `uppercase` rule on
  // the root `<div>` in App.tsx — terminal output must preserve case.
  //
  // Mobile model/tools sheet is portaled to `document.body` so it stacks
  // above the app sidebar (`z-50`) and mobile chrome (`z-40`).  The main
  // dashboard column uses `relative z-2`, which traps `position:fixed`
  // descendants below those layers (see Toast.tsx).
  const mobileModelToolsPortal =
    narrow &&
    portalRoot &&
    createPortal(
      <>
        {mobilePanelOpen && (
          <button
            type="button"
            aria-label={t.app.closeModelTools}
            onClick={closeMobilePanel}
            className={cn(
              "fixed inset-0 z-[55]",
              "bg-black/60 backdrop-blur-sm cursor-pointer",
            )}
          />
        )}

        <div
          id="chat-side-panel"
          role="complementary"
          aria-label={modelToolsLabel}
          className={cn(
            "font-mondwest fixed top-0 right-0 z-[60] flex h-dvh max-h-dvh w-64 min-w-0 flex-col antialiased",
            "border-l border-current/20 text-midground",
            "bg-background-base/95 backdrop-blur-sm",
            "transition-transform duration-200 ease-out",
            "[background:var(--component-sidebar-background)]",
            "[clip-path:var(--component-sidebar-clip-path)]",
            "[border-image:var(--component-sidebar-border-image)]",
            mobilePanelOpen
              ? "translate-x-0"
              : "pointer-events-none translate-x-full",
          )}
        >
          <div
            className={cn(
              "flex h-14 shrink-0 items-center justify-between gap-2 border-b border-current/20 px-5",
            )}
          >
            <Typography
              className="font-bold text-[1.125rem] leading-[0.95] tracking-[0.0525rem] text-midground"
              style={{ mixBlendMode: "plus-lighter" }}
            >
              {t.app.modelToolsSheetTitle}
              <br />
              {t.app.modelToolsSheetSubtitle}
            </Typography>

            <button
              type="button"
              onClick={closeMobilePanel}
              aria-label={t.app.closeModelTools}
              className={cn(
                "inline-flex h-7 w-7 items-center justify-center",
                "text-midground/70 hover:text-midground transition-colors cursor-pointer",
                "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-midground",
              )}
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          <div
            className={cn(
              "min-h-0 flex-1 overflow-y-auto overflow-x-hidden",
              "border-t border-current/10",
            )}
          >
            <ChatSidebar channel={channel} />
          </div>
        </div>
      </>,
      portalRoot,
    );

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2 normal-case">
      <PluginSlot name="chat:top" />
      {mobileModelToolsPortal}

      {banner && (
        <div className="border border-warning/50 bg-warning/10 text-warning px-3 py-2 text-xs tracking-wide">
          {banner}
        </div>
      )}

      <div className="flex min-h-0 flex-1 flex-col gap-2 lg:flex-row lg:gap-3">
        <div
          className={cn(
            "relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden rounded-lg",
            "p-2 sm:p-3",
          )}
          style={{
            backgroundColor: TERMINAL_THEME.background,
            boxShadow: "0 8px 32px rgba(0, 0, 0, 0.4)",
          }}
        >
          <div
            ref={hostRef}
            className="hermes-chat-xterm-host min-h-0 min-w-0 flex-1"
          />

          <button
            type="button"
            onClick={handleCopyLast}
            title="Copy last assistant response as raw markdown"
            aria-label="Copy last assistant response"
            className={cn(
              "absolute z-10 flex items-center gap-1.5",
              "rounded border border-current/30",
              "bg-black/20 backdrop-blur-sm",
              "opacity-60 hover:opacity-100 hover:border-current/60",
              "transition-opacity duration-150",
              "focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-current",
              "cursor-pointer",
              "bottom-2 right-2 px-2 py-1 text-[0.65rem] sm:bottom-3 sm:right-3 sm:px-2.5 sm:py-1.5 sm:text-xs",
              "lg:bottom-4 lg:right-4",
            )}
            style={{ color: TERMINAL_THEME.foreground }}
          >
            <Copy className="h-3 w-3 shrink-0" />
            <span className="hidden min-[400px]:inline tracking-wide">
              {copyState === "copied" ? "copied" : "copy last response"}
            </span>
          </button>
        </div>

        {!narrow && (
          <div
            id="chat-side-panel"
            role="complementary"
            aria-label={modelToolsLabel}
            className="flex min-h-0 shrink-0 flex-col lg:h-full lg:w-80"
          >
            <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden">
              <ChatSidebar channel={channel} />
            </div>
          </div>
        )}
      </div>
      <PluginSlot name="chat:bottom" />
    </div>
  );
}

declare global {
  interface Window {
    __HERMES_SESSION_TOKEN__?: string;
  }
}
