import { FitAddon } from "@xterm/addon-fit";
import { Unicode11Addon } from "@xterm/addon-unicode11";
import { WebLinksAddon } from "@xterm/addon-web-links";
import { WebglAddon } from "@xterm/addon-webgl";
import { Terminal } from "@xterm/xterm";
import "@xterm/xterm/css/xterm.css";
import { useEffect, useMemo, useRef, useState } from "react";

import { HERMES_BASE_PATH, buildWsAuthParam } from "@/lib/api";
import { cn } from "@/lib/utils";

const TERMINAL_THEME = {
  background: "#0d2626",
  foreground: "#f0e6d2",
  cursor: "#f0e6d2",
  cursorAccent: "#0d2626",
  selectionBackground: "#f0e6d244",
};

function generateChannelId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `pty-${Math.random().toString(36).slice(2)}-${Date.now().toString(36)}`;
}

function tierWidthPx(host: HTMLElement | null): number {
  if (typeof window === "undefined") return 1280;
  const fromHost = host?.clientWidth ?? 0;
  if (fromHost > 2) return Math.round(fromHost);
  const doc = document.documentElement?.clientWidth ?? 0;
  return Math.max(1, Math.round(doc || window.innerWidth || 1280));
}

function fontSizeForWidth(width: number): number {
  if (width < 300) return 7;
  if (width < 360) return 8;
  if (width < 420) return 9;
  if (width < 520) return 10;
  if (width < 720) return 11;
  if (width < 1024) return 12;
  return 14;
}

function lineHeightForWidth(width: number): number {
  return width < 1024 ? 1.02 : 1.15;
}

export type PtyTerminalPaneProps = {
  tmuxTarget?: string;
  resume?: string | null;
  className?: string;
  title?: string;
  autoFocus?: boolean;
};

export function PtyTerminalPane({ tmuxTarget, resume, className, title, autoFocus = false }: PtyTerminalPaneProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const termRef = useRef<Terminal | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [banner, setBanner] = useState<string | null>(() =>
    typeof window !== "undefined" && !window.__HERMES_SESSION_TOKEN__ && !window.__HERMES_AUTH_REQUIRED__
      ? "Session token unavailable. Open this page through `hermes dashboard`, not directly."
      : null,
  );
  const channel = useMemo(() => generateChannelId(), [tmuxTarget, resume]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const token = window.__HERMES_SESSION_TOKEN__;
    const gated = !!window.__HERMES_AUTH_REQUIRED__;
    if (!token && !gated) return;

    const width = tierWidthPx(host);
    const term = new Terminal({
      allowProposedApi: true,
      cursorBlink: true,
      fontFamily: "'JetBrains Mono', 'Cascadia Mono', 'Fira Code', 'MesloLGS NF', 'Source Code Pro', Menlo, Consolas, 'DejaVu Sans Mono', monospace",
      fontSize: fontSizeForWidth(width),
      lineHeight: lineHeightForWidth(width),
      fontWeight: "400",
      fontWeightBold: "700",
      macOptionIsMeta: true,
      macOptionClickForcesSelection: true,
      rightClickSelectsWord: true,
      scrollback: 5000,
      theme: TERMINAL_THEME,
    });
    termRef.current = term;

    term.attachCustomKeyEventHandler((ev) => {
      if (ev.type !== "keydown") return true;
      const isMac = typeof navigator !== "undefined" && /Mac/i.test(navigator.platform);
      const copyModifier = isMac ? ev.metaKey : ev.ctrlKey && ev.shiftKey;
      const pasteModifier = isMac ? ev.metaKey : ev.ctrlKey && ev.shiftKey;
      if (copyModifier && ev.key.toLowerCase() === "c") {
        const sel = term.getSelection();
        if (sel) {
          navigator.clipboard.writeText(sel).catch(() => {});
          term.clearSelection();
          ev.preventDefault();
          return false;
        }
      }
      if (pasteModifier && ev.key.toLowerCase() === "v") {
        navigator.clipboard.readText().then((text) => { if (text) term.paste(text); }).catch(() => {});
        ev.preventDefault();
        return false;
      }
      return true;
    });

    const fit = new FitAddon();
    term.loadAddon(fit);
    const unicode11 = new Unicode11Addon();
    term.loadAddon(unicode11);
    term.unicode.activeVersion = "11";
    term.loadAddon(new WebLinksAddon());
    term.open(host);
    if (tierWidthPx(host) >= 768) {
      try {
        const webgl = new WebglAddon();
        webgl.onContextLoss(() => webgl.dispose());
        term.loadAddon(webgl);
      } catch {
        // Canvas renderer fallback.
      }
    }

    let raf = 0;
    const sync = () => {
      if (!host.isConnected || host.clientWidth <= 0 || host.clientHeight <= 0) return;
      const nextWidth = tierWidthPx(host);
      term.options.fontSize = fontSizeForWidth(nextWidth);
      term.options.lineHeight = lineHeightForWidth(nextWidth);
      try { fit.fit(); } catch { return; }
      if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send(`\x1b[RESIZE:${term.cols};${term.rows}]`);
    };
    const scheduleSync = () => {
      if (raf) cancelAnimationFrame(raf);
      raf = requestAnimationFrame(sync);
    };
    const ro = new ResizeObserver(scheduleSync);
    ro.observe(host);
    scheduleSync();
    window.addEventListener("resize", scheduleSync);

    let unmounting = false;
    let onDataDisposable: { dispose(): void } | null = null;
    let onResizeDisposable: { dispose(): void } | null = null;
    void (async () => {
      const authParam = await buildWsAuthParam();
      if (unmounting) return;
      const qs = new URLSearchParams({ [authParam[0]]: authParam[1], channel });
      if (resume) qs.set("resume", resume);
      if (tmuxTarget) {
        qs.set("pty_mode", "tmux");
        qs.set("tmux_session", tmuxTarget);
      }
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(`${proto}//${window.location.host}${HERMES_BASE_PATH}/api/pty?${qs.toString()}`);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;
      ws.onopen = () => {
        setBanner(null);
        ws.send(`\x1b[RESIZE:${term.cols};${term.rows}]`);
      };
      ws.onmessage = (ev) => {
        if (typeof ev.data === "string") term.write(ev.data);
        else term.write(new Uint8Array(ev.data as ArrayBuffer));
      };
      ws.onclose = (ev) => {
        wsRef.current = null;
        if (unmounting || ev.code === 1011) return;
        if (ev.code === 4401) setBanner("Auth failed. Reload the page to refresh the session token.");
        else if (ev.code === 4403) setBanner("Terminal is only reachable from localhost.");
        else term.write("\r\n\x1b[90m[session ended]\x1b[0m\r\n");
      };
      // eslint-disable-next-line no-control-regex -- xterm SGR mouse report parser
      const SGR_MOUSE_RE = /^\x1b\[<(\d+);(\d+);(\d+)([Mm])$/;
      onDataDisposable = term.onData((data) => {
        if (ws.readyState !== WebSocket.OPEN || SGR_MOUSE_RE.test(data)) return;
        ws.send(data);
      });
      onResizeDisposable = term.onResize(({ cols, rows }) => {
        if (ws.readyState === WebSocket.OPEN) ws.send(`\x1b[RESIZE:${cols};${rows}]`);
      });
    })();

    if (autoFocus) term.focus();

    return () => {
      unmounting = true;
      onDataDisposable?.dispose();
      onResizeDisposable?.dispose();
      ro.disconnect();
      window.removeEventListener("resize", scheduleSync);
      if (raf) cancelAnimationFrame(raf);
      wsRef.current?.close();
      wsRef.current = null;
      term.dispose();
      termRef.current = null;
    };
  }, [channel, resume, tmuxTarget, autoFocus]);

  return (
    <div className={cn("relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden rounded-lg p-2", className)} style={{ backgroundColor: TERMINAL_THEME.background }} title={title}>
      {banner ? <div className="border border-warning/50 bg-warning/10 text-warning px-3 py-2 text-xs tracking-wide">{banner}</div> : null}
      <div ref={hostRef} className="hermes-chat-xterm-host min-h-0 min-w-0 flex-1" />
    </div>
  );
}

declare global {
  interface Window {
    __HERMES_SESSION_TOKEN__?: string;
    __HERMES_AUTH_REQUIRED__?: boolean;
  }
}
