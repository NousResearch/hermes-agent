import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { FitAddon } from "@xterm/addon-fit";
import { Unicode11Addon } from "@xterm/addon-unicode11";
import { WebLinksAddon } from "@xterm/addon-web-links";
import { Terminal as XtermTerminal } from "@xterm/xterm";
import "@xterm/xterm/css/xterm.css";
import { Terminal, X } from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { useModalBehavior } from "@/hooks/useModalBehavior";
import { useProfileScope } from "@/contexts/useProfileScope";
import { api } from "@/lib/api";
import { cn, themedBody } from "@/lib/utils";
import { useTheme } from "@/themes";

type ConsoleFrame =
  | {
      type: "ready";
      context?: string;
      profile?: string;
      prompt?: string;
    }
  | {
      type: "output";
      data?: string;
      stream?: string;
    }
  | {
      type: "error";
      message?: string;
    }
  | {
      type: "confirm_required";
      command?: string;
      message?: string;
      prompt?: string;
    }
  | {
      type: "complete";
      status?: string;
      prompt?: string;
    }
  | {
      type: "clear";
    }
  | {
      type: "pong";
    };

type ConnectionState = "connecting" | "ready" | "running" | "closed" | "error";

interface HermesConsoleModalProps {
  open: boolean;
  onClose: () => void;
}

function buildTerminalTheme(background: string, foreground: string) {
  return {
    background,
    foreground,
    cursor: foreground,
    cursorAccent: background,
    selectionBackground: "rgba(255, 255, 255, 0.25)",
    black: "#000000",
    red: "#ff5f67",
    green: "#5fffb0",
    yellow: "#ffd166",
    blue: "#7aa2ff",
    magenta: "#d597ff",
    cyan: "#58e6ff",
    white: foreground,
    brightBlack: "#666666",
    brightRed: "#ff8b90",
    brightGreen: "#8dffc8",
    brightYellow: "#ffe08a",
    brightBlue: "#9dbaff",
    brightMagenta: "#e4b7ff",
    brightCyan: "#8ef0ff",
    brightWhite: "#ffffff",
  };
}

function normalizeTerminalText(text: string): string {
  return text.replace(/\r?\n/g, "\r\n");
}

function writeLine(term: XtermTerminal, text = ""): void {
  term.write(`${normalizeTerminalText(text)}\r\n`);
}

function writeBlock(term: XtermTerminal, text: string): void {
  const normalized = normalizeTerminalText(text);
  term.write(normalized.endsWith("\r\n") ? normalized : `${normalized}\r\n`);
}

function isPrintable(data: string): boolean {
  return data >= " " || data === "\t";
}

export function HermesConsoleModal({ open, onClose }: HermesConsoleModalProps) {
  const modalRef = useModalBehavior({ open, onClose });
  const hostRef = useRef<HTMLDivElement | null>(null);
  const termRef = useRef<XtermTerminal | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const lineRef = useRef("");
  const promptRef = useRef("hermes> ");
  const inputPromptRef = useRef("hermes> ");
  const historyRef = useRef<string[]>([]);
  const historyIndexRef = useRef<number | null>(null);
  const activeCommandRef = useRef(false);
  const pendingCommandRef = useRef<string | null>(null);
  const hasReadyFrameRef = useRef(false);
  const [connectionState, setConnectionState] =
    useState<ConnectionState>("connecting");
  const [consoleContext, setConsoleContext] = useState("pending");
  const [consoleProfile, setConsoleProfile] = useState("current");
  const { profile } = useProfileScope();
  const { theme } = useTheme();

  const redrawInput = useCallback((line = lineRef.current) => {
    const term = termRef.current;
    if (!term) return;
    lineRef.current = line;
    term.write(`\r\x1b[2K${inputPromptRef.current}${line}`);
  }, []);

  const showPrompt = useCallback(() => {
    const term = termRef.current;
    if (!term) return;
    lineRef.current = "";
    historyIndexRef.current = null;
    inputPromptRef.current = promptRef.current;
    term.write(inputPromptRef.current);
  }, []);

  const sendFrame = useCallback((payload: Record<string, unknown>) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return false;
    ws.send(JSON.stringify(payload));
    return true;
  }, []);

  const cancelCommand = useCallback(() => {
    pendingCommandRef.current = null;
    activeCommandRef.current = false;
    sendFrame({ type: "cancel" });
  }, [sendFrame]);

  const submitLine = useCallback(
    (rawLine: string) => {
      const term = termRef.current;
      if (!term) return;
      const line = rawLine.trim();
      term.write("\r\n");
      lineRef.current = "";
      historyIndexRef.current = null;

      const pending = pendingCommandRef.current;
      if (pending) {
        const answer = line.toLowerCase();
        if (answer === "y" || answer === "yes") {
          pendingCommandRef.current = null;
          activeCommandRef.current = true;
          setConnectionState("running");
          sendFrame({ type: "confirm", command: pending });
          return;
        }
        cancelCommand();
        return;
      }

      if (!line) {
        showPrompt();
        return;
      }

      historyRef.current = [...historyRef.current, line].slice(-200);
      activeCommandRef.current = true;
      setConnectionState("running");
      if (!sendFrame({ type: "input", line })) {
        activeCommandRef.current = false;
        writeLine(term, "\x1b[31mConsole is not connected.\x1b[0m");
        showPrompt();
      }
    },
    [cancelCommand, sendFrame, showPrompt],
  );

  const recallHistory = useCallback(
    (direction: -1 | 1) => {
      const history = historyRef.current;
      if (!history.length) return;
      const current = historyIndexRef.current;
      if (current === null) {
        if (direction > 0) return;
        historyIndexRef.current = history.length - 1;
      } else {
        const next = current + direction;
        if (next < 0) historyIndexRef.current = 0;
        else if (next >= history.length) {
          historyIndexRef.current = null;
          redrawInput("");
          return;
        } else {
          historyIndexRef.current = next;
        }
      }
      const idx = historyIndexRef.current;
      redrawInput(idx === null ? "" : history[idx] ?? "");
    },
    [redrawInput],
  );

  const handleInputData = useCallback(
    (data: string) => {
      const term = termRef.current;
      if (!term) return;

      if (data === "\x1b[A") {
        recallHistory(-1);
        return;
      }
      if (data === "\x1b[B") {
        recallHistory(1);
        return;
      }

      for (const ch of data) {
        if (ch === "\u0003") {
          term.write("^C\r\n");
          if (activeCommandRef.current || pendingCommandRef.current) {
            cancelCommand();
          } else {
            showPrompt();
          }
          continue;
        }
        if (ch === "\u000c") {
          term.clear();
          showPrompt();
          continue;
        }
        if (activeCommandRef.current) {
          term.write("\x07");
          continue;
        }
        if (ch === "\r" || ch === "\n") {
          submitLine(lineRef.current);
          continue;
        }
        if (ch === "\u007f" || ch === "\b") {
          if (lineRef.current.length > 0) {
            lineRef.current = lineRef.current.slice(0, -1);
            term.write("\b \b");
          }
          continue;
        }
        if (ch === "\x1b") {
          continue;
        }
        if (isPrintable(ch)) {
          lineRef.current += ch;
          term.write(ch);
        }
      }
    },
    [cancelCommand, recallHistory, showPrompt, submitLine],
  );

  const handleFrame = useCallback(
    (frame: ConsoleFrame) => {
      const term = termRef.current;
      if (!term) return;

      if (frame.type === "ready") {
        const nextPrompt = frame.prompt || "hermes> ";
        promptRef.current = nextPrompt;
        inputPromptRef.current = nextPrompt;
        hasReadyFrameRef.current = true;
        setConsoleContext(frame.context || "local");
        setConsoleProfile(frame.profile || "current");
        activeCommandRef.current = false;
        setConnectionState("ready");
        term.clear();
        showPrompt();
        return;
      }

      if (frame.type === "output") {
        if (frame.data) writeBlock(term, frame.data);
        return;
      }

      if (frame.type === "error") {
        writeLine(term, `\x1b[31m${frame.message || "Command failed."}\x1b[0m`);
        return;
      }

      if (frame.type === "confirm_required") {
        pendingCommandRef.current = frame.command || "";
        activeCommandRef.current = false;
        setConnectionState("ready");
        if (frame.message) {
          writeLine(term, `\x1b[33m${frame.message}\x1b[0m`);
        }
        inputPromptRef.current = "Confirm? [y/N] ";
        lineRef.current = "";
        term.write(inputPromptRef.current);
        return;
      }

      if (frame.type === "complete") {
        activeCommandRef.current = false;
        if (frame.prompt) promptRef.current = frame.prompt;
        if (frame.status === "confirm_required") return;
        if (frame.status === "exit") {
          setConnectionState("closed");
          wsRef.current?.close();
          return;
        }
        if (frame.status === "timeout") {
          writeLine(term, "\x1b[31mCommand timed out.\x1b[0m");
        }
        if (frame.status === "cancelled") {
          writeLine(term, "\x1b[33mCancelled.\x1b[0m");
        }
        pendingCommandRef.current = null;
        setConnectionState("ready");
        showPrompt();
        return;
      }

      if (frame.type === "clear") {
        term.clear();
        showPrompt();
      }
    },
    [showPrompt],
  );

  useEffect(() => {
    if (!open) return;
    const host = hostRef.current;
    if (!host) return;

    let cancelled = false;
    let resizeFrame = 0;
    const term = new XtermTerminal({
      allowProposedApi: true,
      cursorBlink: true,
      fontFamily:
        "'JetBrains Mono', 'Cascadia Mono', 'Fira Code', 'MesloLGS NF', 'Source Code Pro', Menlo, Consolas, 'DejaVu Sans Mono', monospace",
      fontSize: 13,
      lineHeight: 1.25,
      letterSpacing: 0,
      macOptionIsMeta: true,
      scrollback: 3000,
      theme: buildTerminalTheme(
        theme.terminalBackground ?? "#000000",
        theme.terminalForeground ?? "#f0e6d2",
      ),
    });
    termRef.current = term;

    const fit = new FitAddon();
    term.loadAddon(fit);
    const unicode11 = new Unicode11Addon();
    term.loadAddon(unicode11);
    term.unicode.activeVersion = "11";
    term.loadAddon(new WebLinksAddon());
    term.open(host);
    term.focus();

    const fitTerminal = () => {
      if (!host.isConnected || host.clientWidth <= 0 || host.clientHeight <= 0) {
        return;
      }
      try {
        fit.fit();
      } catch {
        /* fit can fail while the modal is closing */
      }
    };
    const scheduleFit = () => {
      if (resizeFrame) return;
      resizeFrame = requestAnimationFrame(() => {
        resizeFrame = 0;
        fitTerminal();
      });
    };
    const ro = new ResizeObserver(scheduleFit);
    ro.observe(host);
    scheduleFit();

    const dataDisposable = term.onData(handleInputData);
    setConnectionState("connecting");
    setConsoleContext("pending");
    setConsoleProfile(profile || "current");
    hasReadyFrameRef.current = false;
    writeLine(term, "\x1b[2mConnecting to IX Agency Console...\x1b[0m");

    void (async () => {
      try {
        const params = profile ? { profile } : undefined;
        const url = await api.buildWsUrl("/api/console", params);
        if (cancelled) return;
        const ws = new WebSocket(url);
        wsRef.current = ws;

        ws.onopen = () => {
          setConnectionState("connecting");
        };

        ws.onmessage = (ev) => {
          try {
            const frame = JSON.parse(String(ev.data)) as ConsoleFrame;
            handleFrame(frame);
          } catch {
            writeLine(term, "\x1b[31mMalformed console frame.\x1b[0m");
          }
        };

        ws.onerror = () => {
          setConnectionState("error");
          writeLine(term, "\x1b[31mConsole websocket error.\x1b[0m");
        };

        ws.onclose = (ev) => {
          wsRef.current = null;
          activeCommandRef.current = false;
          pendingCommandRef.current = null;
          if (cancelled) return;
          setConnectionState(ev.code === 1000 ? "closed" : "error");
          const reason = ev.reason ? ` ${ev.reason}` : "";
          const message =
            ev.code === 1006 && !hasReadyFrameRef.current
              ? "Console connection failed before the server handshake. Check that this dashboard is connected to a backend with /api/console."
              : `Console closed (${ev.code}).${reason}`;
          writeLine(term, `\x1b[31m${message}\x1b[0m`);
        };
      } catch (err) {
        if (cancelled) return;
        setConnectionState("error");
        writeLine(term, `\x1b[31mConsole unavailable: ${err}\x1b[0m`);
      }
    })();

    return () => {
      cancelled = true;
      dataDisposable.dispose();
      ro.disconnect();
      if (resizeFrame) cancelAnimationFrame(resizeFrame);
      wsRef.current?.close();
      wsRef.current = null;
      term.dispose();
      termRef.current = null;
      lineRef.current = "";
      pendingCommandRef.current = null;
      activeCommandRef.current = false;
      hasReadyFrameRef.current = false;
    };
  }, [handleFrame, handleInputData, open, profile, theme]);

  useEffect(() => {
    if (!open) return;
    const term = termRef.current;
    if (!term) return;
    term.options.theme = buildTerminalTheme(
      theme.terminalBackground ?? "#000000",
      theme.terminalForeground ?? "#f0e6d2",
    );
  }, [open, theme]);

  if (!open) return null;

  const statusTone =
    connectionState === "ready"
      ? "success"
      : connectionState === "running"
        ? "warning"
        : connectionState === "connecting"
          ? "secondary"
          : "destructive";

  return createPortal(
    <div
      ref={modalRef}
      className="fixed inset-0 z-[100] flex items-center justify-center bg-background/85 p-3 sm:p-4"
      onClick={(event) => event.target === event.currentTarget && onClose()}
      role="dialog"
      aria-modal="true"
      aria-labelledby="hermes-console-title"
    >
      <div
        className={cn(
          themedBody,
          "relative flex h-[min(82dvh,760px)] w-full max-w-5xl flex-col border border-border bg-card shadow-2xl",
        )}
      >
        <header className="flex min-h-14 items-center gap-3 border-b border-border px-4 py-3">
          <div className="flex h-9 w-9 items-center justify-center border border-border bg-background/60 text-primary">
            <Terminal className="h-4 w-4" />
          </div>
          <div className="min-w-0 flex-1">
            <h2
              id="hermes-console-title"
              className="font-mondwest text-display text-base tracking-wider"
            >
              IX Agency Console
            </h2>
            <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
              <Badge tone={statusTone}>{connectionState}</Badge>
              <span className="font-mono">{consoleContext}</span>
              <span className="font-mono">{consoleProfile}</span>
            </div>
          </div>
          <Button
            ghost
            size="icon"
            onClick={onClose}
            className="text-muted-foreground hover:text-foreground"
            aria-label="Close console"
          >
            <X />
          </Button>
        </header>
        <div className="min-h-0 flex-1 bg-black">
          <div
            ref={hostRef}
            className="h-full min-h-0 w-full overflow-hidden p-2 [&_.xterm]:h-full [&_.xterm-viewport]:!bg-transparent"
          />
        </div>
      </div>
    </div>,
    document.body,
  );
}
