import type { RefObject } from "react";
import type { Terminal } from "@xterm/xterm";

/**
 * Sends `/copy` over the open PTY WebSocket, waits long enough for Ink's
 * tokenizer to emit a keypress event per character (not coalesce them into
 * a paste), then sends Return as its own event. The 100ms timing is
 * empirical — safely past Node's default stdin coalescing window and well
 * inside UI responsiveness. Flips `copyState` to "copied" for 1.5s and
 * refocuses the terminal.
 */
export function sendCopyLastCommand(opts: {
  wsRef: RefObject<WebSocket | null>;
  termRef: RefObject<Terminal | null>;
  copyResetRef: RefObject<ReturnType<typeof setTimeout> | null>;
  onCopied: () => void;
  onCopyReset: () => void;
}): void {
  const ws = opts.wsRef.current;
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send("/copy");
  setTimeout(() => {
    const s = opts.wsRef.current;
    if (s && s.readyState === WebSocket.OPEN) s.send("\r");
  }, 100);
  opts.onCopied();
  if (opts.copyResetRef.current) clearTimeout(opts.copyResetRef.current);
  opts.copyResetRef.current = setTimeout(() => opts.onCopyReset(), 1500);
  opts.termRef.current?.focus();
}
