import { shouldBlockPtyInput, type PtyConnectionState } from "./pty-reconnect";

const WS_OPEN = 1;

export type PtyKeyboardShortcut =
  | "copy"
  | "delete-word-backward"
  | "delete-word-forward"
  | "insert-newline"
  | "pass";

export function resolvePtyKeyboardShortcut(
  ev: Pick<KeyboardEvent, "altKey" | "ctrlKey" | "key" | "metaKey" | "shiftKey">,
  isMac: boolean,
  hasTerminalSelection: boolean,
): PtyKeyboardShortcut {
  const key = ev.key.toLowerCase();
  const copyPressed = isMac
    ? ev.metaKey && !ev.ctrlKey
    : ev.ctrlKey && !ev.altKey && !ev.metaKey;

  if (copyPressed && key === "c" && hasTerminalSelection) {
    return "copy";
  }

  if (
    ev.ctrlKey &&
    !ev.shiftKey &&
    !ev.altKey &&
    !ev.metaKey &&
    ev.key === "Backspace"
  ) {
    return "delete-word-backward";
  }

  if (
    ev.ctrlKey &&
    !ev.shiftKey &&
    !ev.altKey &&
    !ev.metaKey &&
    ev.key === "Delete"
  ) {
    return "delete-word-forward";
  }

  // Modified Returns that the browser collapses to a bare CR/LF byte. xterm.js
  // encodes Shift+Enter and Ctrl+Enter as plain \r and Ctrl+J as \n, so the
  // TUI cannot see the modifier and submits instead of inserting a newline
  // (#139337). Alt+Enter keeps xterm's native ESC \r (meta+return → newline).
  if (ev.key === "Enter" && (ev.shiftKey || ev.ctrlKey) && !ev.altKey && !ev.metaKey) {
    return "insert-newline";
  }

  if (ev.key === "j" && ev.ctrlKey && !ev.shiftKey && !ev.altKey && !ev.metaKey) {
    return "insert-newline";
  }

  return "pass";
}

// Guarded sender for shortcut escape sequences. Applies the same gate as the
// term.onData path (socket OPEN + shouldBlockPtyInput) so a reconnecting or
// closed session never receives shortcut bytes.
export function sendPtyShortcutSequence(
  ws: Pick<WebSocket, "readyState" | "send"> | null,
  ptyState: PtyConnectionState,
  sequence: string,
): boolean {
  if (!ws || ws.readyState !== WS_OPEN || shouldBlockPtyInput(ptyState)) {
    return false;
  }

  ws.send(sequence);

  return true;
}
