import { shouldBlockPtyInput, type PtyConnectionState } from "./pty-reconnect";

const WS_OPEN = 1;

export type PtyKeyboardShortcut =
  | "copy"
  | "delete-word-backward"
  | "delete-word-forward"
  | "native-paste"
  | "pass";

// Bare Ctrl (no Shift/Alt/Meta) word-deletion chords, keyed by `ev.key`.
const BARE_CTRL_SHORTCUTS: ReadonlyMap<string, PtyKeyboardShortcut> = new Map([
  ["Backspace", "delete-word-backward"],
  ["Delete", "delete-word-forward"],
]);

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

  // macOS Cmd+V must reach the browser's native paste event, which
  // chat-image-drop.ts reads synchronously (text and images). Intercepting it
  // for navigator.clipboard.readText() needs an async clipboard permission
  // that plain-HTTPS tailnet origins deny or lose with the gesture, and the
  // paste then inserts nothing.
  if (isMac && ev.metaKey && !ev.ctrlKey && key === "v") {
    return "native-paste";
  }

  const bareCtrl = ev.ctrlKey && !ev.shiftKey && !ev.altKey && !ev.metaKey;

  return (bareCtrl && BARE_CTRL_SHORTCUTS.get(ev.key)) || "pass";
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
