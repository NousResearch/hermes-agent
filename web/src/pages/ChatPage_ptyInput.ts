/**
 * Keystrokes → PTY for the dashboard chat (ChatPage's connect effect).
 *
 * IMPORTANT:
 * The embedded web chat has occasionally surfaced stray letters/digits
 * in the input line after a turn completes. The most likely culprit is
 * browser-side terminal control traffic being forwarded back into the
 * PTY as if it were user text. SGR mouse tracking is the highest-risk
 * path here: xterm.js emits raw CSI reports (`\x1b[<...`) that look like
 * ordinary bytes to the backend.
 *
 * For the browser embed we prefer input stability over terminal-style
 * mouse reporting, so we drop SGR mouse reports entirely instead of
 * forwarding them into Hermes. Keyboard input, paste, and resize still
 * behave normally.
 *
 * Input is never sent into a PTY that is not attached. While a socket is on
 * its way (connecting/reconnecting) it is held and flushed once that socket
 * opens (#145511); once the PTY is closed or ended it is dropped.
 */
import { normalizePtyMobileInput } from "@/lib/pty-mobile-input";
import { SGR_MOUSE_RE } from "@/lib/pty-mouse-report";
import {
  enqueuePtyInput,
  PTY_QUEUED_INPUT_FLUSH_DELAY_MS,
  PTY_RECONNECT_INPUT_MESSAGE,
  type PtyConnectionState,
  shouldBlockPtyInput,
  shouldQueuePtyInput,
} from "@/lib/pty-reconnect";

type Ref<T> = { current: T };

/**
 * Input blocked while the socket is not open: whether the one-time notice was
 * shown, and the keystrokes held for the next socket. Kept in a component ref,
 * not effect state: every reconnect rebuilds the terminal effect, and held
 * keystrokes must outlive it.
 */
export interface BlockedPtyInput {
  notice: boolean;
  queue: string[];
}

export interface PtyInputRefs {
  ptyState: Ref<PtyConnectionState>;
  blocked: Ref<BlockedPtyInput>;
  inputLine: Ref<string>;
  mobileReplacementUntil: Ref<number>;
}

export function createPtyInputForwarder(
  ws: WebSocket,
  term: { write(data: string): void },
  refs: PtyInputRefs,
) {
  // Send what was held while the socket was coming up, in typing order.
  const flushHeld = () => {
    const held = refs.blocked.current.queue;
    refs.blocked.current.queue = [];
    for (const chunk of held) {
      forward(chunk, false);
    }
  };

  const forward = (data: string, useMobileReplacement = true): void => {
    // Mouse reports (scroll wheel etc.) are not typed input — swallow
    // them before the blocked-input check so scrolling a disconnected
    // terminal doesn't trip the "reconnecting" notice.
    if (SGR_MOUSE_RE.test(data)) {
      return;
    }

    if (ws.readyState !== WebSocket.OPEN || shouldBlockPtyInput(refs.ptyState.current)) {
      if (shouldQueuePtyInput(refs.ptyState.current)) {
        refs.blocked.current.queue = enqueuePtyInput(refs.blocked.current.queue, data);
      }
      if (!refs.blocked.current.notice) {
        refs.blocked.current.notice = true;
        term.write(`\r\n\x1b[33m[${PTY_RECONNECT_INPUT_MESSAGE}]\x1b[0m\r\n`);
      }
      return;
    }

    // Held input still waiting on the post-open delay goes out first, so a
    // key typed during that delay cannot overtake it.
    if (refs.blocked.current.queue.length) {
      flushHeld();
    }

    const normalized = normalizePtyMobileInput(
      data,
      refs.inputLine.current,
      useMobileReplacement && Date.now() <= refs.mobileReplacementUntil.current,
    );
    refs.inputLine.current = normalized.nextLine;
    if (normalized.normalized) {
      refs.mobileReplacementUntil.current = 0;
    }
    ws.send(normalized.data);
  };

  // From `onopen`: deliver held input after a short delay so the server has
  // attached the PTY first. A drop inside the delay leaves the queue for the
  // next socket.
  const flushAfterOpen = (isCurrentSocket: () => boolean) => {
    if (!refs.blocked.current.queue.length) {
      return;
    }
    setTimeout(() => {
      if (isCurrentSocket() && ws.readyState === WebSocket.OPEN) {
        flushHeld();
      }
    }, PTY_QUEUED_INPUT_FLUSH_DELAY_MS);
  };

  return { forward, flushAfterOpen };
}
