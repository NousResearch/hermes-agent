export type PtyConnectionState =
  | "connecting"
  | "open"
  | "reconnecting"
  | "closed"
  | "ended";

export const PTY_RECONNECT_INPUT_MESSAGE =
  "Chat is reconnecting. Input will resume when connected.";

// Minimum gap (ms) between page-resume-triggered reconnect attempts, so a
// burst of visibilitychange/pageshow/focus/online events on tab-return
// collapses into a single reconnect.
export const PTY_RESUME_RECONNECT_THROTTLE_MS = 1000;

// If a socket sits in WS_CONNECTING past this budget it is treated as wedged
// (e.g. a half-open mobile socket after a radio handoff — the NS-591 case)
// and force-closed so `onclose` → scheduleReconnect can recover it.
export const PTY_CONNECTING_TIMEOUT_MS = 8000;

// The same budget for the phase *before* the socket exists: in gated mode a
// connect first awaits a fresh single-use ticket. That request produces no
// WebSocket, so a rejection or a hang is invisible to both `onclose` and the
// CONNECTING timer above — the tab would sit on "connecting" forever with no
// retry. Bound it so the failure routes into the ordinary backoff instead.
export const PTY_TICKET_TIMEOUT_MS = 8000;

// How long after a resumed socket opens we keep suppressing ANSI erase codes
// (`ESC[K` / `ESC[X`) from the PTY stream. Ink's two-pass virtual scroll emits
// them while replaying a long session; past that replay they are legitimate
// in-place redraws (spinners, progress bars, status lines) and must reach
// xterm or stale glyphs are left on screen. The replay of a 200+ message
// session takes ~10-20s, so this is deliberately generous — over-running the
// window only costs a few stale cells on a buffer that is about to be
// repainted, while under-running it re-opens the blank-viewport bug.
export const PTY_RESUME_SANITIZE_WINDOW_MS = 30000;

export interface PtyResumeReconnectInput {
  isActive: boolean;
  visibilityState?: DocumentVisibilityState;
  online: boolean;
  socketReadyState?: number | null;
  ptyState: PtyConnectionState;
  connectInFlight?: boolean;
}

const WS_CONNECTING = 0;
const WS_OPEN = 1;
const WS_CLOSING = 2;
const WS_CLOSED = 3;

export function shouldReconnectPtyOnPageResume({
  isActive,
  visibilityState,
  online,
  socketReadyState,
  ptyState,
  connectInFlight,
}: PtyResumeReconnectInput): boolean {
  if (!isActive || !online || visibilityState === "hidden") {
    return false;
  }
  if (ptyState === "ended") {
    return false;
  }
  if (socketReadyState === WS_OPEN) {
    return false;
  }
  // A connect is mid-flight (the async socket-open IIFE is awaiting its
  // ticket URL and hasn't assigned wsRef yet, or the socket is still
  // CONNECTING on a non-stuck attempt). Don't fire a redundant reconnect
  // into that window unless the tab already believes it is reconnecting or
  // closed and needs a fresh attempt.
  if (
    (connectInFlight || socketReadyState === WS_CONNECTING) &&
    ptyState !== "reconnecting" &&
    ptyState !== "closed"
  ) {
    return false;
  }
  return (
    socketReadyState === null ||
    socketReadyState === undefined ||
    socketReadyState === WS_CLOSING ||
    socketReadyState === WS_CLOSED ||
    ptyState === "reconnecting" ||
    ptyState === "closed"
  );
}

export function shouldBlockPtyInput(ptyState: PtyConnectionState): boolean {
  return ptyState !== "open";
}
