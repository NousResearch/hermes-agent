export type PtyConnectionState =
  | "connecting"
  | "open"
  | "reconnecting"
  | "closed"
  | "ended";

export const PTY_RECONNECT_INPUT_MESSAGE =
  "Chat is reconnecting. Input will resume when connected.";

export interface PtyResumeReconnectInput {
  isActive: boolean;
  visibilityState?: DocumentVisibilityState;
  online: boolean;
  socketReadyState?: number | null;
  ptyState: PtyConnectionState;
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
  if (
    socketReadyState === WS_CONNECTING &&
    ptyState !== "reconnecting" &&
    ptyState !== "closed"
  ) {
    return false;
  }
  return (
    socketReadyState === null ||
    socketReadyState === undefined ||
    socketReadyState === WS_CONNECTING ||
    socketReadyState === WS_CLOSING ||
    socketReadyState === WS_CLOSED ||
    ptyState === "reconnecting" ||
    ptyState === "closed"
  );
}

export function shouldBlockPtyInput(ptyState: PtyConnectionState): boolean {
  return ptyState !== "open";
}
