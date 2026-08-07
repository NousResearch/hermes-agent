import type { PtyConnectionState } from "@/lib/pty-reconnect";

/**
 * Hard cap so a wedged resume (never gets PTY payload) cannot leave the
 * wait notice up forever.
 */
export const PTY_RESUME_LOADING_MAX_MS = 30000;

export const PTY_RESUME_LOADING_MESSAGE =
  "Please wait while the conversation loads…";

export interface ResumeLoadingOverlayInput {
  hasResumeTarget: boolean;
  ptyState: PtyConnectionState;
  hydrating: boolean;
}

/**
 * Show a wait notice only while a resumed chat is still blank. Once the
 * first real PTY payload arrives the terminal has something to show, so
 * the notice hides and history can stream in underneath.
 *
 * Reconnect / ended / closed states keep their own overlays and must not
 * stack this one on top.
 */
export function shouldShowResumeLoadingOverlay({
  hasResumeTarget,
  ptyState,
  hydrating,
}: ResumeLoadingOverlayInput): boolean {
  if (!hasResumeTarget || !hydrating) {
    return false;
  }
  if (
    ptyState === "reconnecting" ||
    ptyState === "closed" ||
    ptyState === "ended"
  ) {
    return false;
  }
  return ptyState === "connecting" || ptyState === "open";
}

/** First non-empty PTY chunk means the blank window is over. */
export function shouldFinishResumeHydrationOnChunk(chunkText: string): boolean {
  return chunkText.length > 0;
}
