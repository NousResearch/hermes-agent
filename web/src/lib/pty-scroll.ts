/**
 * Dashboard chat resume-scroll helpers.
 *
 * When a chat session is resumed (`/chat?resume=<id>`) the PTY backend replays
 * the entire scrollback over the WebSocket the instant it opens. xterm.js writes
 * those bytes into its buffer but leaves the viewport wherever it was (e.g. the
 * top of a fresh terminal), so the transcript looks truncated until something
 * else forces a re-render. See #59591.
 *
 * The fix pins the viewport to the bottom *as each replayed chunk commits* (via
 * xterm's `write` callback) instead of guessing with a fixed double-rAF delay,
 * and releases that pin the moment the user scrolls up so their manual review of
 * the backlog is never yanked back down.
 *
 * These two decisions are pulled out here as pure functions so they can be
 * unit-tested without a live terminal; `ChatPage` wires them to the real xterm
 * instance (see `term.onScroll` and `ws.onmessage`).
 */

/** The subset of xterm's active `IBuffer` these helpers need. */
export interface TerminalViewportPosition {
	/** Row index of the top of the current viewport. */
	viewportY: number;
	/** Row index of the viewport top when scrolled fully to the bottom. */
	baseY: number;
}

/**
 * True when the viewport is scrolled to (or past) the bottom, i.e. the latest
 * output is on screen. xterm reports `viewportY === baseY` at the bottom; the
 * `>=` also covers the transient overshoot while scrollback rows are trimmed.
 */
export function isViewportPinnedToBottom(
	buffer: TerminalViewportPosition,
): boolean {
	return buffer.viewportY >= buffer.baseY;
}

/**
 * Whether a freshly written PTY output chunk should scroll the terminal to the
 * bottom afterwards. We only auto-follow while resuming a session (the replay
 * case) and only while the user hasn't scrolled up to read the backlog. A fresh
 * (non-resume) session returns `false` so normal cursor output is never fought.
 */
export function shouldFollowPtyOutput(
	resumeParam: string | null,
	stickToBottom: boolean,
): boolean {
	return Boolean(resumeParam) && stickToBottom;
}
