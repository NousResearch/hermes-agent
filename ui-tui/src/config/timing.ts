export const STREAM_BATCH_MS = 16
export const STREAM_IDLE_BATCH_MS = 16
export const STREAM_SCROLL_BATCH_MS = 96
export const STREAM_TYPING_BATCH_MS = 80
export const TYPING_IDLE_MS = 250
export const REASONING_PULSE_MS = 700

// A drag-resize fires a burst of SIGWINCH events (one per pixel step in some
// hosts). Each distinct terminal width remounts the visible transcript rows so
// yoga re-measures off live geometry, so reflowing on every tick stutters the
// drag. Coalesce the burst to at most one reflow per this window (~30fps):
// responsive enough to track the drag, cheap enough to stay smooth, and the
// trailing edge always lands the final width so the settled layout is exact.
export const RESIZE_COALESCE_MS = 32

// Two Esc presses within this window discard the draft (Claude Code /
// Gemini CLI parity). Long enough for a deliberate double-tap, short
// enough that two unrelated Escs — dismissing a completion, then a
// selection — don't silently clear the composer.
export const DOUBLE_ESC_MS = 500

// Idle Ctrl+C must be pressed twice within this window to end the session
// (Claude Code parity). Multiplexer hosts (herdr, tmux copy bindings)
// translate a copy chord into plain Ctrl+C for the pane app, so a single
// stray press must never tear the session down.
export const DOUBLE_CTRL_C_EXIT_MS = 2000
