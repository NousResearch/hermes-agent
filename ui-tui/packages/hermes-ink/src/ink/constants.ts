// Shared frame interval for render throttling and animations (~60fps).
export const FRAME_INTERVAL_MS = 16

// Keep clock-driven animations at full speed when terminal focus changes.
// We still pause entirely when there are no keepAlive subscribers.
export const BLURRED_FRAME_INTERVAL_MS = FRAME_INTERVAL_MS

// Issue #31486 (stdout-backpressure strand): when the previous frame's
// stdout.write has NOT drained yet (terminal parser overwhelmed by a wide
// CR+LF burst — CJK + ANSI tool output on a high-context session), piling
// another write on the backed-up pipe both wastes the frame and keeps the
// macrotask queue churning, starving the stdin 'readable' callback. We
// instead COALESCE: skip the frame and retry on the drain tick. This ceiling
// caps how many consecutive frames we'll coalesce before forcing a write
// through, so a terminal whose drain callback never fires (e.g. EIO on
// flush) can't wedge the renderer permanently — it self-heals once the pipe
// recovers. ~10 frames at the drain-tick cadence is a few hundred ms of
// breathing room, well under any human-perceptible render stall.
export const MAX_COALESCED_BACKPRESSURE_FRAMES = 10
