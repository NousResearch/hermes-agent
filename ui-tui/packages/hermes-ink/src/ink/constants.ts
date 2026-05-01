// Shared frame interval for render throttling and animations (~60fps).
export const FRAME_INTERVAL_MS = 16

// Keep clock-driven animations at full speed when terminal focus changes.
// We still pause entirely when there are no keepAlive subscribers.
export const BLURRED_FRAME_INTERVAL_MS = FRAME_INTERVAL_MS
