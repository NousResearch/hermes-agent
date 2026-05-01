export const LARGE_PASTE = { chars: 8000, lines: 80 }

export const LIVE_RENDER_MAX_CHARS = 16_000
export const LIVE_RENDER_MAX_LINES = 240

// History-render bounds for messages outside FULL_RENDER_TAIL. Each rendered
// line ≈ 1 Yoga/Text node + inline spans, so this is the dominant lever on
// cold-mount cost during PageUp catch-up. 16 lines × 25 mounted ≈ 400 nodes
// — comfortably inside the 16ms per-frame budget. User pages back to
// recognize, not to read; full re-render once it falls inside the tail.
export const HISTORY_RENDER_MAX_CHARS = 800
export const HISTORY_RENDER_MAX_LINES = 16
export const FULL_RENDER_TAIL_ITEMS = 8

export const LONG_MSG = 300
export const MAX_HISTORY = 800
export const THINKING_COT_MAX = 160

// Rows per wheel event (pre-accel). 1 keeps Ink's DECSTBM fast path live
// (each scroll < viewport-1) and produces smooth motion. wheelAccel.ts
// ramps this on sustained scrolls.
export const WHEEL_SCROLL_STEP = 1
