export const LARGE_PASTE = { chars: 8000, lines: 80 }

export const LIVE_RENDER_MAX_CHARS = 16_000
export const LIVE_RENDER_MAX_LINES = 240

export const LONG_MSG = 300
export const MAX_HISTORY = 800
export const THINKING_COT_MAX = 160

// Rows per wheel event (pre-accel). 1 keeps Ink's DECSTBM fast path live
// (each scroll < viewport-1) and produces smooth motion. wheelAccel.ts
// ramps this on sustained scrolls.
export const WHEEL_SCROLL_STEP = 1
