import { PLACEHOLDERS } from '../constants.js'
import { pick } from '../lib/text.js'

export const PLACEHOLDER = pick(PLACEHOLDERS)
export const STARTUP_RESUME_ID = (process.env.HERMES_TUI_RESUME ?? '').trim()

export const LARGE_PASTE = { chars: 8000, lines: 80 }
export const MAX_HISTORY = 800
export const REASONING_PULSE_MS = 700
export const STREAM_BATCH_MS = 16
export const WHEEL_SCROLL_STEP = 3
export const MOUSE_TRACKING = !/^(1|true|yes|on)$/.test(
  (process.env.HERMES_TUI_DISABLE_MOUSE ?? '').trim().toLowerCase()
)
export const PASTE_SNIPPET_RE = /\[\[[^\n]*?\]\]/g
