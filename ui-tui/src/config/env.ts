import { isTermuxTuiMode } from '../lib/termux.js'

const truthy = (v?: string) => /^(?:1|true|yes|on)$/i.test((v ?? '').trim())
const falsy = (v?: string) => /^(?:0|false|no|off)$/i.test((v ?? '').trim())

const parseToggle = (v?: string): boolean | null => {
  const raw = (v ?? '').trim()

  if (!raw) {
    return null
  }

  if (truthy(raw)) {
    return true
  }

  if (falsy(raw)) {
    return false
  }

  return null
}

export const TERMUX_TUI_MODE = isTermuxTuiMode()

export const STARTUP_RESUME_ID = (process.env.HERMES_TUI_RESUME ?? '').trim()
export const STARTUP_QUERY = (process.env.HERMES_TUI_QUERY ?? '').trim()
export const STARTUP_IMAGE = (process.env.HERMES_TUI_IMAGE ?? '').trim()

const mouseTrackingOverride = parseToggle(process.env.HERMES_TUI_MOUSE_TRACKING)
const mouseTrackingDisabledLegacy = truthy(process.env.HERMES_TUI_DISABLE_MOUSE)
// Mobile selection UX: on Termux default mouse tracking OFF so touch selection
// is less likely to be intercepted by terminal mouse protocols. Desktop keeps
// prior behavior unless explicitly overridden.
export const MOUSE_TRACKING =
  mouseTrackingOverride ?? (TERMUX_TUI_MODE ? false : !mouseTrackingDisabledLegacy)

export const NO_CONFIRM_DESTRUCTIVE = truthy(process.env.HERMES_TUI_NO_CONFIRM)

const inlineOverride = parseToggle(process.env.HERMES_TUI_INLINE)

// Skip AlternateScreen — TUI renders into the primary buffer so the host
// terminal's native scrollback captures whatever scrolls off the top.
//
// On Termux we default this on: users often background/foreground the app,
// and primary-buffer rendering makes long-thread review and copy/paste much
// less fragile. Override explicitly with HERMES_TUI_INLINE=0/1.
export const INLINE_MODE = inlineOverride ?? TERMUX_TUI_MODE

// Live FPS counter overlay, fed by ink's onFrame (real render rate, not a
// synthetic timer).
export const SHOW_FPS = truthy(process.env.HERMES_TUI_FPS)
