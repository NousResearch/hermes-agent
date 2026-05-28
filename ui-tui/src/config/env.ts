import type { MouseTrackingMode } from '@hermes/ink'

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

// Mouse tracking mode resolution at startup. Per-mode selection (off|wheel|
// buttons|all) lives in display.mouse_tracking in config.yaml — these env
// vars only set the boot-time default before that config is applied.
//
// Precedence (highest first):
//
// - HERMES_TUI_MOUSE_TRACKING (truthy/falsy) explicitly overrides everything.
//   This is the "force a value" knob and intentionally beats the legacy
//   kill-switch and the Termux default.
// - HERMES_TUI_DISABLE_MOUSE=1 forces mouse off — the legacy kill switch.
// - On Termux the default is mouse off so touch selection isn't intercepted
//   by terminal mouse protocols. Desktop defaults to 'all' to preserve prior
//   behavior.
const mouseTrackingOverride = parseToggle(process.env.HERMES_TUI_MOUSE_TRACKING)
const mouseTrackingDisabledLegacy = truthy(process.env.HERMES_TUI_DISABLE_MOUSE)
const resolvedBootMouseEnabled =
  mouseTrackingOverride ?? (TERMUX_TUI_MODE ? false : !mouseTrackingDisabledLegacy)
export const MOUSE_TRACKING: MouseTrackingMode = resolvedBootMouseEnabled ? 'all' : 'off'

export const NO_CONFIRM_DESTRUCTIVE = truthy(process.env.HERMES_TUI_NO_CONFIRM)

// Skip AlternateScreen — TUI renders into the primary buffer so the host
// terminal's native scrollback captures whatever scrolls off the top.
//
// On Termux and tmux we default this on: users often rely on the host/tmux
// scrollback and copy-mode (`Ctrl-b [`). AlternateScreen makes tmux report
// `alternate_on=1` and keeps pane history at 0, so copy-mode opens but cannot
// scroll the Hermes transcript. Override explicitly with HERMES_TUI_INLINE=0/1.
export const isInlineModeEnabled = (env: NodeJS.ProcessEnv = process.env): boolean => {
  const override = parseToggle(env.HERMES_TUI_INLINE)

  return override ?? (isTermuxTuiMode(env) || !!env.TMUX)
}

export const INLINE_MODE = isInlineModeEnabled()

// Live FPS counter overlay, fed by ink's onFrame (real render rate, not a
// synthetic timer).
export const SHOW_FPS = truthy(process.env.HERMES_TUI_FPS)
