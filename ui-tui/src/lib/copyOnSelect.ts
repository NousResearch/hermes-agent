import { isMac } from './platform.js'

/** Config value of `display.copy_on_select`; 'auto' (default) defers to the heuristic. */
export type CopyOnSelectSetting = 'auto' | 'on' | 'off'

const TRUEY = new Set(['1', 'true', 'yes', 'on'])
const FALSEY = new Set(['0', 'false', 'no', 'off'])

/**
 * Normalize `display.copy_on_select` (boolean back-compat plus `auto|on|off`,
 * same shape as `normalizeMouseTracking`). Absent/anything unrecognized: 'auto'.
 */
export const normalizeCopyOnSelect = (display: { copy_on_select?: unknown }): CopyOnSelectSetting => {
  const raw = display.copy_on_select

  if (raw === true) {
    return 'on'
  }

  if (raw === false) {
    return 'off'
  }

  if (typeof raw !== 'string') {
    return 'auto'
  }

  const v = raw.trim().toLowerCase()

  if (TRUEY.has(v)) {
    return 'on'
  }

  if (FALSEY.has(v)) {
    return 'off'
  }

  return 'auto'
}

/**
 * Whether copy-on-select (auto-copy a finished mouse selection to the
 * clipboard) is enabled. Priority: `HERMES_TUI_COPY_ON_SELECT` (boot
 * override) > `display.copy_on_select` (on/off) > heuristic.
 *
 * The 'auto' heuristic enables it when the HOST is macOS (upstream behavior)
 * or the session is an inbound SSH session: remote users need auto-copy
 * most (the app holds the alt screen, so no native selection exists, and
 * the host's platform says nothing about the client terminal — OSC 52
 * travels back over the pty either way, which setClipboard() already uses).
 * Plain local Linux/Windows sessions stay off by default: X11/Wayland
 * already auto-copy selections to the primary register, OSC 52 support
 * there is uneven and sometimes deliberately disabled, and
 * selection-clobbers-clipboard would regress those workflows — all of
 * which are exactly what the explicit setting exists to override.
 */
export function copyOnSelectEnabled(
  env: NodeJS.ProcessEnv = process.env,
  mac: boolean = isMac,
  setting: CopyOnSelectSetting = 'auto'
): boolean {
  const forced = env['HERMES_TUI_COPY_ON_SELECT']?.trim().toLowerCase()

  if (forced !== undefined && forced !== '') {
    if (TRUEY.has(forced)) {
      return true
    }

    if (FALSEY.has(forced)) {
      return false
    }
  }

  if (setting === 'on') {
    return true
  }

  if (setting === 'off') {
    return false
  }

  return mac || !!env['SSH_CONNECTION']
}
