/**
 * DEC (Digital Equipment Corporation) Private Mode Sequences
 *
 * DEC private modes use CSI ? N h (set) and CSI ? N l (reset) format.
 * These are terminal-specific extensions to the ANSI standard.
 */

import { csi } from './csi.js'

/**
 * DEC private mode numbers
 */
export const DEC = {
  CURSOR_VISIBLE: 25,
  ALT_SCREEN: 47,
  ALT_SCREEN_CLEAR: 1049,
  MOUSE_NORMAL: 1000,
  MOUSE_BUTTON: 1002,
  MOUSE_ANY: 1003,
  MOUSE_SGR: 1006,
  FOCUS_EVENTS: 1004,
  BRACKETED_PASTE: 2004,
  SYNCHRONIZED_UPDATE: 2026
} as const

/** Generate CSI ? N h sequence (set mode) */
export function decset(mode: number): string {
  return csi(`?${mode}h`)
}

/** Generate CSI ? N l sequence (reset mode) */
export function decreset(mode: number): string {
  return csi(`?${mode}l`)
}

// Pre-generated sequences for common modes
export const BSU = decset(DEC.SYNCHRONIZED_UPDATE)
export const ESU = decreset(DEC.SYNCHRONIZED_UPDATE)
export const EBP = decset(DEC.BRACKETED_PASTE)
export const DBP = decreset(DEC.BRACKETED_PASTE)
export const EFE = decset(DEC.FOCUS_EVENTS)
export const DFE = decreset(DEC.FOCUS_EVENTS)
export const SHOW_CURSOR = decset(DEC.CURSOR_VISIBLE)
export const HIDE_CURSOR = decreset(DEC.CURSOR_VISIBLE)
export const ENTER_ALT_SCREEN = decset(DEC.ALT_SCREEN_CLEAR)
export const EXIT_ALT_SCREEN = decreset(DEC.ALT_SCREEN_CLEAR)
// Mouse tracking: 1000 reports button press/release/wheel, 1002 adds drag
// events (button-motion), 1003 adds all-motion (no button held — for
// hover), 1006 uses SGR format (CSI < btn;col;row M/m) instead of legacy
// X10 bytes.
//
// Modes are addressable as a preset so users can opt out of 1003 (hover),
// which is the noisy one inside tmux — every cursor cross of the prompt
// row triggers a clipboard probe that surfaces as "No image in clipboard".
// Presets:
//   - 'off'     — no DECSET, terminal/tmux native selection + scroll work
//   - 'wheel'   — 1000 + 1006: click + wheel only, no drag, no hover
//   - 'buttons' — 1000 + 1002 + 1006: adds drag (text selection), no hover
//   - 'all'     — 1000 + 1002 + 1003 + 1006: legacy behavior, hover-driven
//                 UI (scrollbar paginate-on-hover, link mouseenter, etc.)
export type MouseTrackingMode = 'all' | 'buttons' | 'off' | 'wheel'

const MOUSE_NORMAL = decset(DEC.MOUSE_NORMAL)
const MOUSE_BUTTON = decset(DEC.MOUSE_BUTTON)
const MOUSE_ANY = decset(DEC.MOUSE_ANY)
const MOUSE_SGR = decset(DEC.MOUSE_SGR)

/** Sequence to enable the requested mouse tracking preset, or '' for 'off'. */
export function enableMouseTrackingFor(mode: MouseTrackingMode): string {
  switch (mode) {
    case 'all':
      return MOUSE_NORMAL + MOUSE_BUTTON + MOUSE_ANY + MOUSE_SGR

    case 'buttons':
      return MOUSE_NORMAL + MOUSE_BUTTON + MOUSE_SGR

    case 'wheel':
      return MOUSE_NORMAL + MOUSE_SGR

    case 'off':
      return ''

    default:
      // Defensive fallback: the type system guarantees exhaustiveness, but
      // JS callers / corrupted config / hot-reloads in dev could reach this
      // with an unknown value. Without a default, an unmatched mode returns
      // undefined which then concatenates as the literal string "undefined"
      // into the terminal byte stream — visibly garbling output. Treat
      // unknown as 'off' (no DEC sequences) so the worst case is silent
      // input loss rather than a wrecked screen.
      return ''
  }
}

/** Legacy alias for the maximal preset (1000 + 1002 + 1003 + 1006). */
export const ENABLE_MOUSE_TRACKING = enableMouseTrackingFor('all')
/** Reset every mouse mode unconditionally — safe to send when any subset is on. */
export const DISABLE_MOUSE_TRACKING =
  decreset(DEC.MOUSE_SGR) + decreset(DEC.MOUSE_ANY) + decreset(DEC.MOUSE_BUTTON) + decreset(DEC.MOUSE_NORMAL)
