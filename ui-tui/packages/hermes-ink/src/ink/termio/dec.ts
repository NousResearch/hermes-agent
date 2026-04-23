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
// X10 bytes. Combined: wheel + click/drag for selection + hover.
export const ENABLE_MOUSE_TRACKING =
  decset(DEC.MOUSE_NORMAL) + decset(DEC.MOUSE_BUTTON) + decset(DEC.MOUSE_ANY) + decset(DEC.MOUSE_SGR)
export const DISABLE_MOUSE_TRACKING =
  decreset(DEC.MOUSE_SGR) + decreset(DEC.MOUSE_ANY) + decreset(DEC.MOUSE_BUTTON) + decreset(DEC.MOUSE_NORMAL)
