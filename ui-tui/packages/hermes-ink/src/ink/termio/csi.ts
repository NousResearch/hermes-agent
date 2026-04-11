/**
 * CSI (Control Sequence Introducer) Types
 *
 * Enums and types for CSI command parameters.
 */

import { ESC, ESC_TYPE, SEP } from './ansi.js'

export const CSI_PREFIX = ESC + String.fromCharCode(ESC_TYPE.CSI)

/**
 * CSI parameter byte ranges
 */
export const CSI_RANGE = {
  PARAM_START: 0x30,
  PARAM_END: 0x3f,
  INTERMEDIATE_START: 0x20,
  INTERMEDIATE_END: 0x2f,
  FINAL_START: 0x40,
  FINAL_END: 0x7e
} as const

/** Check if a byte is a CSI parameter byte */
export function isCSIParam(byte: number): boolean {
  return byte >= CSI_RANGE.PARAM_START && byte <= CSI_RANGE.PARAM_END
}

/** Check if a byte is a CSI intermediate byte */
export function isCSIIntermediate(byte: number): boolean {
  return byte >= CSI_RANGE.INTERMEDIATE_START && byte <= CSI_RANGE.INTERMEDIATE_END
}

/** Check if a byte is a CSI final byte (@ through ~) */
export function isCSIFinal(byte: number): boolean {
  return byte >= CSI_RANGE.FINAL_START && byte <= CSI_RANGE.FINAL_END
}

/**
 * Generate a CSI sequence: ESC [ p1;p2;...;pN final
 * Single arg: treated as raw body
 * Multiple args: last is final byte, rest are params joined by ;
 */
export function csi(...args: (string | number)[]): string {
  if (args.length === 0) {
    return CSI_PREFIX
  }

  if (args.length === 1) {
    return `${CSI_PREFIX}${args[0]}`
  }

  const params = args.slice(0, -1)
  const final = args[args.length - 1]

  return `${CSI_PREFIX}${params.join(SEP)}${final}`
}

/**
 * CSI final bytes - the command identifier
 */
export const CSI = {
  // Cursor movement
  CUU: 0x41, // A - Cursor Up
  CUD: 0x42, // B - Cursor Down
  CUF: 0x43, // C - Cursor Forward
  CUB: 0x44, // D - Cursor Back
  CNL: 0x45, // E - Cursor Next Line
  CPL: 0x46, // F - Cursor Previous Line
  CHA: 0x47, // G - Cursor Horizontal Absolute
  CUP: 0x48, // H - Cursor Position
  CHT: 0x49, // I - Cursor Horizontal Tab
  VPA: 0x64, // d - Vertical Position Absolute
  HVP: 0x66, // f - Horizontal Vertical Position

  // Erase
  ED: 0x4a, // J - Erase in Display
  EL: 0x4b, // K - Erase in Line
  ECH: 0x58, // X - Erase Character

  // Insert/Delete
  IL: 0x4c, // L - Insert Lines
  DL: 0x4d, // M - Delete Lines
  ICH: 0x40, // @ - Insert Characters
  DCH: 0x50, // P - Delete Characters

  // Scroll
  SU: 0x53, // S - Scroll Up
  SD: 0x54, // T - Scroll Down

  // Modes
  SM: 0x68, // h - Set Mode
  RM: 0x6c, // l - Reset Mode

  // SGR
  SGR: 0x6d, // m - Select Graphic Rendition

  // Other
  DSR: 0x6e, // n - Device Status Report
  DECSCUSR: 0x71, // q - Set Cursor Style (with space intermediate)
  DECSTBM: 0x72, // r - Set Top and Bottom Margins
  SCOSC: 0x73, // s - Save Cursor Position
  SCORC: 0x75, // u - Restore Cursor Position
  CBT: 0x5a // Z - Cursor Backward Tabulation
} as const

/**
 * Erase in Display regions (ED command parameter)
 */
export const ERASE_DISPLAY = ['toEnd', 'toStart', 'all', 'scrollback'] as const

/**
 * Erase in Line regions (EL command parameter)
 */
export const ERASE_LINE_REGION = ['toEnd', 'toStart', 'all'] as const

/**
 * Cursor styles (DECSCUSR)
 */
export type CursorStyle = 'block' | 'underline' | 'bar'

export const CURSOR_STYLES: Array<{ style: CursorStyle; blinking: boolean }> = [
  { style: 'block', blinking: true }, // 0 - default
  { style: 'block', blinking: true }, // 1
  { style: 'block', blinking: false }, // 2
  { style: 'underline', blinking: true }, // 3
  { style: 'underline', blinking: false }, // 4
  { style: 'bar', blinking: true }, // 5
  { style: 'bar', blinking: false } // 6
]

// Cursor movement generators

/** Move cursor up n lines (CSI n A) */
export function cursorUp(n = 1): string {
  return n === 0 ? '' : csi(n, 'A')
}

/** Move cursor down n lines (CSI n B) */
export function cursorDown(n = 1): string {
  return n === 0 ? '' : csi(n, 'B')
}

/** Move cursor forward n columns (CSI n C) */
export function cursorForward(n = 1): string {
  return n === 0 ? '' : csi(n, 'C')
}

/** Move cursor back n columns (CSI n D) */
export function cursorBack(n = 1): string {
  return n === 0 ? '' : csi(n, 'D')
}

/** Move cursor to column n (1-indexed) (CSI n G) */
export function cursorTo(col: number): string {
  return csi(col, 'G')
}

/** Move cursor to column 1 (CSI G) */
export const CURSOR_LEFT = csi('G')

/** Move cursor to row, col (1-indexed) (CSI row ; col H) */
export function cursorPosition(row: number, col: number): string {
  return csi(row, col, 'H')
}

/** Move cursor to home position (CSI H) */
export const CURSOR_HOME = csi('H')

/**
 * Move cursor relative to current position
 * Positive x = right, negative x = left
 * Positive y = down, negative y = up
 */
export function cursorMove(x: number, y: number): string {
  let result = ''

  // Horizontal first (matches ansi-escapes behavior)
  if (x < 0) {
    result += cursorBack(-x)
  } else if (x > 0) {
    result += cursorForward(x)
  }

  // Then vertical
  if (y < 0) {
    result += cursorUp(-y)
  } else if (y > 0) {
    result += cursorDown(y)
  }

  return result
}

// Save/restore cursor position

/** Save cursor position (CSI s) */
export const CURSOR_SAVE = csi('s')

/** Restore cursor position (CSI u) */
export const CURSOR_RESTORE = csi('u')

// Erase generators

/** Erase from cursor to end of line (CSI K) */
export function eraseToEndOfLine(): string {
  return csi('K')
}

/** Erase from cursor to start of line (CSI 1 K) */
export function eraseToStartOfLine(): string {
  return csi(1, 'K')
}

/** Erase entire line (CSI 2 K) */
export function eraseLine(): string {
  return csi(2, 'K')
}

/** Erase entire line - constant form */
export const ERASE_LINE = csi(2, 'K')

/** Erase from cursor to end of screen (CSI J) */
export function eraseToEndOfScreen(): string {
  return csi('J')
}

/** Erase from cursor to start of screen (CSI 1 J) */
export function eraseToStartOfScreen(): string {
  return csi(1, 'J')
}

/** Erase entire screen (CSI 2 J) */
export function eraseScreen(): string {
  return csi(2, 'J')
}

/** Erase entire screen - constant form */
export const ERASE_SCREEN = csi(2, 'J')

/** Erase scrollback buffer (CSI 3 J) */
export const ERASE_SCROLLBACK = csi(3, 'J')

/**
 * Erase n lines starting from cursor line, moving cursor up
 * This erases each line and moves up, ending at column 1
 */
export function eraseLines(n: number): string {
  if (n <= 0) {
    return ''
  }

  let result = ''

  for (let i = 0; i < n; i++) {
    result += ERASE_LINE

    if (i < n - 1) {
      result += cursorUp(1)
    }
  }

  result += CURSOR_LEFT

  return result
}

// Scroll

/** Scroll up n lines (CSI n S) */
export function scrollUp(n = 1): string {
  return n === 0 ? '' : csi(n, 'S')
}

/** Scroll down n lines (CSI n T) */
export function scrollDown(n = 1): string {
  return n === 0 ? '' : csi(n, 'T')
}

/** Set scroll region (DECSTBM, CSI top;bottom r). 1-indexed, inclusive. */
export function setScrollRegion(top: number, bottom: number): string {
  return csi(top, bottom, 'r')
}

/** Reset scroll region to full screen (DECSTBM, CSI r). Homes the cursor. */
export const RESET_SCROLL_REGION = csi('r')

// Bracketed paste markers (input from terminal, not output)
// These are sent by the terminal to delimit pasted content when
// bracketed paste mode is enabled (via DEC mode 2004)

/** Sent by terminal before pasted content (CSI 200 ~) */
export const PASTE_START = csi('200~')

/** Sent by terminal after pasted content (CSI 201 ~) */
export const PASTE_END = csi('201~')

// Focus event markers (input from terminal, not output)
// These are sent by the terminal when focus changes while
// focus events mode is enabled (via DEC mode 1004)

/** Sent by terminal when it gains focus (CSI I) */
export const FOCUS_IN = csi('I')

/** Sent by terminal when it loses focus (CSI O) */
export const FOCUS_OUT = csi('O')

// Kitty keyboard protocol (CSI u)
// Enables enhanced key reporting with modifier information
// See: https://sw.kovidgoyal.net/kitty/keyboard-protocol/

/**
 * Enable Kitty keyboard protocol with basic modifier reporting
 * CSI > 1 u - pushes mode with flags=1 (disambiguate escape codes)
 * This makes Shift+Enter send CSI 13;2 u instead of just CR
 */
export const ENABLE_KITTY_KEYBOARD = csi('>1u')

/**
 * Disable Kitty keyboard protocol
 * CSI < u - pops the keyboard mode stack
 */
export const DISABLE_KITTY_KEYBOARD = csi('<u')

/**
 * Enable xterm modifyOtherKeys level 2.
 * tmux accepts this (not the kitty stack) to enable extended keys — when
 * extended-keys-format is csi-u, tmux then emits keys in kitty format.
 */
export const ENABLE_MODIFY_OTHER_KEYS = csi('>4;2m')

/**
 * Disable xterm modifyOtherKeys (reset to default).
 */
export const DISABLE_MODIFY_OTHER_KEYS = csi('>4m')
