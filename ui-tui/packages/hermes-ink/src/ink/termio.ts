/**
 * ANSI Parser Module
 *
 * A semantic ANSI escape sequence parser inspired by ghostty, tmux, and iTerm2.
 *
 * Key features:
 * - Semantic output: produces structured actions, not string tokens
 * - Streaming: can parse input incrementally via Parser class
 * - Style tracking: maintains text style state across parse calls
 * - Comprehensive: supports SGR, CSI, OSC, ESC sequences
 *
 * Usage:
 *
 * ```typescript
 * import { Parser } from './termio.js'
 *
 * const parser = new Parser()
 * const actions = parser.feed('\x1b[31mred\x1b[0m')
 * // => [{ type: 'text', graphemes: [...], style: { fg: { type: 'named', name: 'red' }, ... } }]
 * ```
 */

// Parser
export { Parser } from './termio/parser.js'
// Types
export type {
  Action,
  Color,
  CursorAction,
  CursorDirection,
  EraseAction,
  Grapheme,
  LinkAction,
  ModeAction,
  NamedColor,
  ScrollAction,
  TextSegment,
  TextStyle,
  TitleAction,
  UnderlineStyle
} from './termio/types.js'
export { colorsEqual, defaultStyle, stylesEqual } from './termio/types.js'
