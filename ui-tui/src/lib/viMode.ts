/**
 * Vi/Vim mode state machine for the TUI text input.
 *
 * Implements a subset of vim's command language for single-line and
 * multi-line editing. Supports normal mode, insert mode, and operator-pending
 * mode for motions like dw, cw, etc.
 *
 * Enabled via HERMES_TUI_VIM_MODE=1 environment variable.
 */

export type ViModeType = 'normal' | 'insert' | 'visual' | 'operator-pending'

export interface ViState {
  mode: ViModeType
  /** Pending operator (d, c, y) waiting for a motion */
  operator: string | null
  /** Numeric repeat count (e.g., 3w moves 3 words) */
  count: number
  /** Visual mode anchor position */
  visualAnchor: number | null
  /** Last f/F/t/T character for ; and , repeat */
  lastFindChar: { char: string; forward: boolean; till: boolean } | null
  /** Register for yank/paste (simplified: just one unnamed register) */
  register: string
}

export const initialViState = (): ViState => ({
  mode: 'insert', // Start in insert mode for familiarity
  operator: null,
  count: 0,
  visualAnchor: null,
  lastFindChar: null,
  register: ''
})

export interface ViAction {
  type:
    | 'cursor' // Move cursor
    | 'insert' // Enter insert mode at position
    | 'delete' // Delete text
    | 'change' // Delete and enter insert mode
    | 'yank' // Copy text
    | 'paste' // Paste register
    | 'replace' // Replace single char
    | 'undo'
    | 'redo'
    | 'submit' // Submit the input (Enter in normal mode)
    | 'none' // No action (consumed but no effect)
    | 'passthrough' // Let the default handler process this
  cursor?: number
  deleteRange?: { start: number; end: number }
  text?: string
  newMode?: ViModeType
}

/**
 * Find the start of the current word (for b/B motion).
 */
function wordStart(s: string, pos: number, bigWord = false): number {
  if (pos <= 0) return 0
  let i = pos - 1

  // Skip whitespace backwards
  while (i > 0 && /\s/.test(s[i]!)) i--

  if (bigWord) {
    // WORD: non-whitespace sequence
    while (i > 0 && !/\s/.test(s[i - 1]!)) i--
  } else {
    // word: alphanumeric or punctuation sequence
    const isWord = (c: string) => /\w/.test(c)
    const startType = isWord(s[i]!)

    while (i > 0 && isWord(s[i - 1]!) === startType && !/\s/.test(s[i - 1]!)) i--
  }

  return Math.max(0, i)
}

/**
 * Find the end of the current/next word (for e/E motion).
 */
function wordEnd(s: string, pos: number, bigWord = false): number {
  const len = s.length
  if (pos >= len - 1) return len - 1
  let i = pos + 1

  // Skip whitespace forward
  while (i < len && /\s/.test(s[i]!)) i++
  if (i >= len) return len - 1

  if (bigWord) {
    while (i < len - 1 && !/\s/.test(s[i + 1]!)) i++
  } else {
    const isWord = (c: string) => /\w/.test(c)
    const startType = isWord(s[i]!)

    while (i < len - 1 && isWord(s[i + 1]!) === startType && !/\s/.test(s[i + 1]!)) i++
  }

  return i
}

/**
 * Find the start of the next word (for w/W motion).
 */
function nextWordStart(s: string, pos: number, bigWord = false): number {
  const len = s.length
  if (pos >= len) return len
  let i = pos

  if (bigWord) {
    // Skip current WORD
    while (i < len && !/\s/.test(s[i]!)) i++
    // Skip whitespace
    while (i < len && /\s/.test(s[i]!)) i++
  } else {
    const isWord = (c: string) => /\w/.test(c)
    const startType = /\s/.test(s[i]!) ? null : isWord(s[i]!)

    if (startType === null) {
      // Already on whitespace, skip it
      while (i < len && /\s/.test(s[i]!)) i++
    } else {
      // Skip current word/punct sequence
      while (i < len && isWord(s[i]!) === startType && !/\s/.test(s[i]!)) i++
      // Skip whitespace
      while (i < len && /\s/.test(s[i]!)) i++
    }
  }

  return i
}

/**
 * Find character forward (f/t motion).
 */
function findCharForward(s: string, pos: number, char: string, till: boolean): number | null {
  const idx = s.indexOf(char, pos + 1)
  if (idx === -1) return null
  return till ? idx - 1 : idx
}

/**
 * Find character backward (F/T motion).
 */
function findCharBackward(s: string, pos: number, char: string, till: boolean): number | null {
  const idx = s.lastIndexOf(char, pos - 1)
  if (idx === -1) return null
  return till ? idx + 1 : idx
}

/**
 * Get line start position (for 0 and ^ motions).
 */
function lineStart(s: string, pos: number): number {
  const lineBreak = s.lastIndexOf('\n', pos - 1)
  return lineBreak === -1 ? 0 : lineBreak + 1
}

/**
 * Get first non-whitespace position on current line (for ^ motion).
 */
function lineFirstNonWhitespace(s: string, pos: number): number {
  const start = lineStart(s, pos)
  let i = start
  while (i < s.length && s[i] !== '\n' && /\s/.test(s[i]!)) i++
  return i
}

/**
 * Get line end position (for $ motion).
 */
function lineEnd(s: string, pos: number): number {
  const lineBreak = s.indexOf('\n', pos)
  return lineBreak === -1 ? s.length : lineBreak
}

export interface ViKeyEvent {
  input: string
  key: {
    ctrl?: boolean
    shift?: boolean
    meta?: boolean
    escape?: boolean
    backspace?: boolean
    delete?: boolean
    return?: boolean
    upArrow?: boolean
    downArrow?: boolean
    leftArrow?: boolean
    rightArrow?: boolean
  }
}

/**
 * Process a key in vi mode and return the resulting action.
 */
export function processViKey(
  state: ViState,
  value: string,
  cursor: number,
  event: ViKeyEvent
): { action: ViAction; newState: ViState } {
  const { input, key } = event
  let newState = { ...state }

  // Handle Escape - always returns to normal mode
  if (key.escape || (key.ctrl && input === '[')) {
    // In insert mode, move cursor back one (vim behavior)
    const newCursor = state.mode === 'insert' && cursor > 0 ? cursor - 1 : cursor
    newState = { ...newState, mode: 'normal', operator: null, count: 0, visualAnchor: null }
    return {
      action: { type: 'cursor', cursor: newCursor, newMode: 'normal' },
      newState
    }
  }

  // Insert mode - pass through most keys
  if (state.mode === 'insert') {
    // Ctrl+C exits insert mode (alternative to Escape)
    if (key.ctrl && input === 'c') {
      const newCursor = cursor > 0 ? cursor - 1 : cursor
      newState = { ...newState, mode: 'normal', operator: null, count: 0 }
      return {
        action: { type: 'cursor', cursor: newCursor, newMode: 'normal' },
        newState
      }
    }

    // Let all other keys pass through to default handling
    return { action: { type: 'passthrough' }, newState }
  }

  // Normal mode key handling
  if (state.mode === 'normal' || state.mode === 'operator-pending') {
    // Numeric prefix (count)
    if (/^[1-9]$/.test(input) || (state.count > 0 && input === '0')) {
      newState.count = state.count * 10 + parseInt(input, 10)
      return { action: { type: 'none' }, newState }
    }

    const count = Math.max(1, state.count)
    const resetCount = () => {
      newState.count = 0
    }

    // Motion keys
    let targetPos: number | null = null
    let isLineWise = false

    switch (input) {
      // Basic movement
      case 'h':
      case '\x08': // Backspace in some terminals
        targetPos = Math.max(0, cursor - count)
        break
      case 'l':
      case ' ':
        targetPos = Math.min(value.length, cursor + count)
        break
      case 'j':
        // Move down (in multiline) or to end
        {
          const lineBreak = value.indexOf('\n', cursor)
          if (lineBreak !== -1) {
            const col = cursor - lineStart(value, cursor)
            const nextLineStart = lineBreak + 1
            const nextLineEnd = lineEnd(value, nextLineStart)
            targetPos = Math.min(nextLineStart + col, nextLineEnd)
          } else {
            targetPos = value.length
          }
        }
        break
      case 'k':
        // Move up (in multiline) or to start
        {
          const currentLineStart = lineStart(value, cursor)
          if (currentLineStart > 0) {
            const col = cursor - currentLineStart
            const prevLineEnd = currentLineStart - 1
            const prevLineStart = lineStart(value, prevLineEnd)
            targetPos = Math.min(prevLineStart + col, prevLineEnd)
          } else {
            targetPos = 0
          }
        }
        break

      // Word motions
      case 'w':
        targetPos = cursor
        for (let i = 0; i < count; i++) targetPos = nextWordStart(value, targetPos)
        break
      case 'W':
        targetPos = cursor
        for (let i = 0; i < count; i++) targetPos = nextWordStart(value, targetPos, true)
        break
      case 'b':
        targetPos = cursor
        for (let i = 0; i < count; i++) targetPos = wordStart(value, targetPos)
        break
      case 'B':
        targetPos = cursor
        for (let i = 0; i < count; i++) targetPos = wordStart(value, targetPos, true)
        break
      case 'e':
        targetPos = cursor
        for (let i = 0; i < count; i++) targetPos = wordEnd(value, targetPos)
        break
      case 'E':
        targetPos = cursor
        for (let i = 0; i < count; i++) targetPos = wordEnd(value, targetPos, true)
        break

      // Line motions
      case '0':
        targetPos = lineStart(value, cursor)
        break
      case '^':
        targetPos = lineFirstNonWhitespace(value, cursor)
        break
      case '$':
        targetPos = lineEnd(value, cursor) - (state.operator ? 0 : 1)
        targetPos = Math.max(lineStart(value, cursor), targetPos)
        break
      case 'g':
        // gg = go to start
        // We'd need to track 'g' as pending, simplify for now
        targetPos = 0
        break
      case 'G':
        targetPos = value.length
        break

      // Find character
      case 'f':
      case 'F':
      case 't':
      case 'T':
        // Need next character - enter a pending state
        // For simplicity, we'll handle this differently - queue input
        newState.operator = input
        newState.mode = 'operator-pending'
        return { action: { type: 'none' }, newState }

      // Repeat find
      case ';':
        if (state.lastFindChar) {
          const { char, forward, till } = state.lastFindChar
          targetPos = forward ? findCharForward(value, cursor, char, till) : findCharBackward(value, cursor, char, till)
        }
        break
      case ',':
        if (state.lastFindChar) {
          const { char, forward, till } = state.lastFindChar
          targetPos = forward
            ? findCharBackward(value, cursor, char, till)
            : findCharForward(value, cursor, char, till)
        }
        break

      // Operators
      case 'd':
        if (state.operator === 'd') {
          // dd - delete whole line(s)
          const start = lineStart(value, cursor)
          let end = lineEnd(value, cursor)
          // Include the newline if not at end
          if (end < value.length && value[end] === '\n') end++
          else if (start > 0) {
            // At last line, delete preceding newline
            const newStart = start - 1
            resetCount()
            newState.operator = null
            newState.register = value.slice(newStart, end)
            return {
              action: { type: 'delete', deleteRange: { start: newStart, end }, cursor: Math.max(0, newStart) },
              newState
            }
          }
          resetCount()
          newState.operator = null
          newState.register = value.slice(start, end)
          return {
            action: { type: 'delete', deleteRange: { start, end }, cursor: Math.max(0, start) },
            newState
          }
        }
        newState.operator = 'd'
        newState.mode = 'operator-pending'
        return { action: { type: 'none' }, newState }

      case 'c':
        if (state.operator === 'c') {
          // cc - change whole line (keep newline)
          const start = lineStart(value, cursor)
          const end = lineEnd(value, cursor)
          resetCount()
          newState.operator = null
          newState.mode = 'insert'
          newState.register = value.slice(start, end)
          return {
            action: { type: 'change', deleteRange: { start, end }, cursor: start, newMode: 'insert' },
            newState
          }
        }
        newState.operator = 'c'
        newState.mode = 'operator-pending'
        return { action: { type: 'none' }, newState }

      case 'y':
        if (state.operator === 'y') {
          // yy - yank whole line
          const start = lineStart(value, cursor)
          let end = lineEnd(value, cursor)
          if (end < value.length && value[end] === '\n') end++
          resetCount()
          newState.operator = null
          newState.register = value.slice(start, end)
          return { action: { type: 'yank', text: newState.register }, newState }
        }
        newState.operator = 'y'
        newState.mode = 'operator-pending'
        return { action: { type: 'none' }, newState }

      // Insert mode entry points
      case 'i':
        resetCount()
        newState.mode = 'insert'
        return { action: { type: 'insert', cursor, newMode: 'insert' }, newState }

      case 'I':
        resetCount()
        newState.mode = 'insert'
        targetPos = lineFirstNonWhitespace(value, cursor)
        return { action: { type: 'insert', cursor: targetPos, newMode: 'insert' }, newState }

      case 'a':
        resetCount()
        newState.mode = 'insert'
        return { action: { type: 'insert', cursor: Math.min(cursor + 1, value.length), newMode: 'insert' }, newState }

      case 'A':
        resetCount()
        newState.mode = 'insert'
        targetPos = lineEnd(value, cursor)
        return { action: { type: 'insert', cursor: targetPos, newMode: 'insert' }, newState }

      case 'o':
        // Open line below
        resetCount()
        newState.mode = 'insert'
        targetPos = lineEnd(value, cursor)
        return {
          action: { type: 'insert', cursor: targetPos, text: '\n', newMode: 'insert' },
          newState
        }

      case 'O':
        // Open line above
        resetCount()
        newState.mode = 'insert'
        targetPos = lineStart(value, cursor)
        return {
          action: { type: 'insert', cursor: targetPos, text: '\n', newMode: 'insert' },
          newState
        }

      case 's':
        // Substitute character(s)
        resetCount()
        newState.mode = 'insert'
        {
          const end = Math.min(cursor + count, value.length)
          newState.register = value.slice(cursor, end)
          return {
            action: { type: 'change', deleteRange: { start: cursor, end }, cursor, newMode: 'insert' },
            newState
          }
        }

      case 'S':
        // Substitute line
        resetCount()
        newState.mode = 'insert'
        {
          const start = lineStart(value, cursor)
          const end = lineEnd(value, cursor)
          newState.register = value.slice(start, end)
          return {
            action: { type: 'change', deleteRange: { start, end }, cursor: start, newMode: 'insert' },
            newState
          }
        }

      case 'C':
        // Change to end of line
        resetCount()
        newState.mode = 'insert'
        {
          const end = lineEnd(value, cursor)
          newState.register = value.slice(cursor, end)
          return {
            action: { type: 'change', deleteRange: { start: cursor, end }, cursor, newMode: 'insert' },
            newState
          }
        }

      case 'D':
        // Delete to end of line
        resetCount()
        {
          const end = lineEnd(value, cursor)
          newState.register = value.slice(cursor, end)
          return {
            action: { type: 'delete', deleteRange: { start: cursor, end }, cursor: Math.max(0, cursor - 1) },
            newState
          }
        }

      // Delete/change single character
      case 'x':
        resetCount()
        {
          const end = Math.min(cursor + count, value.length)
          newState.register = value.slice(cursor, end)
          return {
            action: { type: 'delete', deleteRange: { start: cursor, end }, cursor: Math.min(cursor, value.length - count) },
            newState
          }
        }

      case 'X':
        resetCount()
        {
          const start = Math.max(0, cursor - count)
          newState.register = value.slice(start, cursor)
          return {
            action: { type: 'delete', deleteRange: { start, end: cursor }, cursor: start },
            newState
          }
        }

      case 'r':
        // Replace single character - need next char
        newState.operator = 'r'
        newState.mode = 'operator-pending'
        return { action: { type: 'none' }, newState }

      // Paste
      case 'p':
        resetCount()
        if (newState.register) {
          const insertPos = newState.register.includes('\n') ? lineEnd(value, cursor) : cursor + 1
          return {
            action: { type: 'paste', cursor: insertPos, text: newState.register },
            newState
          }
        }
        return { action: { type: 'none' }, newState }

      case 'P':
        resetCount()
        if (newState.register) {
          const insertPos = newState.register.includes('\n') ? lineStart(value, cursor) : cursor
          return {
            action: { type: 'paste', cursor: insertPos, text: newState.register },
            newState
          }
        }
        return { action: { type: 'none' }, newState }

      // Undo/Redo
      case 'u':
        resetCount()
        return { action: { type: 'undo' }, newState }

      case '\x12': // Ctrl+R
        resetCount()
        return { action: { type: 'redo' }, newState }

      // Submit (Enter in normal mode)
      case '\r':
      case '\n':
        if (!key.shift && !key.ctrl) {
          resetCount()
          return { action: { type: 'submit' }, newState }
        }
        break

      default:
        // Handle pending operator with character argument (f, F, t, T, r)
        if (state.mode === 'operator-pending' && state.operator && input.length === 1) {
          const op = state.operator
          resetCount()
          newState.operator = null
          newState.mode = 'normal'

          if (op === 'f' || op === 'F' || op === 't' || op === 'T') {
            const forward = op === 'f' || op === 't'
            const till = op === 't' || op === 'T'
            newState.lastFindChar = { char: input, forward, till }
            targetPos = forward ? findCharForward(value, cursor, input, till) : findCharBackward(value, cursor, input, till)
            if (targetPos === null) {
              return { action: { type: 'none' }, newState }
            }
          } else if (op === 'r') {
            // Replace character
            if (cursor < value.length) {
              return {
                action: {
                  type: 'replace',
                  deleteRange: { start: cursor, end: cursor + 1 },
                  text: input,
                  cursor
                },
                newState
              }
            }
            return { action: { type: 'none' }, newState }
          }
        }
        break
    }

    // Apply operator if we have a motion target
    if (targetPos !== null && state.operator) {
      const start = Math.min(cursor, targetPos)
      const end = Math.max(cursor, targetPos)
      const op = state.operator

      resetCount()
      newState.operator = null
      newState.mode = op === 'c' ? 'insert' : 'normal'

      if (op === 'd') {
        newState.register = value.slice(start, end)
        return {
          action: { type: 'delete', deleteRange: { start, end }, cursor: start },
          newState
        }
      } else if (op === 'c') {
        newState.register = value.slice(start, end)
        return {
          action: { type: 'change', deleteRange: { start, end }, cursor: start, newMode: 'insert' },
          newState
        }
      } else if (op === 'y') {
        newState.register = value.slice(start, end)
        return { action: { type: 'yank', text: newState.register }, newState }
      }
    }

    // Just a motion (no operator)
    if (targetPos !== null) {
      resetCount()
      newState.operator = null
      newState.mode = 'normal'
      return { action: { type: 'cursor', cursor: targetPos }, newState }
    }

    // Unknown key in normal mode - do nothing
    resetCount()
    return { action: { type: 'none' }, newState }
  }

  // Fallback
  return { action: { type: 'passthrough' }, newState }
}

/**
 * Check if vim mode is enabled via environment variable.
 */
export const isViModeEnabled = (): boolean => {
  const env = process.env.HERMES_TUI_VIM_MODE ?? ''
  return /^(?:1|true|yes|on)$/i.test(env.trim())
}

/**
 * Get display indicator for current mode.
 */
export const viModeIndicator = (mode: ViModeType): string => {
  switch (mode) {
    case 'normal':
      return 'NORMAL'
    case 'insert':
      return 'INSERT'
    case 'visual':
      return 'VISUAL'
    case 'operator-pending':
      return 'OPERATOR'
    default:
      return ''
  }
}
