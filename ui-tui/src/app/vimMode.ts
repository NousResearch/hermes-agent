/**
 * Vim mode for Hermes TUI input.
 *
 * Provides normal/insert mode state machine with basic vim operations:
 * - hjkl cursor movement
 * - w/b word movement
 * - 0/$ line start/end
 * - x delete char
 * - dd delete line
 * - yy yank line
 * - p/P paste
 * - u/Ctrl+r undo/redo
 * - i/a/o/O enter insert mode
 * - Esc enter normal mode
 * - gg/G first/last line
 */

export type VimMode = 'insert' | 'normal'
export type VimRegister = 'unnamed' | 'yank'

export interface VimState {
  mode: VimMode
  /** Undo stack — each entry is the previous state of the input */
  undoStack: string[]
  /** Redo stack */
  redoStack: string[]
  /** Yank register */
  yankRegister: string | null
}

/** Current vim state (file-level singleton — safe since Ink TUI is single-user) */
let state: VimState = {
  mode: 'normal',
  undoStack: [],
  redoStack: [],
  yankRegister: null,
}

export function getVimMode(): VimMode {
  return state.mode
}

export function setVimMode(mode: VimMode): void {
  state.mode = mode
}

export function toggleVimMode(): VimMode {
  state.mode = state.mode === 'normal' ? 'insert' : 'normal'
  return state.mode
}

/** Save current input to undo stack before mutating */
function pushUndo(input: string): void {
  state.undoStack.push(input)
  // Keep stack bounded
  if (state.undoStack.length > 100) {
    state.undoStack.shift()
  }
  // New mutation clears redo
  state.redoStack = []
}

/** Return cursor position within a multi-line string */
function cursorPos(input: string, cursor: number): { line: number; col: number; lineStart: number; lineEnd: number } {
  const before = input.slice(0, cursor)
  const line = before.split('\n').length - 1
  const lastNewline = before.lastIndexOf('\n')
  const col = cursor - lastNewline - 1

  // Find line boundaries
  let lineStart = input.lastIndexOf('\n', cursor - 1) + 1
  if (lineStart < 0) lineStart = 0

  let lineEnd = input.indexOf('\n', cursor)
  if (lineEnd < 0) lineEnd = input.length

  return { line, col, lineStart, lineEnd }
}

/** Find the start of the current/previous word */
function wordStart(input: string, cursor: number): number {
  // If at word boundary or on whitespace, skip back
  let pos = cursor
  // If at start, stay
  if (pos <= 0) return 0

  // If on a word character, go to start of this word
  // If on whitespace, skip back past whitespace, then go to start of word
  pos = Math.min(pos, input.length)

  // Skip trailing whitespace from cursor position
  let i = pos - 1
  while (i >= 0 && /\s/.test(input[i]!)) {
    i--
  }
  // Skip to start of word
  while (i >= 0 && !/\s/.test(input[i]!)) {
    i--
  }

  return Math.max(0, i + 1)
}

/** Find the end of the next word */
function wordEnd(input: string, cursor: number): number {
  let i = cursor
  if (i >= input.length) return input.length

  // Skip current word
  while (i < input.length && !/\s/.test(input[i]!)) {
    i++
  }
  // Skip whitespace
  while (i < input.length && /\s/.test(input[i]!)) {
    i++
  }

  return i
}

/** Find previous non-whitespace character boundary (b motion) */
function wordBack(input: string, cursor: number): number {
  return wordStart(input, cursor)
}

/** Find next word start (w motion) */
function wordForward(input: string, cursor: number): number {
  return wordEnd(input, cursor)
}

/** Move cursor to first non-whitespace of line */
function firstNonBlank(input: string, cursor: number): number {
  const pos = cursorPos(input, cursor)
  let i = pos.lineStart
  while (i < pos.lineEnd && /\s/.test(input[i]!)) {
    i++
  }
  return i
}

/** Line up (k) — preserve column if possible */
function lineUp(input: string, cursor: number): number | null {
  const pos = cursorPos(input, cursor)
  if (pos.line === 0) return null

  // Find previous line
  const prevLineEnd = pos.lineStart - 1
  let prevLineStart = input.lastIndexOf('\n', prevLineEnd - 1) + 1
  if (prevLineStart < 0) prevLineStart = 0

  const targetCol = Math.min(pos.col, prevLineEnd - prevLineStart)
  return Math.min(prevLineStart + targetCol, prevLineEnd)
}

/** Line down (j) — preserve column if possible */
function lineDown(input: string, cursor: number): number | null {
  const pos = cursorPos(input, cursor)
  // Check if there's a next line
  if (pos.lineEnd >= input.length) return null

  const nextLineStart = pos.lineEnd + 1
  const nextLineEnd = input.indexOf('\n', nextLineStart)
  const end = nextLineEnd < 0 ? input.length : nextLineEnd

  const targetCol = Math.min(pos.col, end - nextLineStart)
  return nextLineStart + targetCol
}

/** Go to specific line (1-indexed) */
function goToLine(input: string, targetLine: number): number {
  if (targetLine <= 1) return 0
  let lineCount = 1
  for (let i = 0; i < input.length; i++) {
    if (input[i] === '\n') {
      lineCount++
      if (lineCount === targetLine) return i + 1
    }
  }
  // Target line beyond last line — go to end
  return input.length
}

/** Count newlines in input */
function lineCount(input: string): number {
  if (!input) return 0
  let count = 1
  for (let i = 0; i < input.length; i++) {
    if (input[i] === '\n') count++
  }
  return count
}

/** Count newlines before cursor */
function currentLineNum(input: string, cursor: number): number {
  const before = input.slice(0, cursor)
  let count = 1
  for (let i = 0; i < before.length; i++) {
    if (before[i] === '\n') count++
  }
  return count
}

// --- Vim command handlers ---

export interface VimActionResult {
  /** New input text (or undefined if unchanged) */
  input?: string
  /** New cursor position (or undefined if unchanged) */
  cursor?: number
  /** New vim mode (or undefined if unchanged) */
  mode?: VimMode
  /** If true, the key was consumed by vim */
  consumed: boolean
}

/**
 * Process a keypress in vim normal mode.
 * Returns the action result or null if the key is not a vim command.
 *
 * @param ch The character from Ink's useInput
 * @param key The key object from Ink's useInput
 * @param currentInput Current input text
 * @param currentCursor Current cursor position
 * @param count Prefix count (e.g. 3 for "3dw")
 */
export function processVimKey(
  ch: string,
  key: {
    return?: boolean
    escape?: boolean
    backspace?: boolean
    delete?: boolean
    leftArrow?: boolean
    rightArrow?: boolean
    upArrow?: boolean
    downArrow?: boolean
    shift?: boolean
    ctrl?: boolean
    meta?: boolean
    tab?: boolean
    space?: boolean
    pageUp?: boolean
    pageDown?: boolean
    home?: boolean
    end?: boolean
  },
  currentInput: string,
  currentCursor: number,
  count: number
): VimActionResult {
  const input = currentInput
  let cursor = Math.min(currentCursor, input.length)
  const n = count || 1

  // --- Cursor movement (don't modify text) ---

  if (ch === 'h' || key.leftArrow) {
    cursor = Math.max(0, cursor - n)
    return { cursor, consumed: true }
  }

  if (ch === 'l' || key.rightArrow) {
    cursor = Math.min(input.length, cursor + n)
    return { cursor, consumed: true }
  }

  if (ch === 'j' || key.downArrow) {
    const newPos = lineDown(input, cursor)
    if (newPos !== null) {
      cursor = newPos
    }
    return { cursor, consumed: true }
  }

  if (ch === 'k' || key.upArrow) {
    const newPos = lineUp(input, cursor)
    if (newPos !== null) {
      cursor = newPos
    }
    return { cursor, consumed: true }
  }

  if (ch === 'w') {
    for (let i = 0; i < n; i++) {
      cursor = wordForward(input, cursor)
    }
    return { cursor, consumed: true }
  }

  if (ch === 'b') {
    for (let i = 0; i < n; i++) {
      cursor = wordBack(input, cursor)
    }
    return { cursor, consumed: true }
  }

  if (ch === '0') {
    const pos = cursorPos(input, cursor)
    cursor = pos.lineStart
    return { cursor, consumed: true }
  }

  if (ch === '^') {
    cursor = firstNonBlank(input, cursor)
    return { cursor, consumed: true }
  }

  if (ch === '$') {
    const pos = cursorPos(input, cursor)
    cursor = pos.lineEnd
    return { cursor, consumed: true }
  }

  if (ch === 'g') {
    // gg = go to line 1
    cursor = 0
    return { cursor, consumed: true }
  }

  if (ch === 'G') {
    // Go to last line start
    const lastNewline = input.lastIndexOf('\n')
    cursor = lastNewline < 0 ? 0 : lastNewline + 1
    return { cursor, consumed: true }
  }

  if (ch === '%') {
    // Find matching bracket (basic — just move to next (, {, [ or their closes)
    const at = input[cursor]
    if (at === '(' || at === '{' || at === '[') {
      const close = at === '(' ? ')' : at === '{' ? '}' : ']'
      let depth = 1
      for (let i = cursor + 1; i < input.length; i++) {
        if (input[i] === at) depth++
        if (input[i] === close) {
          depth--
          if (depth === 0) {
            cursor = i
            break
          }
        }
      }
    } else if (at === ')' || at === '}' || at === ']') {
      const open = at === ')' ? '(' : at === '}' ? '{' : '['
      let depth = 1
      for (let i = cursor - 1; i >= 0; i--) {
        if (input[i] === at) depth++
        if (input[i] === open) {
          depth--
          if (depth === 0) {
            cursor = i
            break
          }
        }
      }
    }
    return { cursor, consumed: true }
  }

  // --- Enter insert mode ---

  if (ch === 'i') {
    return { consumed: true, mode: 'insert' }
  }

  if (ch === 'A' || (ch === 'a' && key.shift)) {
    // Append at end of line. Ink may report Shift+A as ch='a' with key.shift.
    const pos = cursorPos(input, cursor)
    cursor = pos.lineEnd
    return { cursor, consumed: true, mode: 'insert' }
  }

  if (ch === 'a') {
    // Append: move cursor right by 1, then enter insert
    cursor = Math.min(input.length, cursor + 1)
    return { cursor, consumed: true, mode: 'insert' }
  }

  if (ch === 'I') {
    // Insert at start of line (first non-whitespace)
    cursor = firstNonBlank(input, cursor)
    return { cursor, consumed: true, mode: 'insert' }
  }

  if (ch === 'o') {
    // Open new line below
    const pos = cursorPos(input, cursor)
    const insertion = '\n' + ' '.repeat(pos.col)
    pushUndo(input)
    const newInput = input.slice(0, pos.lineEnd) + insertion + input.slice(pos.lineEnd)
    cursor = pos.lineEnd + insertion.length
    return { input: newInput, cursor, consumed: true, mode: 'insert' }
  }

  if (ch === 'O') {
    // Open new line above
    const pos = cursorPos(input, cursor)
    pushUndo(input)
    const insertion = '\n'
    let newInput: string
    let newCursor: number
    if (pos.lineStart === 0) {
      newInput = insertion + input
      newCursor = insertion.length
    } else {
      newInput = input.slice(0, pos.lineStart) + insertion + input.slice(pos.lineStart)
      newCursor = pos.lineStart + insertion.length
    }
    return { input: newInput, cursor: newCursor, consumed: true, mode: 'insert' }
  }

  // --- Edit operations ---

  if (ch === 'x' || key.delete) {
    // Delete character under cursor
    if (cursor < input.length) {
      pushUndo(input)
      const newInput = input.slice(0, cursor) + input.slice(cursor + 1)
      return { input: newInput, cursor, consumed: true }
    }
    return { consumed: true }
  }

  if (ch === 'X' || key.backspace) {
    // Delete character before cursor
    if (cursor > 0) {
      pushUndo(input)
      const newInput = input.slice(0, cursor - 1) + input.slice(cursor)
      return { input: newInput, cursor: cursor - 1, consumed: true }
    }
    return { consumed: true }
  }

  if (ch === 'd') {
    // dd = delete line
    const pos = cursorPos(input, cursor)
    pushUndo(input)
    let newInput: string
    let newCursor: number

    if (pos.lineStart === 0) {
      // First line
      const after = input.indexOf('\n')
      if (after < 0) {
        // Only one line
        newInput = ''
        newCursor = 0
      } else {
        newInput = input.slice(after + 1)
        newCursor = 0
      }
    } else {
      const beforeNewline = input.lastIndexOf('\n', pos.lineStart - 2)
      const prevLineStart = beforeNewline < 0 ? 0 : beforeNewline + 1
      newCursor = prevLineStart

      if (pos.lineEnd >= input.length) {
        // Last line
        newInput = input.slice(0, pos.lineStart - 1) // Remove trailing newline too
      } else {
        newInput = input.slice(0, pos.lineStart) + input.slice(pos.lineEnd + 1)
      }
    }

    // Store deleted text in yank register
    const deletedText = input.slice(pos.lineStart, pos.lineEnd)
    state.yankRegister = deletedText

    return { input: newInput, cursor: newCursor, consumed: true }
  }

  if (ch === 'y') {
    // yy = yank (copy) line
    const pos = cursorPos(input, cursor)
    state.yankRegister = input.slice(pos.lineStart, pos.lineEnd)
    return { consumed: true }
  }

  if (ch === 'p') {
    // Paste below current line
    if (state.yankRegister !== null) {
      pushUndo(input)
      const pos = cursorPos(input, cursor)
      const insertion = '\n' + state.yankRegister
      const newInput = input.slice(0, pos.lineEnd) + insertion + input.slice(pos.lineEnd)
      cursor = pos.lineEnd + insertion.length
      return { input: newInput, cursor, consumed: true }
    }
    return { consumed: true }
  }

  if (ch === 'P') {
    // Paste above current line
    if (state.yankRegister !== null) {
      pushUndo(input)
      const pos = cursorPos(input, cursor)
      const insertion = state.yankRegister + '\n'
      const newInput = input.slice(0, pos.lineStart) + insertion + input.slice(pos.lineStart)
      cursor = pos.lineStart + state.yankRegister.length
      return { input: newInput, cursor, consumed: true }
    }
    return { consumed: true }
  }

  if (ch === 'u' || (ch === 'z' && key.ctrl)) {
    // Undo
    if (state.undoStack.length > 0) {
      state.redoStack.push(input)
      const prev = state.undoStack.pop()!
      cursor = Math.min(cursor, prev.length)
      return { input: prev, cursor, consumed: true }
    }
    return { consumed: true }
  }

  if (ch === '\u0012' || (key.ctrl && ch.toLowerCase() === 'r')) {
    // Redo
    if (state.redoStack.length > 0) {
      state.undoStack.push(input)
      const next = state.redoStack.pop()!
      cursor = Math.min(cursor, next.length)
      return { input: next, cursor, consumed: true }
    }
    return { consumed: true }
  }

  if (ch === 'r') {
    // Replace single character (enter insert mode after replacing one char)
    return { consumed: true }
  }

  // Not a vim command — key will fall through to normal handling
  return { consumed: false }
}

/** Reset vim state for a new session */
export function resetVimState(): void {
  state = {
    mode: 'normal',
    undoStack: [],
    redoStack: [],
    yankRegister: null,
  }
}
