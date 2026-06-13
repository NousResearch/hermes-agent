/**
 * Input Tokenizer - Escape sequence boundary detection
 *
 * Splits terminal input into tokens: text chunks and raw escape sequences.
 * Unlike the Parser which interprets sequences semantically, this just
 * identifies boundaries for use by keyboard input parsing.
 */

import { C0, ESC_TYPE, isEscFinal } from './ansi.js'
import { isCSIFinal, isCSIIntermediate, isCSIParam } from './csi.js'

export type Token = { type: 'text'; value: string } | { type: 'sequence'; value: string }

type State = 'ground' | 'escape' | 'escapeIntermediate' | 'csi' | 'ss3' | 'osc' | 'dcs' | 'apc'

export type Tokenizer = {
  /** Feed input and get resulting tokens */
  feed(input: string): Token[]
  /** Flush any buffered incomplete sequences */
  flush(): Token[]
  /** Reset tokenizer state */
  reset(): void
  /** Get any buffered incomplete sequence */
  buffer(): string
}

type TokenizerOptions = {
  /**
   * Treat `CSI M` as an X10 mouse event prefix and consume 3 payload bytes.
   * Only enable for stdin input — `\x1b[M` is also CSI DL (Delete Lines) in
   * output streams, and enabling this there swallows display text. Default false.
   */
  x10Mouse?: boolean
}

/**
 * Create a streaming tokenizer for terminal input.
 *
 * Usage:
 * ```typescript
 * const tokenizer = createTokenizer()
 * const tokens1 = tokenizer.feed('hello\x1b[')
 * const tokens2 = tokenizer.feed('A')  // completes the escape sequence
 * const remaining = tokenizer.flush()  // force output incomplete sequences
 * ```
 */
export function createTokenizer(options?: TokenizerOptions): Tokenizer {
  let currentState: State = 'ground'
  let currentBuffer = ''
  const x10Mouse = options?.x10Mouse ?? false

  return {
    feed(input: string): Token[] {
      const result = tokenize(input, currentState, currentBuffer, false, x10Mouse)

      currentState = result.state.state
      currentBuffer = result.state.buffer

      return result.tokens
    },

    flush(): Token[] {
      const result = tokenize('', currentState, currentBuffer, true, x10Mouse)
      currentState = result.state.state
      currentBuffer = result.state.buffer

      return result.tokens
    },

    reset(): void {
      currentState = 'ground'
      currentBuffer = ''
    },

    buffer(): string {
      return currentBuffer
    }
  }
}

type InternalState = {
  state: State
  buffer: string
}

function tokenize(
  input: string,
  initialState: State,
  initialBuffer: string,
  flush: boolean,
  x10Mouse: boolean
): { tokens: Token[]; state: InternalState } {
  const tokens: Token[] = []

  const result: InternalState = {
    state: initialState,
    buffer: ''
  }

  const data = initialBuffer + input
  let i = 0
  let textStart = 0
  let seqStart = 0

  const flushText = (): void => {
    if (i > textStart) {
      const text = data.slice(textStart, i)

      if (text) {
        tokens.push({ type: 'text', value: text })
      }
    }

    textStart = i
  }

  const emitSequence = (seq: string): void => {
    if (seq) {
      tokens.push({ type: 'sequence', value: seq })
    }

    result.state = 'ground'
    textStart = i
  }

  while (i < data.length) {
    const code = data.charCodeAt(i)

    switch (result.state) {
      case 'ground':
        if (code === C0.ESC) {
          flushText()
          seqStart = i
          result.state = 'escape'
          i++
        } else {
          i++
        }

        break

      case 'escape':
        if (code === ESC_TYPE.CSI) {
          result.state = 'csi'
          i++
        } else if (code === ESC_TYPE.OSC) {
          result.state = 'osc'
          i++
        } else if (code === ESC_TYPE.DCS) {
          result.state = 'dcs'
          i++
        } else if (code === ESC_TYPE.APC) {
          result.state = 'apc'
          i++
        } else if (code === 0x4f) {
          // 'O' - SS3
          result.state = 'ss3'
          i++
        } else if (isCSIIntermediate(code)) {
          // Intermediate byte (e.g., ESC ( for charset) - continue buffering
          result.state = 'escapeIntermediate'
          i++
        } else if (isEscFinal(code)) {
          // Two-character escape sequence
          i++
          emitSequence(data.slice(seqStart, i))
        } else if (code === C0.ESC) {
          // Double escape - emit first, start new
          emitSequence(data.slice(seqStart, i))
          seqStart = i
          result.state = 'escape'
          i++
        } else {
          // Invalid - treat ESC as text
          result.state = 'ground'
          textStart = seqStart
        }

        break

      case 'escapeIntermediate':
        // After intermediate byte(s), wait for final byte
        if (isCSIIntermediate(code)) {
          // More intermediate bytes
          i++
        } else if (isEscFinal(code)) {
          // Final byte - complete the sequence
          i++
          emitSequence(data.slice(seqStart, i))
        } else {
          // Invalid - treat as text
          result.state = 'ground'
          textStart = seqStart
        }

        break

      case 'csi':
        // X10 mouse: CSI M + 3 raw payload bytes (Cb+32, Cx+32, Cy+32).
        // M immediately after [ (offset 2) means no params — SGR mouse
        // (CSI < … M) has a `<` param byte first and reaches M at offset > 2.
        // Terminals that ignore DECSET 1006 but honor 1000/1002 emit this
        // legacy encoding; without this branch the 3 payload bytes leak
        // through as text (`` `rK `` / `arK` garbage in the prompt).
        //
        // Gated on x10Mouse — `\x1b[M` is also CSI DL (Delete Lines) and
        // blindly consuming 3 chars corrupts output rendering (Parser/Ansi)
        // and fragments bracketed-paste PASTE_END. Only stdin enables this.
        // The ≥0x20 check on each payload slot is belt-and-suspenders: X10
        // guarantees Cb≥32, Cx≥33, Cy≥33, so a control byte (ESC=0x1B) in
        // any slot means this is CSI DL adjacent to another sequence, not a
        // mouse event. Checking all three slots prevents PASTE_END's ESC
        // from being consumed when paste content ends in `\x1b[M`+0-2 chars.
        //
        // Known limitation: this counts JS string chars, but X10 is byte-
        // oriented and stdin uses utf8 encoding (App.tsx). At col 162-191 ×
        // row 96-159 the two coord bytes (0xC2-0xDF, 0x80-0xBF) form a valid
        // UTF-8 2-byte sequence and collapse to one char — the length check
        // fails and the event buffers until the next keypress absorbs it.
        // Fixing this requires latin1 stdin; X10's 223-coord cap is exactly
        // why SGR was invented, and no-SGR terminals at 162+ cols are rare.
        if (
          x10Mouse &&
          code === 0x4d /* M */ &&
          i - seqStart === 2 &&
          (i + 1 >= data.length || data.charCodeAt(i + 1) >= 0x20) &&
          (i + 2 >= data.length || data.charCodeAt(i + 2) >= 0x20) &&
          (i + 3 >= data.length || data.charCodeAt(i + 3) >= 0x20)
        ) {
          if (i + 4 <= data.length) {
            i += 4
            emitSequence(data.slice(seqStart, i))
          } else {
            // Incomplete — exit loop; end-of-input buffers from seqStart.
            // Re-entry re-tokenizes from ground via the invalid-CSI fallthrough.
            i = data.length
          }

          break
        }

        if (isCSIFinal(code)) {
          i++
          emitSequence(data.slice(seqStart, i))
        } else if (isCSIParam(code) || isCSIIntermediate(code)) {
          i++
        } else {
          // Invalid CSI - abort, treat as text
          result.state = 'ground'
          textStart = seqStart
        }

        break

      case 'ss3':
        // SS3 sequences: ESC O followed by a single final byte
        if (code >= 0x40 && code <= 0x7e) {
          i++
          emitSequence(data.slice(seqStart, i))
        } else {
          // Invalid - treat as text
          result.state = 'ground'
          textStart = seqStart
        }

        break

      case 'osc':
        if (code === C0.BEL) {
          i++
          emitSequence(data.slice(seqStart, i))
        } else if (code === C0.ESC && i + 1 < data.length && data.charCodeAt(i + 1) === ESC_TYPE.ST) {
          i += 2
          emitSequence(data.slice(seqStart, i))
        } else {
          i++
        }

        break

      case 'dcs':

      case 'apc':
        if (code === C0.BEL) {
          i++
          emitSequence(data.slice(seqStart, i))
        } else if (code === C0.ESC && i + 1 < data.length && data.charCodeAt(i + 1) === ESC_TYPE.ST) {
          i += 2
          emitSequence(data.slice(seqStart, i))
        } else {
          i++
        }

        break
    }
  }

  // Handle end of input
  if (result.state === 'ground') {
    flushText()
  } else if (flush) {
    // Force output incomplete sequence
    const remaining = data.slice(seqStart)

    if (remaining) {
      tokens.push({ type: 'sequence', value: remaining })
    }

    result.state = 'ground'
  } else {
    // Buffer incomplete sequence for next call
    result.buffer = data.slice(seqStart)
  }

  return { tokens, state: result }
}
