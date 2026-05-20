import { describe, expect, it } from 'vitest'

import { stripAnsiSequences } from '../components/textInput.js'

// Regression: #28419 — after the TUI gateway's stdout pipe breaks and
// auto-restarts, rendering ANSI bytes leak into stdin. The composer must
// not echo terminal control bytes into the input field as if they were
// keystrokes.
describe('stripAnsiSequences (#28419 TUI input ANSI flood)', () => {
  it('removes full CSI sequences', () => {
    expect(stripAnsiSequences('\x1b[35mhello\x1b[0m')).toBe('hello')
    expect(stripAnsiSequences('\x1b[2J\x1b[H')).toBe('')
  })

  it('removes OSC sequences (window title etc.)', () => {
    expect(stripAnsiSequences('\x1b]0;title\x07rest')).toBe('rest')
    expect(stripAnsiSequences('\x1b]2;abc\x1b\\rest')).toBe('rest')
  })

  it('removes two-byte ESC sequences and bare ESC bytes', () => {
    expect(stripAnsiSequences('\x1bOPfoo')).toBe('foo')
    // Trailing bare ESC after legitimate text is dropped.
    expect(stripAnsiSequences('bar\x1b')).toBe('bar')
  })

  it('removes naked control bytes but preserves \\t and \\n', () => {
    expect(stripAnsiSequences('a\x00b\x07c')).toBe('abc')
    expect(stripAnsiSequences('line1\nline2\tend')).toBe('line1\nline2\tend')
  })

  it('strips the orphan-CSI-tail garbage shape from the bug report', () => {
    // Sample from #28419: leaked ANSI whose ESC prefix was already
    // consumed earlier in the byte stream, leaving CSI tails of the form
    // `<digits>;<digits>;<digits>M` etc. These must not be inserted into
    // the input box.
    const garbage =
      '102;71M5;104;62M5;106;60M61M35;22;72M35;21;71M35;17;70MM35;10;70M;70M5;70M;85;72M2;48;72M68m1M'
    expect(stripAnsiSequences(garbage)).toBe('')
  })

  it('handles bracketed-paste-wrapped ANSI leak end-to-end', () => {
    const wrapped = '\x1b[200~\x1b[35;104;62mleak\x1b[0m\x1b[201~'
    // The composer strips bracketed-paste markers before calling
    // stripAnsiSequences; emulate that here.
    const afterPasteMarkers = wrapped.replace(/\x1b?\[20[01]~/g, '')
    expect(stripAnsiSequences(afterPasteMarkers)).toBe('leak')
  })

  it('leaves ordinary user text (including innocent semicolons) intact', () => {
    expect(stripAnsiSequences('hello, world!')).toBe('hello, world!')
    expect(stripAnsiSequences('a; b; c')).toBe('a; b; c')
    expect(stripAnsiSequences('CSS: color: red;')).toBe('CSS: color: red;')
    expect(stripAnsiSequences('')).toBe('')
  })

  it('passes through unicode and printable ASCII unchanged', () => {
    expect(stripAnsiSequences('café — 日本語 — 🎉')).toBe('café — 日本語 — 🎉')
  })
})
