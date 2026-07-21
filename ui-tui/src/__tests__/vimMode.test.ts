import { beforeEach, describe, expect, it } from 'vitest'

import { beginVimInsertSession, processVimKey, recordVimInsertEdit, resetVimState } from '../app/vimMode.js'
import { setVimMode } from '../app/vimModeStore.js'
import type { Key } from '@hermes/ink'

function makeKey(overrides: Partial<Key> = {}): Key {
  return {
    ctrl: false,
    meta: false,
    shift: false,
    ...overrides
  }
}

/**
 * Helper: perform a full yy (yank) operation by calling processVimKey('y')
 * which sets the yank register, then returns the state.
 */
function yy(input: string, cursor: number): void {
  processVimKey('y', makeKey(), input, cursor, 1)
  processVimKey('y', makeKey(), input, cursor, 1)
}

describe('processVimKey', () => {
  beforeEach(() => {
    resetVimState()
    setVimMode('normal')
  })

  // ── Cursor movement: hjkl ──────────────────────────────────────────

  describe('h — cursor left', () => {
    it('moves cursor left by one', () => {
      const r = processVimKey('h', makeKey(), 'hello', 3, 1)
      expect(r.consumed).toBe(true)
      expect(r.cursor).toBe(2)
    })

    it('clamps at 0', () => {
      const r = processVimKey('h', makeKey(), 'hi', 0, 1)
      expect(r.cursor).toBe(0)
    })
  })

  describe('l — cursor right', () => {
    it('moves cursor right by one', () => {
      const r = processVimKey('l', makeKey(), 'hello', 1, 1)
      expect(r.consumed).toBe(true)
      expect(r.cursor).toBe(2)
    })

    it('clamps at input length', () => {
      const r = processVimKey('l', makeKey(), 'hi', 2, 1)
      expect(r.cursor).toBe(2)
    })
  })

  // ── Word movement: w / b ───────────────────────────────────────────

  describe('w — next word start', () => {
    it('moves to next word', () => {
      const r = processVimKey('w', makeKey(), 'hello world', 0, 1)
      expect(r.cursor).toBe(6)
    })

    it('stays at end for single word', () => {
      // wordForward from 0 to end of 'hello' = 5
      const r = processVimKey('w', makeKey(), 'hello', 0, 1)
      expect(r.cursor).toBe(5)
    })
  })

  describe('b — previous word start', () => {
    it('moves to start of current word when inside a word', () => {
      const r = processVimKey('b', makeKey(), 'hello world', 8, 1)
      expect(r.cursor).toBe(6)
    })

    it('stays at 0 on first word', () => {
      const r = processVimKey('b', makeKey(), 'hello world', 3, 1)
      expect(r.cursor).toBe(0)
    })
  })

  // ── Line movement: 0 / $ / ^ ───────────────────────────────────────

  describe('0 — line start', () => {
    it('moves cursor to 0', () => {
      const r = processVimKey('0', makeKey(), 'hello', 3, 1)
      expect(r.cursor).toBe(0)
    })
  })

  describe('$ — line end', () => {
    it('moves cursor to end', () => {
      const r = processVimKey('$', makeKey({ shift: true }), 'hello', 0, 1)
      expect(r.cursor).toBe(5)
    })
  })

  // ── Insert: i / a / I / A / o / O ─────────────────────────────────

  describe('i — insert before cursor', () => {
    it('enters insert mode at current cursor', () => {
      const r = processVimKey('i', makeKey(), 'hello', 2, 1)
      expect(r.consumed).toBe(true)
      expect(r.mode).toBe('insert')
      // Cursor unchanged
      expect(r.cursor).toBeUndefined()
    })

    it('returns expected shape', () => {
      const r = processVimKey('i', makeKey(), 'hello', 2, 1)
      expect(r).toEqual({ consumed: true, mode: 'insert' })
    })
  })

  describe('a — insert after cursor', () => {
    it('moves cursor right by 1 and enters insert', () => {
      const r = processVimKey('a', makeKey(), 'hello', 2, 1)
      expect(r.mode).toBe('insert')
      expect(r.cursor).toBe(3)
    })

    it('stays at end when at end of line', () => {
      const r = processVimKey('a', makeKey(), 'hi', 2, 1)
      expect(r.cursor).toBe(2)
      expect(r.mode).toBe('insert')
    })
  })

  describe('I — insert at start of line', () => {
    it('moves cursor to 0 and enters insert', () => {
      const r = processVimKey('I', makeKey({ shift: true }), 'hello', 4, 1)
      expect(r.mode).toBe('insert')
      expect(r.cursor).toBe(0)
    })
  })

  describe('A — insert at end of line', () => {
    it('moves cursor to end and enters insert', () => {
      const r = processVimKey('A', makeKey({ shift: true }), 'hello', 0, 1)
      expect(r.mode).toBe('insert')
      expect(r.cursor).toBe(5)
    })

    it('treats shifted a as A and moves to line end', () => {
      const r = processVimKey('a', makeKey({ shift: true }), 'hello', 1, 1)
      expect(r.mode).toBe('insert')
      expect(r.cursor).toBe(5)
    })
  })

  describe('o — new line below (auto-indent)', () => {
    it('appends newline with indent and enters insert', () => {
      const r = processVimKey('o', makeKey(), 'hello', 3, 1)
      expect(r.mode).toBe('insert')
      // Inserts '\n' + (cursor.col spaces) before end of current line
      expect(r.input).toBe('hello\n   ')
      // Cursor lands at end of the inserted line (after indent)
      expect(r.cursor).toBe(9)
    })
  })

  describe('O — new line above', () => {
    it('inserts newline at line start and enters insert', () => {
      const r = processVimKey('O', makeKey({ shift: true }), 'hello', 3, 1)
      expect(r.mode).toBe('insert')
      // Inserts '\n' at start of current line
      expect(r.input).toBe('\nhello')
      // Cursor is at position 1 (after the newline character)
      expect(r.cursor).toBe(1)
    })
  })

  // ── Delete: x / dd ─────────────────────────────────────────────────

  describe('x — delete character under cursor', () => {
    it('removes character at cursor', () => {
      const r = processVimKey('x', makeKey(), 'hello', 2, 1)
      expect(r.input).toBe('helo')
      expect(r.cursor).toBe(2)
    })

    it('consumes the key at end of input but does not modify text', () => {
      const r = processVimKey('x', makeKey(), 'hi', 2, 1)
      expect(r.consumed).toBe(true)
      expect(r.input).toBeUndefined()
    })
  })

  describe('dd — delete line', () => {
    it('does not delete on the first d', () => {
      const r = processVimKey('d', makeKey(), 'hello world', 3, 1)
      expect(r.consumed).toBe(true)
      expect(r.input).toBeUndefined()
      expect(r.cursor).toBeUndefined()
    })

    it('deletes entire content for single-line input', () => {
      processVimKey('d', makeKey(), 'hello world', 3, 1)
      const r = processVimKey('d', makeKey(), 'hello world', 3, 1)
      expect(r.consumed).toBe(true)
      expect(r.input).toBe('')
      expect(r.cursor).toBe(0)
    })

    it('deletes first line in multi-line input', () => {
      processVimKey('d', makeKey(), 'hello\nworld', 2, 1)
      const r = processVimKey('d', makeKey(), 'hello\nworld', 2, 1)
      expect(r.input).toBe('world')
      expect(r.cursor).toBe(0)
    })

    it('deletes middle line in multi-line input', () => {
      processVimKey('d', makeKey(), 'a\nb\nc', 3, 1)
      const r = processVimKey('d', makeKey(), 'a\nb\nc', 3, 1)
      expect(r.input).toBe('a\nc')
      // Cursor lands at start of previous line
      expect(r.cursor).toBe(0)
    })
  })

  // ── Yank / Paste: yy / p / P ──────────────────────────────────────

  describe('yy + p — yank line and paste below', () => {
    it('does not yank on the first y', () => {
      processVimKey('y', makeKey(), 'hello', 3, 1)
      const r = processVimKey('p', makeKey(), 'abc', 1, 1)
      expect(r.consumed).toBe(true)
      expect(r.input).toBeUndefined()
    })

    it('yanks the line and pastes it below cursor line', () => {
      yy('hello', 3)
      const r = processVimKey('p', makeKey(), 'abc', 1, 1)
      expect(r.consumed).toBe(true)
      // p does line-wise paste: inserts '\n' + yanked text at line end
      expect(r.input).toBe('abc\nhello')
      expect(r.cursor).toBe(9)
    })
  })

  describe('yy + P — paste above current line', () => {
    it('yanks a line and pastes it above cursor line', () => {
      yy('hello', 3)
      const r = processVimKey('P', makeKey({ shift: true }), 'abc', 1, 1)
      expect(r.consumed).toBe(true)
      // P does line-wise paste above: inserts yanked text + '\n' at line start
      expect(r.input).toBe('hello\nabc')
      expect(r.cursor).toBe(5)
    })
  })

  // ── Undo / Redo: u / Ctrl+r ────────────────────────────────────────

  describe('u — undo', () => {
    it('undoes a delete operation', () => {
      // Make a change (delete a char)
      const r1 = processVimKey('x', makeKey(), 'hello', 4, 1)
      expect(r1.input).toBe('hell')

      // Undo it
      const r2 = processVimKey('u', makeKey(), r1.input!, r1.cursor!, 1)
      expect(r2.consumed).toBe(true)
      expect(r2.input).toBe('hello')
    })

    it('undoes an insert-mode typing checkpoint and redoes it', () => {
      beginVimInsertSession()
      recordVimInsertEdit('')

      const undo = processVimKey('u', makeKey(), 'hello', 5, 1)
      expect(undo.consumed).toBe(true)
      expect(undo.input).toBe('')
      expect(undo.cursor).toBe(0)

      const redo = processVimKey('\u0012', { ...makeKey(), ctrl: true }, undo.input!, undo.cursor!, 1)
      expect(redo.consumed).toBe(true)
      expect(redo.input).toBe('hello')
      expect(redo.cursor).toBe(0)
    })
  })

  // ── Multi-line: j / k / gg / G / % ─────────────────────────────────

  describe('j — cursor down', () => {
    it('moves cursor to the next line', () => {
      const r = processVimKey('j', makeKey(), 'line1\nline2', 0, 1)
      // First line is 0-5, second is 6-10. Col 0 → next line col 0 = position 6
      expect(r.cursor).toBe(6)
    })

    it('stays on last line', () => {
      const r = processVimKey('j', makeKey(), 'line1\nline2', 8, 1)
      expect(r.cursor).toBe(8)
    })
  })

  describe('k — cursor up', () => {
    it('moves cursor to the previous line preserving column', () => {
      const r = processVimKey('k', makeKey(), 'line1\nline2', 8, 1)
      // Cursor 8 is at column 2 on line2 (li|ne2)
      // Previous line start=0, column 2 → position 2
      expect(r.cursor).toBe(2)
    })

    it('stays on first line', () => {
      const r = processVimKey('k', makeKey(), 'line1\nline2', 0, 1)
      expect(r.cursor).toBe(0)
    })
  })

  describe('gg — go to first line', () => {
    it('does not move on the first g', () => {
      const r = processVimKey('g', makeKey(), 'abc\ndef\nghi', 10, 1)
      expect(r.consumed).toBe(true)
      expect(r.cursor).toBeUndefined()
    })

    it('moves cursor to start of first line', () => {
      processVimKey('g', makeKey(), 'abc\ndef\nghi', 10, 1)
      const r = processVimKey('g', makeKey(), 'abc\ndef\nghi', 10, 1)
      expect(r.consumed).toBe(true)
      expect(r.cursor).toBe(0)
    })
  })

  describe('G — go to last line', () => {
    it('moves cursor to start of last line', () => {
      const r = processVimKey('G', makeKey({ shift: true }), 'abc\ndef\nghi', 0, 1)
      // Last line "ghi" starts at position 8
      expect(r.cursor).toBe(8)
    })
  })

  // ── Non-vim keys are not consumed ──────────────────────────────────

  describe('unrecognized keys', () => {
    it('does not consume Enter', () => {
      const r = processVimKey('\r', makeKey({ return: true }), 'hello', 0, 1)
      expect(r.consumed).toBe(false)
    })
  })

  // ── Undo/redo lifecycle ────────────────────────────────────────────

  describe('undo redo lifecycle', () => {
    it('pastes and undoes correctly', () => {
      yy('hello', 3)
      const r1 = processVimKey('p', makeKey(), 'abc', 1, 1)
      expect(r1.consumed).toBe(true)
      expect(r1.input).toBe('abc\nhello')

      // Undo the paste
      const r2 = processVimKey('u', makeKey(), r1.input!, r1.cursor!, 1)
      expect(r2.consumed).toBe(true)
      expect(r2.input).toBe('abc')
      // Cursor clamps to undo-state length (undo stack stores text only)
      expect(r2.cursor).toBe(3)
    })

    it('redo restores undone change', () => {
      yy('hello', 3)
      const r1 = processVimKey('p', makeKey(), 'abc', 1, 1)
      expect(r1.consumed).toBe(true)

      const r2 = processVimKey('u', makeKey(), r1.input!, r1.cursor!, 1)
      expect(r2.input).toBe('abc')

      // Ctrl+r = redo
      const r3 = processVimKey('\u0012', { ...makeKey(), ctrl: true }, r2.input!, r2.cursor!, 1)
      expect(r3.consumed).toBe(true)
      expect(r3.input).toBe('abc\nhello')
    })
  })
})
