import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React, { useState } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { getInputSelection } from '../app/inputSelectionStore.js'
import {
  applyVimCommand,
  TextInput,
  VIM_BADGE_WIDTH,
  vimBadgeColor,
  vimBadgeLabel,
  type VimCommandState,
  type VimInputMode
} from '../components/textInput.js'

class FakeInput extends EventEmitter {
  chunks: string[] = []
  isRaw = false
  isTTY = true
  readableLength = 0
  read() {
    const next = this.chunks.shift() ?? null
    this.readableLength = this.chunks.length

    return next
  }
  ref = vi.fn()
  send(...chunks: string[]) {
    this.chunks.push(...chunks)
    this.readableLength = this.chunks.length
    this.emit('readable')
  }
  setEncoding = vi.fn()
  setRawMode = vi.fn((enabled: boolean) => {
    this.isRaw = enabled
  })
  unref = vi.fn()
}

const settle = (ms = 25) => new Promise(resolve => setTimeout(resolve, ms))
const normal = (value: string, cursor = 0): VimCommandState => ({ cursor, mode: 'normal', pending: '', value })
const key = (escape = false, ctrl = false) => ({ ctrl, escape })

describe('TextInput Vim command reducer', () => {
  it('implements modes, motions, edits, undo and redo routing', () => {
    const entered = applyVimCommand({ cursor: 11, mode: 'insert', pending: '', value: 'hello world' }, '', key(true))
    expect(entered).toMatchObject({ cursor: 10, handled: true, mode: 'normal' })
    expect(applyVimCommand(entered, '0', key()).cursor).toBe(0)
    expect(applyVimCommand(normal('hello world'), 'w', key()).cursor).toBe(6)
    expect(applyVimCommand(normal('hello world', 10), 'b', key()).cursor).toBe(6)
    expect(applyVimCommand(normal('hello\nworld', 1), 'j', key()).cursor).toBe(7)
    expect(applyVimCommand(normal('abc', 1), 'x', key())).toMatchObject({ value: 'ac' })
    expect(applyVimCommand(normal('abc def', 3), 'D', key())).toMatchObject({ value: 'abc' })
    expect(applyVimCommand(normal('abc def', 3), 'C', key())).toMatchObject({ mode: 'insert', value: 'abc' })
    expect(applyVimCommand(normal('abc', 2), 'u', key()).action).toBe('undo')
    expect(applyVimCommand(normal('abc', 2), 'r', key(false, true)).action).toBe('redo')
  })

  it('opens a new line below with o and enters insert mode', () => {
    expect(applyVimCommand(normal('one\ntwo', 1), 'o', key())).toMatchObject({
      cursor: 4,
      mode: 'insert',
      value: 'one\n\ntwo'
    })
    expect(applyVimCommand(normal('one', 1), 'o', key())).toMatchObject({
      cursor: 4,
      mode: 'insert',
      value: 'one\n'
    })
  })

  it('opens a new line above with O and enters insert mode', () => {
    expect(applyVimCommand(normal('one\ntwo', 5), 'O', key())).toMatchObject({
      cursor: 4,
      mode: 'insert',
      value: 'one\n\ntwo'
    })
  })

  it('replaces one grapheme with r{char} and overwrites in R mode', () => {
    const pending = applyVimCommand(normal('a👍b', 1), 'r', key())
    expect(pending.pending).toBe('r')
    expect(applyVimCommand(pending, 'Z', key())).toMatchObject({ cursor: 1, value: 'aZb' })

    const replace = applyVimCommand(normal('abc', 1), 'R', key())
    expect(replace.mode).toBe('replace')
    expect(applyVimCommand(replace, 'XY', key())).toMatchObject({ cursor: 3, mode: 'replace', value: 'aXY' })
  })

  it('finds the next character on the current line with f{char}', () => {
    const pending = applyVimCommand(normal('abc def abc', 0), 'f', key())
    expect(pending.pending).toBe('f')
    expect(applyVimCommand(pending, 'd', key())).toMatchObject({ cursor: 4, pending: '' })

    const missing = applyVimCommand(normal('abc\ndef', 0), 'f', key())
    expect(applyVimCommand(missing, 'd', key())).toMatchObject({ cursor: 0, pending: '' })
  })

  it('composes d with motions', () => {
    const dw = applyVimCommand(applyVimCommand(normal('one two', 0), 'd', key()), 'w', key())
    expect(dw).toMatchObject({ cursor: 0, pending: '', value: 'two' })

    const dDollar = applyVimCommand(applyVimCommand(normal('one two', 4), 'd', key()), '$', key())
    expect(dDollar).toMatchObject({ cursor: 3, pending: '', value: 'one ' })

    const dZero = applyVimCommand(applyVimCommand(normal('one two', 5), 'd', key()), '0', key())
    expect(dZero).toMatchObject({ cursor: 0, pending: '', value: 'wo' })
  })

  it('selects with visual mode and deletes the inclusive range with d', () => {
    const visual = applyVimCommand(normal('one two', 0), 'v', key())
    expect(visual).toMatchObject({ anchor: 0, mode: 'visual' })

    const moved = applyVimCommand(visual, 'w', key())
    expect(moved).toMatchObject({ anchor: 0, cursor: 4, mode: 'visual' })
    expect(applyVimCommand(moved, 'd', key())).toMatchObject({
      anchor: undefined,
      cursor: 0,
      mode: 'normal',
      value: 'wo'
    })
  })

  it('leaves visual mode with Escape without changing text', () => {
    const visual = applyVimCommand(normal('abc', 1), 'v', key())
    expect(applyVimCommand(visual, '', key(true))).toMatchObject({
      anchor: undefined,
      cursor: 1,
      mode: 'normal',
      value: 'abc'
    })
  })

  it('deletes a complete logical line with dd', () => {
    const pending = applyVimCommand(normal('one\ntwo\nthree', 5), 'd', key())
    expect(pending.pending).toBe('d')
    expect(applyVimCommand(pending, 'd', key())).toMatchObject({ cursor: 4, pending: '', value: 'one\nthree' })
  })

  it('keeps the normal-mode cursor off the newline separator', () => {
    // "ab\ncd": l from the last character of a line must not step onto index 2.
    expect(applyVimCommand(normal('ab\ncd', 1), 'l', key()).cursor).toBe(1)
    // Leaving insert mode at the line break clamps back onto the character.
    expect(applyVimCommand({ cursor: 2, mode: 'insert', pending: '', value: 'ab\ncd' }, '', key(true)).cursor).toBe(1)
    // So neither x nor D can join two logical lines.
    expect(applyVimCommand(normal('ab\ncd', 1), 'x', key()).value).toBe('a\ncd')
    expect(applyVimCommand(normal('ab\ncd', 1), 'D', key()).value).toBe('a\ncd')
  })

  it('treats x and D on an empty line as no-ops', () => {
    // "ab\n\ncd": index 3 is the empty line; its only position IS the newline,
    // so deleting there would join lines rather than remove a character.
    expect(applyVimCommand(normal('ab\n\ncd', 3), 'x', key()).value).toBe('ab\n\ncd')
    expect(applyVimCommand(normal('ab\n\ncd', 3), 'D', key()).value).toBe('ab\n\ncd')
    expect(applyVimCommand(normal('ab\n\ncd', 3), 'l', key()).cursor).toBe(3)
  })

  it('cancels a pending d when a special key falls through', () => {
    const pending = applyVimCommand(normal('one\ntwo\nthree', 5), 'd', key())
    expect(pending.pending).toBe('d')

    // An arrow / Enter arrives: no input string and not Escape. The composer
    // still owns the key (handled: false), but the operator is cancelled so a
    // later `d` cannot pair with it and delete an unrelated line.
    const special = applyVimCommand(pending, '', key())
    expect(special).toMatchObject({ handled: false, pending: '' })
  })

  it('changes with cw ce c$ and cc and enters insert mode', () => {
    const run = (value: string, cursor: number, ...commands: string[]) =>
      commands.reduce((state, command) => applyVimCommand(state, command, key()), normal(value, cursor))

    expect(run('one two', 0, 'c', 'w')).toMatchObject({ mode: 'insert', value: ' two' })
    expect(run('one two three', 0, 'c', '2', 'w')).toMatchObject({ mode: 'insert', value: ' three' })
    expect(run('one two three', 0, '2', 'c', 'w')).toMatchObject({ mode: 'insert', value: ' three' })
    expect(run('one two', 0, 'c', 'e')).toMatchObject({ mode: 'insert', value: ' two' })
    expect(run('one two', 4, 'c', '$')).toMatchObject({ mode: 'insert', value: 'one ' })
    expect(run('one\ntwo', 0, 'c', 'c')).toMatchObject({ mode: 'insert', value: '\ntwo' })
  })

  it('updates and pastes the unnamed register with charwise and linewise edits and yanks', () => {
    const cut = applyVimCommand(normal('abc', 1), 'x', key())
    expect(cut.register).toEqual({ linewise: false, text: 'b' })
    expect(applyVimCommand(cut, 'p', key()).value).toBe('acb')

    const yy = applyVimCommand(applyVimCommand(normal('one\ntwo', 0), 'y', key()), 'y', key())
    expect(yy.register).toEqual({ linewise: true, text: 'one\n' })
    expect(applyVimCommand(yy, 'p', key()).value).toBe('one\none\ntwo')
  })

  it('supports first-nonblank and counted document motions', () => {
    expect(applyVimCommand(normal('  one', 4), '^', key()).cursor).toBe(2)
    const lines = 'one\n  two\nthree'
    expect(applyVimCommand(applyVimCommand(normal(lines, 8), 'g', key()), 'g', key()).cursor).toBe(0)
    expect(applyVimCommand(normal(lines), 'G', key()).cursor).toBe(10)
    expect(applyVimCommand(applyVimCommand(normal(lines), '2', key()), 'G', key()).cursor).toBe(6)
  })

  it('supports directional find, till, and repeats bounded to a logical line', () => {
    const run = (...commands: string[]) =>
      commands.reduce((s, c) => applyVimCommand(s, c, key()), normal('a-b-b\na-b', 0))

    expect(run('f', 'b').cursor).toBe(2)
    expect(run('t', 'b').cursor).toBe(1)
    expect(run('$', 'F', 'a').cursor).toBe(0)
    expect(run('f', 'b', ';').cursor).toBe(4)
    expect(run('f', 'b', ';', ';').cursor).toBe(4)
    expect(run('f', 'b', ';', ';', ';').cursor).toBe(4)
    expect(run('f', 'b', ';', ',').cursor).toBe(2)
  })

  it('applies motion and operator counts multiplicatively', () => {
    const run = (value: string, ...commands: string[]) =>
      commands.reduce((s, c) => applyVimCommand(s, c, key()), normal(value))

    expect(run('a b c d', '3', 'w').cursor).toBe(6)
    expect(run('a\nb\nc\nd\ne\nf', '5', 'j').cursor).toBe(10)
    expect(run('a\nb\nc\nd', '2', 'd', 'd').value).toBe('c\nd')
    expect(run('a b c d e f g', 'd', '3', 'w').value).toBe('d e f g')
    expect(run('a b c d e f g', '3', 'd', 'w').value).toBe('d e f g')
    expect(run('a b c d e f g', '2', 'd', '3', 'w').value).toBe('g')
    expect(run('abc\ndef\nghi', 'd', '2', '$').value).toBe('ghi')
    expect(run('abc\ndef\nghi', '2', '$').cursor).toBe(6)
  })

  it('supports iw and aw text objects for operators and visual mode', () => {
    const run = (value: string, cursor: number, ...commands: string[]) =>
      commands.reduce((s, c) => applyVimCommand(s, c, key()), normal(value, cursor))

    expect(run('one two three', 5, 'd', 'i', 'w').value).toBe('one  three')
    expect(run('one two three', 5, 'd', 'a', 'w').value).toBe('one three')
    expect(run('one two', 5, 'v', 'i', 'w', 'y').register).toEqual({ linewise: false, text: 'two' })
  })

  it('repeats successful non-insert changes with dot', () => {
    const changed = applyVimCommand(normal('abcd', 0), 'x', key())
    expect(applyVimCommand(changed, '.', key()).value).toBe('cd')
  })

  it('repeats visual edits with their original extent', () => {
    const run = (state: VimCommandState, ...commands: string[]) =>
      commands.reduce((s, c) => applyVimCommand(s, c, key()), state)

    const changed = run(normal('abcdef'), 'v', 'l', 'd')
    expect(changed.value).toBe('cdef')
    expect(run(changed, '.')).toMatchObject({ mode: 'normal', pending: '', value: 'ef' })

    // Linewise visual edits repeat as a line count, not as the V keystroke.
    const lines = run(normal('a\nb\nc\nd\ne'), 'V', 'j', 'd')
    expect(lines.value).toBe('c\nd\ne')
    expect(run(lines, '.').value).toBe('e')

    // A visual yank leaves nothing to repeat, exactly as in Vim.
    const yanked = run(normal('abcdef'), 'v', 'l', 'y')
    expect(run(yanked, '.').value).toBe('abcdef')

    // A visual change repeats the extent AND the text typed into it.
    const visualChange = run(normal('abcdef'), 'v', 'l', 'c')
    expect(visualChange).toMatchObject({ mode: 'insert', value: 'cdef' })
    const closed = applyVimCommand({ ...visualChange, cursor: 1, value: 'Zcdef' }, '', key(true))
    expect(applyVimCommand({ ...closed, cursor: 1 }, '.', key()).value).toBe('ZZef')
  })

  it('finalizes an insert session on Escape so dot repeats the whole change', () => {
    const opened = applyVimCommand(normal('ab', 0), 'o', key())
    expect(opened).toMatchObject({ cursor: 3, mode: 'insert', value: 'ab\n' })

    // The composer owns typing in insert mode; it hands the reducer the value
    // and cursor it produced. Escape closes the session.
    const typed = { ...opened, cursor: 5, value: 'ab\nXY' }
    const closed = applyVimCommand(typed, '', key(true))
    expect(closed).toMatchObject({ mode: 'normal' })
    expect(applyVimCommand(closed, '.', key())).toMatchObject({ mode: 'normal', value: 'ab\nXY\nXY' })

    const changing = applyVimCommand(applyVimCommand(normal('one two', 0), 'c', key()), 'w', key())
    const replaced = applyVimCommand({ ...changing, cursor: 3, value: 'ZZZ two' }, '', key(true))
    expect(applyVimCommand({ ...replaced, cursor: 4 }, '.', key()).value).toBe('ZZZ ZZZ')
  })

  it('selects and edits complete lines with visual-line mode', () => {
    const selected = applyVimCommand(applyVimCommand(normal('one\ntwo\nthree', 1), 'V', key()), 'j', key())
    expect(selected).toMatchObject({ anchor: 1, cursor: 5, mode: 'visual-line' })
    expect(applyVimCommand(selected, 'd', key())).toMatchObject({ mode: 'normal', value: 'three' })
  })

  it('saturates adversarial counts and supports counted replace and dot', () => {
    const run = (value: string, ...commands: string[]) =>
      commands.reduce((s, c) => applyVimCommand(s, c, key()), normal(value))

    const huge = '9'.repeat(400)
    const moved = run('abc', ...huge, 'l')
    expect(moved.cursor).toBe(2)
    expect(moved.count).toBe('')
    expect(run('abcdef', '3', 'r', 'X').value).toBe('XXXdef')
    const combining = `a${'\u0301'.repeat(100)}`
    expect(run('x'.repeat(10_000), ...'10000', 'r', combining).value.length).toBeLessThanOrEqual(20_000)

    const changed = run('abcdef', 'x')
    expect(applyVimCommand(applyVimCommand(changed, '3', key()), '.', key()).value).toBe('ef')
  })

  it('replays deleting changes when the buffer already exceeds the growth cap', () => {
    const original = 'x'.repeat(10_002)
    const changed = applyVimCommand(normal(original), 'x', key())
    const repeated = applyVimCommand(changed, '.', key())

    expect(repeated.value).toBe('x'.repeat(10_000))
  })

  it('bounds counted put output by register size while preserving reasonable counts', () => {
    const put = (register: string, count: string) => {
      let state: VimCommandState = { ...normal('a'), register: { linewise: false, text: register } }

      for (const command of [...count, 'p']) {
        state = applyVimCommand(state, command, key())
      }

      return state
    }

    expect(put('bc', '3').value).toBe('abcbcbc')
    expect(put('x'.repeat(100), '10000').value.length).toBeLessThanOrEqual(10_001)

    const full: VimCommandState = {
      ...normal('x'.repeat(10_000)),
      register: { linewise: true, text: 'line\n' }
    }

    expect(applyVimCommand(full, 'p', key()).value).toBe(full.value)
  })

  it('caps total dot replay growth across nested counts', () => {
    let state: VimCommandState = { ...normal('a'), register: { linewise: false, text: 'b' } }

    for (const command of [...'10000', 'p', ...'10000', '.']) {
      state = applyVimCommand(state, command, key())
    }

    expect(state.value.length).toBeLessThanOrEqual(10_001)

    const replacement = '👨‍👩‍👧‍👦'

    const replaced = applyVimCommand(
      { ...normal('x'.repeat(20_000)), lastChange: ['100', 'r', replacement], count: '10000' },
      '.',
      key()
    )

    expect(replaced.value.length).toBeLessThanOrEqual(30_000)
  }, 500)

  it('keeps linewise registers canonical at EOF and pastes complete lines', () => {
    const run = (state: VimCommandState, ...commands: string[]) =>
      commands.reduce((s, c) => applyVimCommand(s, c, key()), state)

    const finalYank = run(normal('one\ntwo', 4), 'y', 'y')
    expect(finalYank.register).toEqual({ linewise: true, text: 'two\n' })
    expect(run(finalYank, 'p').value).toBe('one\ntwo\ntwo')
    expect(run(finalYank, 'P').value).toBe('one\ntwo\ntwo')

    const singleYank = run(normal('one'), 'y', 'y')
    expect(singleYank.register).toEqual({ linewise: true, text: 'one\n' })
    expect(run(singleYank, 'p').value).toBe('one\none')
    expect(run(singleYank, 'P').value).toBe('one\none')

    const finalDelete = run(normal('one\ntwo', 4), 'd', 'd')
    expect(finalDelete).toMatchObject({ cursor: 2, value: 'one' })
    expect(finalDelete.register).toEqual({ linewise: true, text: 'two\n' })
    expect(run(normal('one'), 'd', 'd')).toMatchObject({ cursor: 0, value: '' })
  })

  it('leaves an empty replacement line for cc and visual-line c', () => {
    const run = (state: VimCommandState, ...commands: string[]) =>
      commands.reduce((s, c) => applyVimCommand(s, c, key()), state)

    const first = run(normal('one\ntwo'), 'c', 'c')
    expect(first).toMatchObject({ cursor: 0, mode: 'insert', value: '\ntwo' })
    const firstTyped = applyVimCommand({ ...first, cursor: 1, value: 'X\ntwo' }, '', key(true))
    expect(firstTyped.value).toBe('X\ntwo')

    const final = run(normal('one\ntwo', 4), 'c', 'c')
    expect(final).toMatchObject({ cursor: 4, mode: 'insert', value: 'one\n' })
    const finalTyped = applyVimCommand({ ...final, cursor: 5, value: 'one\nX' }, '', key(true))
    expect(finalTyped.value).toBe('one\nX')

    const visual = run(normal('one\ntwo', 4), 'V', 'c')
    expect(visual).toMatchObject({ cursor: 4, mode: 'insert', value: 'one\n' })

    const onlyLine = run(normal('one'), 'V', 'c')
    expect(onlyLine).toMatchObject({ cursor: 0, mode: 'insert', value: '' })

    const onlyLineCc = run(normal('one'), 'c', 'c')
    expect(onlyLineCc).toMatchObject({ cursor: 0, mode: 'insert', value: '' })
  })

  it('replays replace edits after cursor movement and backspace', () => {
    const opened = applyVimCommand(normal('abcd'), 'R', key())
    const typed = applyVimCommand(opened, 'XY', key())
    const closedAfterLeft = applyVimCommand({ ...typed, cursor: 1 }, '', key(true))
    expect(applyVimCommand({ ...closedAfterLeft, cursor: 2 }, '.', key()).value).toBe('XYXY')

    const backed = applyVimCommand({ ...typed, cursor: 1, value: 'Xcd' }, '', key(true))
    expect(applyVimCommand({ ...backed, cursor: 1 }, '.', key()).value).toBe('XXd')
  })

  it('does not skip adjacent targets when repeating t and T', () => {
    const run = (value: string, cursor: number, ...commands: string[]) =>
      commands.reduce((s, c) => applyVimCommand(s, c, key()), normal(value, cursor))

    expect(run('abb', 0, 't', 'b', ';').cursor).toBe(1)
    expect(run('bba', 2, 'T', 'b', ';').cursor).toBe(1)
    expect(run('a-b-b-c', 0, 't', 'b', ';').cursor).toBe(3)
    expect(run('a-b-b-c', 6, 'T', 'b', ';').cursor).toBe(3)
  })

  it('counts text objects and finds whole graphemes', () => {
    const run = (value: string, cursor: number, ...commands: string[]) =>
      commands.reduce((s, c) => applyVimCommand(s, c, key()), normal(value, cursor))

    expect(run('one two three four five', 0, 'd', '2', 'i', 'w').value).toBe(' three four five')
    const changed = run('one two three four five', 0, 'd', '2', 'i', 'w')
    expect(applyVimCommand(changed, '.', key()).value).toBe('  five')
    expect(run('a👍🏽b👍🏽c', 0, 'f', '👍🏽').cursor).toBe(1)
  })

  it('labels every mode for the composer badge in a fixed-width cell', () => {
    expect(vimBadgeLabel('insert')).toBe('INSERT ')
    expect(vimBadgeLabel('normal')).toBe('NORMAL ')
    expect(vimBadgeLabel('visual')).toBe('VISUAL ')
    expect(vimBadgeLabel('visual-line')).toBe('V-LINE ')
    expect(vimBadgeLabel('replace')).toBe('REPLACE')

    for (const mode of ['insert', 'normal', 'visual', 'visual-line', 'replace'] as VimInputMode[]) {
      expect(vimBadgeLabel(mode)).toHaveLength(VIM_BADGE_WIDTH)
    }
  })

  it('gives visual selection modes a distinct accent badge color', () => {
    expect(vimBadgeColor('insert')).toBe('ok')
    expect(vimBadgeColor('normal')).toBe('warn')
    expect(vimBadgeColor('replace')).toBe('warn')
    expect(vimBadgeColor('visual')).toBe('accent')
    expect(vimBadgeColor('visual-line')).toBe('accent')
  })
})

describe('TextInput Vim integration', () => {
  it('treats one emoji grapheme as an f target without mutating the buffer', async () => {
    const stdin = new FakeInput()
    const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const stderr = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const changes: string[] = []

    function Harness() {
      const [value, setValue] = useState('a😀b😀c')

      return (
        <TextInput
          onChange={next => {
            changes.push(next)
            setValue(next)
          }}
          value={value}
          vim
        />
      )
    }

    const view = renderSync(<Harness />, {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    })

    await settle()
    stdin.send('\x1b')
    await settle(100)
    stdin.send('0', 'f', '😀')
    await settle()
    expect(changes).toEqual([])
    expect(getInputSelection()).toMatchObject({ start: 1, end: 1, value: 'a😀b😀c' })

    view.unmount()
    view.cleanup()
  })

  it('cancels visual state before bracketed paste and IME input', async () => {
    const stdin = new FakeInput()
    const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const stderr = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const modes: VimInputMode[] = []
    const changes: string[] = []

    function Harness() {
      const [value, setValue] = useState('abc')

      return (
        <TextInput
          onChange={next => {
            changes.push(next)
            setValue(next)
          }}
          onVimModeChange={mode => modes.push(mode)}
          value={value}
          vim
        />
      )
    }

    const view = renderSync(<Harness />, {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    })

    await settle()
    stdin.send('\x1b')
    await settle(100)
    stdin.send('0', 'v', 'l', '\x1b[200~Z\x1b[201~')
    await settle()
    expect(modes.at(-1)).toBe('normal')
    expect(changes.at(-1)).toBe('aZbc')

    stdin.send('v', '漢字')
    await settle()
    expect(modes.at(-1)).toBe('normal')
    expect(changes.at(-1)).toBe('aZ漢字bc')
    view.unmount()
    view.cleanup()
  })

  it('preserves insert and replace modes for paste and IME commits', async () => {
    const stdin = new FakeInput()
    const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const stderr = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const modes: VimInputMode[] = []
    const changes: string[] = []

    function Harness() {
      const [value, setValue] = useState('abc')

      return (
        <TextInput
          onChange={next => {
            changes.push(next)
            setValue(next)
          }}
          onVimModeChange={mode => modes.push(mode)}
          value={value}
          vim
        />
      )
    }

    const view = renderSync(<Harness />, {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    })

    await settle()
    stdin.send('漢字')
    await settle()
    expect(modes.at(-1)).toBe('insert')
    expect(changes.at(-1)).toBe('abc漢字')

    stdin.send('\x1b[200~ZZ\x1b[201~')
    await settle()
    expect(modes.at(-1)).toBe('insert')
    expect(changes.at(-1)).toBe('abc漢字ZZ')

    stdin.send('\x1b')
    await settle(100)
    stdin.send('0', 'R', '😀')
    await settle()
    expect(modes.at(-1)).toBe('replace')
    expect(changes.at(-1)).toBe('😀bc漢字ZZ')

    view.unmount()
    view.cleanup()
  })

  it('cancels transient Vim state when the controlled value is replaced externally', async () => {
    const stdin = new FakeInput()
    const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const stderr = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const modes: VimInputMode[] = []
    let replace!: (value: string) => void

    function Harness() {
      const [value, setValue] = useState('abc')
      replace = setValue

      return <TextInput onChange={setValue} onVimModeChange={mode => modes.push(mode)} value={value} vim />
    }

    const view = renderSync(<Harness />, {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    })

    await settle()
    stdin.send('\x1b')
    await settle(100)
    stdin.send('0', 'v', 'l')
    await settle()
    replace('external')
    await settle()
    expect(modes.at(-1)).toBe('normal')
    expect(getInputSelection()).toMatchObject({ start: 8, end: 8 })
    stdin.send('d')
    await settle()
    expect(getInputSelection()?.value).toBe('external')
    view.unmount()
    view.cleanup()
  })

  it('leaves normal typing unchanged when Vim mode is disabled', async () => {
    const stdin = new FakeInput()
    const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const stderr = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const changes: string[] = []

    function Harness() {
      const [value, setValue] = useState('')

      return (
        <TextInput
          onChange={next => {
            changes.push(next)
            setValue(next)
          }}
          value={value}
        />
      )
    }

    const view = renderSync(React.createElement(Harness), {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    })

    await settle()
    stdin.send('hjkli')
    await settle()
    expect(changes.at(-1)).toBe('hjkli')

    view.unmount()
    view.cleanup()
  })

  it('switches to normal mode with Escape and edits the actual controlled buffer', async () => {
    const stdin = new FakeInput()
    const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const stderr = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const modes: VimInputMode[] = []
    const changes: string[] = []

    function Harness() {
      const [value, setValue] = useState('abc')

      return (
        <TextInput
          columns={80}
          onChange={next => {
            changes.push(next)
            setValue(next)
          }}
          onVimModeChange={mode => modes.push(mode)}
          value={value}
          vim
        />
      )
    }

    const view = renderSync(React.createElement(Harness), {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    })

    await settle()
    stdin.send('\x1b')
    // Bare Escape is held briefly to disambiguate Alt-prefixed sequences.
    await settle(100)
    expect(modes.at(-1)).toBe('normal')
    stdin.send('0', 'x')
    await settle()
    expect(changes.at(-1)).toBe('bc')
    expect(getInputSelection()?.start).toBe(0)
    stdin.send('i', 'Z')
    await settle()
    expect(modes.at(-1)).toBe('insert')
    expect(changes.at(-1)).toBe('Zbc')

    view.unmount()
    view.cleanup()
  })

  it('executes o, f, operator motions, and visual deletion in the controlled buffer', async () => {
    const stdin = new FakeInput()
    const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const stderr = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const changes: string[] = []

    function Harness() {
      const [value, setValue] = useState('one two')

      return (
        <TextInput
          columns={80}
          onChange={next => {
            changes.push(next)
            setValue(next)
          }}
          value={value}
          vim
        />
      )
    }

    const view = renderSync(React.createElement(Harness), {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    })

    await settle()
    stdin.send('\x1b')
    await settle(100)
    stdin.send('0', 'f', 't', 'd', '$')
    await settle()
    expect(changes.at(-1)).toBe('one ')

    // 0ve selects the whole first word inclusively, so d removes "one" and
    // leaves the trailing space. (Before the vimStateRef fix the anchor was
    // dropped between keystrokes and this deleted a single character.)
    stdin.send('0', 'v', 'e', 'd', 'o', 'Z')
    await settle()
    expect(changes.at(-1)).toBe(' \nZ')

    view.unmount()
    view.cleanup()
  })

  it('keeps a normal-mode Right Arrow on the current line', async () => {
    const stdin = new FakeInput()
    const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const stderr = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })

    const view = renderSync(<TextInput columns={80} onChange={() => {}} value={'ab\ncd'} vim />, {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    })

    await settle()
    stdin.send('\x1b')
    await settle(100)
    stdin.send('0', 'k', '$', '\x1b[C')
    await settle()
    expect(getInputSelection()?.start).toBe(1)

    view.unmount()
    view.cleanup()
  })

  it.each([
    ['Left Arrow', '\x1b[D', 4],
    ['Right Arrow', '\x1b[C', 6],
    ['Up Arrow', '\x1b[A', 1],
    ['Down Arrow', '\x1b[B', 5],
    ['Home', '\x1b[H', 0],
    ['End', '\x1b[F', 7]
  ])('cancels visual mode before fallback %s movement', async (_name, sequence, expectedCursor) => {
    const stdin = new FakeInput()
    const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const stderr = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const modes: VimInputMode[] = []
    const changes: string[] = []

    function Harness() {
      const [value, setValue] = useState('abc\ndef')

      return (
        <TextInput
          columns={80}
          onChange={next => {
            changes.push(next)
            setValue(next)
          }}
          onVimModeChange={mode => modes.push(mode)}
          value={value}
          vim
        />
      )
    }

    const view = renderSync(<Harness />, {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    })

    await settle()
    stdin.send('\x1b')
    await settle(100)
    stdin.send('0', 'v', 'l')
    await settle()
    expect(modes.at(-1)).toBe('visual')
    expect(getInputSelection()).toMatchObject({ end: 6, start: 4 })

    stdin.send(sequence)
    await settle()
    expect(modes.at(-1)).toBe('normal')
    expect(getInputSelection()).toMatchObject({ end: expectedCursor, start: expectedCursor, value: 'abc\ndef' })

    stdin.send('d')
    await settle()
    expect(changes).toEqual([])

    view.unmount()
    view.cleanup()
  })

  it('cancels pending d before a modified special key falls through', async () => {
    const stdin = new FakeInput()
    const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const stderr = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const changes: string[] = []

    function Harness() {
      const [value, setValue] = useState('one\ntwo')

      return (
        <TextInput
          columns={80}
          onChange={next => {
            changes.push(next)
            setValue(next)
          }}
          value={value}
          vim
        />
      )
    }

    const view = renderSync(React.createElement(Harness), {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    })

    await settle()
    stdin.send('\x1b')
    await settle(100)
    stdin.send('d', '\x1b[1;2D', 'd')
    await settle()
    expect(changes).toEqual([])

    view.unmount()
    view.cleanup()
  })

  it('persists the visual anchor and register across keystrokes in the mounted composer', async () => {
    const stdin = new FakeInput()
    const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const stderr = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const modes: VimInputMode[] = []
    const changes: string[] = []

    function Harness() {
      const [value, setValue] = useState('abcdef')

      return (
        <TextInput
          columns={80}
          onChange={next => {
            changes.push(next)
            setValue(next)
          }}
          onVimModeChange={mode => modes.push(mode)}
          value={value}
          vim
        />
      )
    }

    const view = renderSync(React.createElement(Harness), {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    })

    await settle()
    stdin.send('\x1b')
    await settle(100)
    stdin.send('0', 'v', 'l', 'l')
    await settle()
    expect(modes.at(-1)).toBe('visual')
    // The anchor must survive the two motion keystrokes: the selection is the
    // inclusive range 0..2, not a single character re-anchored at the cursor.
    expect(getInputSelection()).toMatchObject({ end: 3, start: 0 })
    stdin.send('d')
    await settle()
    expect(changes.at(-1)).toBe('def')
    expect(modes.at(-1)).toBe('normal')

    // The delete filled the unnamed register; p must paste it back.
    stdin.send('p')
    await settle()
    expect(changes.at(-1)).toBe('dabcef')

    view.unmount()
    view.cleanup()
  })

  it('reports V-LINE and REPLACE modes to the badge from the mounted composer', async () => {
    const stdin = new FakeInput()
    const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const stderr = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const modes: VimInputMode[] = []
    const changes: string[] = []

    function Harness() {
      const [value, setValue] = useState('one\ntwo')

      return (
        <TextInput
          columns={80}
          onChange={next => {
            changes.push(next)
            setValue(next)
          }}
          onVimModeChange={mode => modes.push(mode)}
          value={value}
          vim
        />
      )
    }

    const view = renderSync(React.createElement(Harness), {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    })

    await settle()
    stdin.send('\x1b')
    await settle(100)
    stdin.send('g', 'g', 'V')
    await settle()
    expect(modes.at(-1)).toBe('visual-line')
    stdin.send('d')
    await settle()
    expect(changes.at(-1)).toBe('two')

    stdin.send('R')
    await settle()
    expect(modes.at(-1)).toBe('replace')
    stdin.send('X')
    await settle()
    expect(changes.at(-1)).toBe('Xwo')

    view.unmount()
    view.cleanup()
  })

  it('repeats an insert session typed in the mounted composer with dot', async () => {
    const stdin = new FakeInput()
    const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const stderr = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const changes: string[] = []

    function Harness() {
      const [value, setValue] = useState('ab')

      return (
        <TextInput
          columns={80}
          onChange={next => {
            changes.push(next)
            setValue(next)
          }}
          value={value}
          vim
        />
      )
    }

    const view = renderSync(React.createElement(Harness), {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    })

    await settle()
    stdin.send('\x1b')
    await settle(100)
    stdin.send('o')
    await settle()
    stdin.send('X')
    await settle()
    expect(changes.at(-1)).toBe('ab\nX')
    stdin.send('\x1b')
    await settle(100)
    stdin.send('.')
    await settle()
    expect(changes.at(-1)).toBe('ab\nX\nX')

    view.unmount()
    view.cleanup()
  })
})
