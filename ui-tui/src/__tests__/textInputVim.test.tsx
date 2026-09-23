import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React, { useState } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { getInputSelection } from '../app/inputSelectionStore.js'
import { applyVimCommand, TextInput, type VimCommandState, type VimInputMode } from '../components/textInput.js'

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
  setRawMode = vi.fn((enabled: boolean) => { this.isRaw = enabled })
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

  it('deletes a complete logical line with dd', () => {
    const pending = applyVimCommand(normal('one\ntwo\nthree', 5), 'd', key())
    expect(pending.pending).toBe('d')
    expect(applyVimCommand(pending, 'd', key())).toMatchObject({ cursor: 4, pending: '', value: 'one\nthree' })
  })

  it('keeps the normal-mode cursor off the newline separator', () => {
    // "ab\ncd": l from the last character of a line must not step onto index 2.
    expect(applyVimCommand(normal('ab\ncd', 1), 'l', key()).cursor).toBe(1)
    // Leaving insert mode at the line break clamps back onto the character.
    expect(
      applyVimCommand({ cursor: 2, mode: 'insert', pending: '', value: 'ab\ncd' }, '', key(true)).cursor
    ).toBe(1)
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
})

describe('TextInput Vim integration', () => {
  it('leaves normal typing unchanged when Vim mode is disabled', async () => {
    const stdin = new FakeInput()
    const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const stderr = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const changes: string[] = []

    function Harness() {
      const [value, setValue] = useState('')

      return <TextInput onChange={next => { changes.push(next); setValue(next) }} value={value} />
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
          onChange={next => { changes.push(next); setValue(next) }}
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
})
