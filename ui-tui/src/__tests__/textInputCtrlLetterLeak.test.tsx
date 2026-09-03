import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React, { useState } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { TextInput } from '../components/textInput.js'

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

const settle = (ms = 0) => new Promise(resolve => setTimeout(resolve, ms))

function makeStreams() {
  const stdin = new FakeInput()
  const stdout = new PassThrough()
  const stderr = new PassThrough()

  Object.assign(stdout, { columns: 80, isTTY: false, rows: 24 })
  Object.assign(stderr, { columns: 80, isTTY: false, rows: 24 })

  return { stderr, stdin, stdout }
}

// The dashboard gateway writes \x0c (Ctrl+L) to the chat PTY on every
// reattach force-redraw (hermes_cli/pty_session.py TUI_FORCE_REDRAW).
// hermes-ink parses it as {name: 'l', ctrl: true} and InputEvent surfaces
// input='l'; without the ctrl guard in the composer's insert gate a literal
// "l" lands at the cursor (#101393: every model-switcher reload appended
// one "l" to the chat input).
describe('TextInput ctrl-letter leak', () => {
  it('does not insert a literal letter for the reattach force-redraw byte \\x0c (ctrl+l)', async () => {
    const streams = makeStreams()
    const changes: string[] = []

    function Harness() {
      const [value, setValue] = useState('')

      return (
        <TextInput
          columns={80}
          onChange={next => {
            changes.push(next)
            setValue(next)
          }}
          onSubmit={vi.fn()}
          value={value}
        />
      )
    }

    const instance = renderSync(React.createElement(Harness), {
      patchConsole: false,
      stderr: streams.stderr as NodeJS.WriteStream,
      stdin: streams.stdin as unknown as NodeJS.ReadStream,
      stdout: streams.stdout as NodeJS.WriteStream
    })

    await settle()

    streams.stdin.send('\x0c')
    await settle(25)

    streams.stdin.send('\x0c', '\x0c')
    await settle(25)

    expect(changes).toEqual([])

    instance.unmount()
  })

  it('keeps existing draft text intact when \\x0c arrives mid-typing', async () => {
    const streams = makeStreams()
    const changes: string[] = []

    function Harness() {
      const [value, setValue] = useState('hello')

      return (
        <TextInput
          columns={80}
          onChange={next => {
            changes.push(next)
            setValue(next)
          }}
          onSubmit={vi.fn()}
          value={value}
        />
      )
    }

    const instance = renderSync(React.createElement(Harness), {
      patchConsole: false,
      stderr: streams.stderr as NodeJS.WriteStream,
      stdin: streams.stdin as unknown as NodeJS.ReadStream,
      stdout: streams.stdout as NodeJS.WriteStream
    })

    await settle()

    streams.stdin.send('\x0c')
    await settle(25)

    expect(changes).toEqual([])

    instance.unmount()
  })

  it('still types a bare letter l and other ctrl-branch keys keep working', async () => {
    const streams = makeStreams()
    const changes: string[] = []

    function Harness() {
      const [value, setValue] = useState('')

      return (
        <TextInput
          columns={80}
          onChange={next => {
            changes.push(next)
            setValue(next)
          }}
          onSubmit={vi.fn()}
          value={value}
        />
      )
    }

    const instance = renderSync(React.createElement(Harness), {
      patchConsole: false,
      stderr: streams.stderr as NodeJS.WriteStream,
      stdin: streams.stdin as unknown as NodeJS.ReadStream,
      stdout: streams.stdout as NodeJS.WriteStream
    })

    await settle()

    streams.stdin.send('l')
    await settle(25)

    streams.stdin.send('\x0c')
    await settle(25)

    expect(changes).toEqual(['l'])

    instance.unmount()
  })
})
