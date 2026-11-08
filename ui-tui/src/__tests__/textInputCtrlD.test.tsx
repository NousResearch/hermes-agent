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

const CTRL_D = '\x04'
const LEFT = '\x1b[D'

/** Types `text`, then sends each key in its own read; returns the composer value. */
async function typeThen(text: string, keys: string[]): Promise<string> {
  const stdin = new FakeInput()
  const stdout = new PassThrough()
  const stderr = new PassThrough()

  Object.assign(stdout, { columns: 80, isTTY: false, rows: 24 })
  Object.assign(stderr, { columns: 80, isTTY: false, rows: 24 })

  let latest = ''

  function Harness() {
    const [value, setValue] = useState('')

    return (
      <TextInput
        columns={80}
        onChange={next => {
          latest = next
          setValue(next)
        }}
        onSubmit={() => {}}
        value={value}
      />
    )
  }

  const instance = renderSync(React.createElement(Harness), {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  await settle()
  stdin.send(text)
  await settle(25)

  for (const key of keys) {
    stdin.send(key)
    await settle(25)
  }

  instance.unmount()
  instance.cleanup()

  return latest
}

describe('TextInput Ctrl+D', () => {
  it('deletes the grapheme under the cursor, not the word', async () => {
    expect(await typeThen('foo bar', [LEFT, LEFT, LEFT, CTRL_D])).toBe('foo ar')
    expect(await typeThen('a🙂b', [LEFT, LEFT, CTRL_D])).toBe('ab')
  })

  it('leaves the draft alone at end of input', async () => {
    expect(await typeThen('foo', [CTRL_D])).toBe('foo')
  })

  it('keeps meta+d (ESC d) as delete-word-forward', async () => {
    expect(await typeThen('foo bar', [LEFT, LEFT, LEFT, '\x1bd'])).toBe('foo ')
  })
})
