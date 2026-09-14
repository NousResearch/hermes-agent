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

const settle = (ms = 25) => new Promise(resolve => setTimeout(resolve, ms))

const ESC_RE = /\[[\d;]*m/g

/** Mount a composer holding `initial`, with `suggestion` as its ghost. */
function mount(initial: string, suggestion: string) {
  const stdin = new FakeInput()
  const stdout = new PassThrough()
  const stderr = new PassThrough()
  const painted: string[] = []
  const changes: string[] = []

  Object.assign(stdout, { columns: 80, isTTY: false, rows: 24 })
  Object.assign(stderr, { columns: 80, isTTY: false, rows: 24 })
  stdout.on('data', (chunk: Buffer) => painted.push(String(chunk)))

  function Harness() {
    const [value, setValue] = useState(initial)

    return (
      <TextInput
        columns={40}
        onChange={next => {
          changes.push(next)
          setValue(next)
        }}
        onSubmit={() => {}}
        suggestion={suggestion}
        value={value}
      />
    )
  }

  const instance = renderSync(React.createElement(Harness), {
    patchConsole: false,
    stderr: stderr as unknown as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream
  })

  return {
    changes,
    close: () => {
      instance.unmount()
      instance.cleanup()
    },
    frames: () => painted.join('').replace(ESC_RE, ''),
    stdin
  }
}

describe('TextInput inline ghost text', () => {
  it('paints the suggestion after the caret without changing the value', async () => {
    const ui = mount('/he', 'lp')

    await settle()
    const frames = ui.frames()
    ui.close()

    expect(frames).toContain('/help')
    expect(ui.changes).toEqual([])
  })

  it('→ at the end of the line accepts the whole suggestion', async () => {
    const ui = mount('/he', 'lp')

    await settle()
    ui.stdin.send('\u001b[C')
    await settle()
    ui.close()

    expect(ui.changes.at(-1)).toBe('/help')
  })

  it('Ctrl+E accepts it too, and a plain keystroke still just types', async () => {
    const ui = mount('/he', 'lp')

    await settle()
    ui.stdin.send('\u0005')
    await settle()
    expect(ui.changes.at(-1)).toBe('/help')

    ui.stdin.send('x')
    await settle()
    ui.close()

    expect(ui.changes.at(-1)).toBe('/helpx')
  })

  it('leaves → alone as a cursor move when there is nothing to suggest', async () => {
    const ui = mount('/he', '')

    await settle()
    ui.stdin.send('\u001b[C')
    await settle()
    ui.close()

    expect(ui.changes).toEqual([])
  })
})
