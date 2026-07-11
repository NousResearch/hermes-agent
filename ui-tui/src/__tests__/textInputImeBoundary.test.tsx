import { EventEmitter } from 'events'

import { renderSync } from '@hermes/ink'
import React, { useState } from 'react'
import { describe, expect, it } from 'vitest'

import { TextInput } from '../components/textInput.js'

class FakeTty extends EventEmitter {
  chunks: string[] = []
  columns = 80
  rows = 24
  isTTY = true
  isRaw = false
  private pendingReads: string[] = []

  ref(): void {}
  unref(): void {}

  read(): string | null {
    return this.pendingReads.shift() ?? null
  }

  send(chunk: string): void {
    this.pendingReads.push(chunk)
    this.emit('readable')
  }

  setEncoding(): this {
    return this
  }

  setRawMode(mode: boolean): this {
    this.isRaw = mode

    return this
  }

  write(chunk: string | Uint8Array, cb?: (err?: Error | null) => void): boolean {
    this.chunks.push(typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8'))
    cb?.()

    return true
  }
}

const tick = () => new Promise<void>(resolve => setImmediate(resolve))

function Harness({ onValue }: { onValue: (value: string) => void }) {
  const [value, setValue] = useState('o')

  return React.createElement(TextInput, {
    onChange: next => {
      setValue(next)
      onValue(next)
    },
    value
  })
}

describe('TextInput IME recomposition boundary', () => {
  it('applies a parser-split backspace plus composed character through useInput', async () => {
    const stdout = new FakeTty()
    const stdin = new FakeTty()
    const stderr = new FakeTty()
    const values: string[] = []

    const instance = renderSync(React.createElement(Harness, { onValue: value => values.push(value) }), {
      patchConsole: false,
      stderr: stderr as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as unknown as NodeJS.WriteStream
    })

    try {
      await tick()
      stdin.send('\x7fô')
      await tick()

      expect(values.at(-1)).toBe('ô')
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })
})
