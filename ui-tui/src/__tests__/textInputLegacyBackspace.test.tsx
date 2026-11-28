import { EventEmitter } from 'events'

import { renderSync } from '@hermes/ink'
import React, { useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { TextInput } from '../components/textInput.js'

class FakeTty extends EventEmitter {
  columns = 80
  rows = 24
  isTTY = true
  isRaw = false
  private pending: string[] = []
  ref(): void {}
  unref(): void {}
  read(): string | null {
    return this.pending.shift() ?? null
  }
  send(chunk: string): void {
    this.pending.push(chunk)
    this.emit('readable')
  }
  setEncoding(): this {
    return this
  }
  setRawMode(mode: boolean): this {
    this.isRaw = mode

    return this
  }
  write(_chunk: string | Uint8Array, cb?: (err?: Error | null) => void): boolean {
    cb?.()

    return true
  }
}

const tick = () => new Promise<void>(resolve => setImmediate(resolve))

describe('composer Backspace terminal encodings', () => {
  beforeEach(() => vi.useFakeTimers({ toFake: ['setTimeout', 'setInterval', 'Date'] }))
  afterEach(() => vi.useRealTimers())

  it.each([
    ['legacy DEL', ['\x1b\x7f'], 'alpha beta '],
    ['legacy BS', ['\x1b\b'], 'alpha beta '],
    ['split legacy DEL', ['\x1b', '\x7f'], 'alpha beta '],
    ['split legacy BS', ['\x1b', '\b'], 'alpha beta '],
    ['bare DEL', ['\x7f'], 'alpha beta gamm'],
    ['bare BS', ['\b'], 'alpha beta gamm'],
    ['standalone Escape', ['\x1b'], 'alpha beta gamma'],
    ['Control+W', ['\x17'], 'alpha beta '],
    ['CSI-u Alt', ['\x1b[127;3u'], 'alpha beta '],
    ['modifyOtherKeys Alt', ['\x1b[27;3;127~'], 'alpha beta '],
    ['CSI-u Super', ['\x1b[127;9u'], '']
  ])('%s deletes only the intended unit', async (_name, reads, expected) => {
    let actual = 'alpha beta gamma'

    function Harness() {
      const [value, setValue] = useState(actual)

      return React.createElement(TextInput, {
        value,
        onChange: (next: string) => {
          actual = next
          setValue(next)
        }
      })
    }

    const stdin = new FakeTty()

    const instance = renderSync(React.createElement(Harness), {
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: new FakeTty() as unknown as NodeJS.WriteStream,
      stderr: new FakeTty() as unknown as NodeJS.WriteStream,
      patchConsole: false
    })

    try {
      await tick()

      for (const read of reads) {
        stdin.send(read)
        await tick()
      }

      vi.advanceTimersByTime(80)
      await tick()
      expect(actual).toBe(expected)
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })
})
