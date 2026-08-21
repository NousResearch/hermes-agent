import { EventEmitter } from 'events'

import { renderSync } from '@hermes/ink'
import React, { useEffect, useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

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

function Harness({
  externalValue,
  initial = '',
  onValue,
  rerenderEveryMs = 0
}: {
  externalValue?: string
  initial?: string
  onValue: (value: string) => void
  rerenderEveryMs?: number
}) {
  const [value, setValue] = useState(initial)
  const [, setPulse] = useState(0)

  useEffect(() => {
    if (!rerenderEveryMs) {
      return
    }

    const timer = setInterval(() => setPulse(current => current + 1), rerenderEveryMs)

    return () => clearInterval(timer)
  }, [rerenderEveryMs])

  useEffect(() => {
    if (externalValue !== undefined) {
      setValue(externalValue)
    }
  }, [externalValue])

  return React.createElement(TextInput, {
    accentColor: '#00aaff',
    color: '#ffffff',
    onChange: (next: string) => {
      setValue(next)
      onValue(next)
    },
    value
  })
}

describe('TextInput sustained STT bursts', () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'setInterval', 'Date'] })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('drains a 2ms single-key stream with line breaks without stalling', async () => {
    const stdout = new FakeTty()
    const stdin = new FakeTty()
    const stderr = new FakeTty()
    const values: string[] = []
    const line = 'speech to text keeps typing into the main composer without pausing '
    const expected = Array.from({ length: 16 }, () => line).join('\n')
    const reads = expected.split('').map(ch => (ch === '\n' ? '\x1b[13;2u' : ch))

    const instance = renderSync(React.createElement(Harness, { onValue: value => values.push(value) }), {
      patchConsole: false,
      stderr: stderr as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as unknown as NodeJS.WriteStream
    })

    try {
      await tick()

      for (const read of reads) {
        stdin.send(read)
        vi.advanceTimersByTime(2)
        await tick()
      }

      vi.advanceTimersByTime(100)
      await tick()

      expect(values.at(-1)).toBe(expected)
      // Sustained injected typing must switch away from one raw stdout write
      // per key; otherwise the composer can outrun the terminal/Ink renderer
      // and appear permanently frozen while its output queue drains.
      expect(stdout.chunks.length).toBeLessThan(400)
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  }, 15_000)

  it('does not roll the input back during unrelated parent renders', async () => {
    const stdout = new FakeTty()
    const stdin = new FakeTty()
    const stderr = new FakeTty()
    const values: string[] = []
    const expected = 'continuous dictation remains intact '.repeat(12)

    const instance = renderSync(
      React.createElement(Harness, { onValue: value => values.push(value), rerenderEveryMs: 1 }),
      {
        patchConsole: false,
        stderr: stderr as unknown as NodeJS.WriteStream,
        stdin: stdin as unknown as NodeJS.ReadStream,
        stdout: stdout as unknown as NodeJS.WriteStream
      }
    )

    try {
      await tick()

      for (const read of expected) {
        stdin.send(read)
        vi.advanceTimersByTime(2)
        await tick()
      }

      vi.advanceTimersByTime(100)
      await tick()

      expect(values.at(-1)).toBe(expected)
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })

  it('lets an external reset cancel a queued local frame', async () => {
    const stdout = new FakeTty()
    const stdin = new FakeTty()
    const stderr = new FakeTty()
    const values: string[] = []
    const onValue = (value: string) => values.push(value)
    const mount = (externalValue?: string) => React.createElement(Harness, { externalValue, onValue })

    const instance = renderSync(mount(), {
      patchConsole: false,
      stderr: stderr as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as unknown as NodeJS.WriteStream
    })

    try {
      await tick()

      for (const read of 'abcdefgh') {
        stdin.send(read)
      }

      // The eighth 0ms key arms the frame-batched local update, but its 16ms
      // parent timer has not fired. An external reset must cancel that timer.
      instance.rerender(mount('RESET'))
      await tick()
      vi.advanceTimersByTime(20)
      await tick()

      stdin.send('z')
      vi.advanceTimersByTime(20)
      await tick()

      expect(values.at(-1)).toBe('RESETz')
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })

  it('ignores delayed out-of-order echoes of its own earlier values', async () => {
    const stdout = new FakeTty()
    const stdin = new FakeTty()
    const stderr = new FakeTty()
    const emitted: string[] = []

    const input = (value: string) =>
      React.createElement(TextInput, {
        accentColor: '#00aaff',
        color: '#ffffff',
        onChange: (next: string) => emitted.push(next),
        value
      })

    const instance = renderSync(input(''), {
      patchConsole: false,
      stderr: stderr as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as unknown as NodeJS.WriteStream
    })

    try {
      await tick()

      for (const read of 'abcdefghijklmnop') {
        stdin.send(read)
        vi.advanceTimersByTime(2)
        await tick()
      }

      vi.advanceTimersByTime(20)
      await tick()

      const first = emitted[0]!
      const latest = emitted.at(-1)!
      expect(first).not.toBe(latest)
      expect(latest).toBe('abcdefghijklmnop')

      // Echo the newest value first, then an older value. The old echo is not
      // an external reset and must not roll the live refs/cursor backward.
      instance.rerender(input(latest))
      await tick()
      instance.rerender(input(first))
      await tick()

      stdin.send('z')
      vi.advanceTimersByTime(20)
      await tick()

      expect(emitted.at(-1)).toBe('abcdefghijklmnopz')
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })

  it('accepts an intentional reset to a previously acknowledged value', async () => {
    const stdout = new FakeTty()
    const stdin = new FakeTty()
    const stderr = new FakeTty()
    const emitted: string[] = []

    const input = (value: string) =>
      React.createElement(TextInput, {
        accentColor: '#00aaff',
        color: '#ffffff',
        onChange: (next: string) => emitted.push(next),
        value
      })

    const instance = renderSync(input(''), {
      patchConsole: false,
      stderr: stderr as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as unknown as NodeJS.WriteStream
    })

    try {
      await tick()

      stdin.send('a')
      vi.advanceTimersByTime(20)
      await tick()
      const acknowledged = emitted.at(-1)!
      instance.rerender(input(acknowledged))
      await tick()

      stdin.send('b')
      vi.advanceTimersByTime(20)
      await tick()
      const latest = emitted.at(-1)!
      instance.rerender(input(latest))
      await tick()

      // Queue another local frame, then intentionally restore the already
      // acknowledged value A. A is no longer an outstanding echo, so it must
      // cancel the queued local value rather than be mistaken for stale input.
      for (const read of '12345678') {
        stdin.send(read)
      }

      instance.rerender(input(acknowledged))
      await tick()
      vi.advanceTimersByTime(20)
      await tick()

      stdin.send('z')
      vi.advanceTimersByTime(20)
      await tick()

      expect(emitted.at(-1)).toBe('az')
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })

  it('drops superseded echoes when duplicate values are coalesced', async () => {
    const stdout = new FakeTty()
    const stdin = new FakeTty()
    const stderr = new FakeTty()
    const emitted: string[] = []

    const input = (value: string) =>
      React.createElement(TextInput, {
        accentColor: '#00aaff',
        color: '#ffffff',
        onChange: (next: string) => emitted.push(next),
        value
      })

    const instance = renderSync(input(''), {
      patchConsole: false,
      stderr: stderr as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as unknown as NodeJS.WriteStream
    })

    try {
      await tick()

      stdin.send('a')
      vi.advanceTimersByTime(20)
      await tick()
      stdin.send('b')
      vi.advanceTimersByTime(20)
      await tick()
      stdin.send('\x7f')
      vi.advanceTimersByTime(20)
      await tick()

      expect(emitted).toEqual(['a', 'ab', 'a'])

      // React may coalesce A → B → A into one final A prop. Acknowledging that
      // newest duplicate must also retire the superseded B expectation.
      instance.rerender(input('a'))
      await tick()
      instance.rerender(input('ab'))
      await tick()

      stdin.send('z')
      vi.advanceTimersByTime(20)
      await tick()

      expect(emitted.at(-1)).toBe('abz')
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })

  it('deletes a word when Alt and Backspace arrive in separate reads', async () => {
    const stdout = new FakeTty()
    const stdin = new FakeTty()
    const stderr = new FakeTty()
    const values: string[] = []

    const instance = renderSync(
      React.createElement(Harness, { initial: 'alpha beta', onValue: value => values.push(value) }),
      {
        patchConsole: false,
        stderr: stderr as unknown as NodeJS.WriteStream,
        stdin: stdin as unknown as NodeJS.ReadStream,
        stdout: stdout as unknown as NodeJS.WriteStream
      }
    )

    try {
      await tick()
      stdin.send('\x1b')
      await tick()
      stdin.send('\x7f')
      await tick()

      expect(values.at(-1)).toBe('alpha ')
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })
})
