import { EventEmitter } from 'events'

import { renderSync } from '@hermes/ink'
import React, { useState } from 'react'
import { describe, expect, it } from 'vitest'

import { TextInput } from '../components/textInput.js'

// End-to-end regression coverage for Vietnamese Telex IME recomposition
// (OpenKey / Unikey / EVKey). These IMEs commit a finished syllable by
// emitting a burst of backspaces (and, for OpenKey, a U+202F NARROW NO-BREAK
// SPACE marker) followed by the recomposed characters. The byte streams below
// are real captures taken from OpenKey and EVKey on macOS while typing the
// phrase "vương sỹ hạnh" (Telex: "vuonwg syx hanhj").
//
// The bug these guard against: characters were dropped and a stray space was
// left mid-syllable (e.g. "hạnh" rendered as "hạ  "). Root causes fixed:
//   1. parse-keypress split fused control-byte+text chunks so the recomposed
//      text survives instead of being discarded with the control byte.
//   2. textInput commits multi-character (IME/paste) inserts synchronously
//      instead of through the 16ms key-burst path that raced re-renders.

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
const wait = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms))

function Harness({ initial = '', onValue }: { initial?: string; onValue: (value: string) => void }) {
  const [value, setValue] = useState(initial)

  return React.createElement(TextInput, {
    onChange: (next: string) => {
      setValue(next)
      onValue(next)
    },
    value
  })
}

async function drive(reads: string[], { initial = '', gapMs = 0 }: { initial?: string; gapMs?: number } = {}): Promise<string> {
  const stdout = new FakeTty()
  const stdin = new FakeTty()
  const stderr = new FakeTty()
  const values: string[] = []

  const instance = renderSync(React.createElement(Harness, { initial, onValue: v => values.push(v) }), {
    patchConsole: false,
    stderr: stderr as unknown as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream
  })

  try {
    await tick()

    for (const r of reads) {
      stdin.send(r)
      await tick()

      if (gapMs) {
        await wait(gapMs)
      }
    }

    await wait(60)

    return values.at(-1) ?? ''
  } finally {
    instance.unmount()
    instance.cleanup()
  }
}

const NNBSP = '\u202f'

describe('Vietnamese Telex IME recomposition', () => {
  it('applies a parser-split backspace plus composed character through useInput', async () => {
    // OpenKey fuses the erase + recomposed glyph into a single stdin read.
    expect(await drive(['\x7fô'], { initial: 'o' })).toBe('ô')
  })

  it('commits a multi-character recompose synchronously (no dropped tail)', async () => {
    // "hanhj" -> a U+202F marker, four backspaces, then the recomposed "ạnh".
    // Only a single microtask after the last read — the sync commit must have
    // already delivered the final value (the deferred path dropped "nh" here).
    const reads = ['h', 'a', 'n', 'h', NNBSP, '\x7f\x7f', '\x7f\x7f', '\u1EA1nh']

    expect(await drive(reads)).toBe('h\u1EA1nh')
  })

  it('reproduces the full phrase "vương sỹ hạnh" from a real OpenKey capture', async () => {
    // Captured byte stream for Telex "vuonwg syx hanhj": each syllable injects a
    // U+202F marker, erases, and re-emits. Verified across read timings.
    const reads = [
      'v', 'u', 'o', NNBSP, '\x7f\x7f', '\x7f\u01B0\u01A1', 'n', 'g',
      ' ', 's', 'y', NNBSP, '\x7f', '\x7f\u1EF9',
      ' ', 'h', 'a', 'n', 'h', NNBSP, '\x7f\x7f\x7f\x7f\u1EA1nh'
    ]

    for (const gapMs of [0, 17, 25]) {
      expect(await drive(reads, { gapMs })).toBe('vương sỹ hạnh')
    }
  })

  it('handles the EVKey capture (clean backspaces, no marker) for "hạnh"', async () => {
    // EVKey emits three clean backspaces and no U+202F; must also yield "hạnh".
    const reads = ['h', 'a', 'n', 'h', '\x7f', '\x7f', '\x7f', '\u1EA1nh']

    expect(await drive(reads)).toBe('h\u1EA1nh')
  })
})
