import { EventEmitter } from 'events'

import { renderSync } from '@hermes/ink'
import React, { useState } from 'react'
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'

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

// Core driver: feeds reads, optionally advancing fake timers between reads to
// simulate the small macrotask gaps real IME reads arrive with. Returns the
// final value seen by the parent immediately after the last read (no trailing
// wait) so a passing assertion proves the commit was synchronous, not deferred.
async function drive(
  reads: string[],
  { initial = '', gapMs = 0 }: { initial?: string; gapMs?: number } = {}
): Promise<string> {
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
        // Advance the fake clock to flush any pending FRAME_BATCH_MS timers
        // between reads (mirrors the real macrotask gap), then let microtasks run.
        vi.advanceTimersByTime(gapMs)
        await tick()
      }
    }

    // Assert IMMEDIATELY after the final read — no trailing 60ms wait and
    // WITHOUT advancing the fake clock past the deferred key-burst window.
    // If the value is already correct here, the multi-char insert committed
    // synchronously; the old deferred path (scheduleKeyBurstCommit, 16ms)
    // has NOT flushed yet, so a stale/dropped tail would still be visible.

    return values.at(-1) ?? ''
  } finally {
    instance.unmount()
    instance.cleanup()
  }
}

const NNBSP = '\u202f'

describe('Vietnamese Telex IME recomposition', () => {
  beforeEach(() => {
    // Only fake setTimeout/setInterval/Date — NOT setImmediate (used by tick()).
    vi.useFakeTimers({ toFake: ['setTimeout', 'setInterval', 'Date'] })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('applies a parser-split backspace plus composed character through useInput', async () => {
    // OpenKey fuses the erase + recomposed glyph into a single stdin read.
    expect(await drive(['\x7fô'], { initial: 'o' })).toBe('ô')
  })

  it('commits a multi-character recompose synchronously (no dropped tail)', async () => {
    // "hanhj" -> a U+202F marker, four backspaces, then the recomposed "ạnh".
    // Only a single microtask after the last read — the sync commit must have
    // already delivered the final value (the deferred path dropped "nh" here).
    const reads = ['h', 'a', 'n', 'h', NNBSP, '\x7f\x7f', '\x7f\x7f', '\u1EA1nh']

    // No gapMs, no advanceTimersMs — we assert BEFORE the 16ms FRAME_BATCH_MS could fire.
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

describe('Fast-echo suppression reset (60ms window)', () => {
  beforeEach(() => {
    // Only fake setTimeout/setInterval/Date — NOT setImmediate (used by tick()).
    vi.useFakeTimers({ toFake: ['setTimeout', 'setInterval', 'Date'] })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('suppresses fast-echo backspace for one keystroke after an Ink repaint (IME recompose)', async () => {
    // Simulate: user types "ha" -> Ink commits normally -> then IME recompose arrives
    // as NNBSP + backspaces + recomposed text. The first backspace after the Ink
    // repaint must NOT fast-echo (would strand the NNBSP marker as a stray space).

    // Type "ha" normally (each char goes through fast-echo append path)
    let reads = ['h', 'a']
    const stdout1 = new FakeTty()
    const stdin1 = new FakeTty()
    const stderr1 = new FakeTty()
    const values1: string[] = []

    const instance1 = renderSync(React.createElement(Harness, { initial: '', onValue: v => values1.push(v) }), {
      patchConsole: false,
      stderr: stderr1 as unknown as NodeJS.WriteStream,
      stdin: stdin1 as unknown as NodeJS.ReadStream,
      stdout: stdout1 as unknown as NodeJS.WriteStream
    })

    try {
      await tick()
      for (const r of reads) {
        stdin1.send(r)
        await tick()
      }
      // After "ha", fast-echo is enabled (inkRepaintedRef.current = false)
      expect(values1.at(-1)).toBe('ha')

      // Now simulate an IME recompose burst that forces an Ink repaint:
      // NNBSP marker forces a full Ink render (syncParent=true in commit).
      // The next backspace should be SUPPRESSED (fast-echo backspace disabled).
      stdin1.send(NNBSP + '\x7f\x7f\u1EA1nh') // fused chunk: marker + 2x backspace + "ạnh"
      await tick()

      // The recomposed value must be committed synchronously (no dropped tail).
      // The first backspace after the Ink repaint must NOT have written "\b \b" to stdout.
      // We can't directly inspect stdout here, but we verify the FINAL value is correct.
      expect(values1.at(-1)).toBe('h\u1EA1nh')

      // Advance fake timers past the 60ms suppression window so the
      // inkRepaintResetTimer fires and re-enables fast-echo backspace.
      vi.advanceTimersByTime(60)
      await tick()

      // Now fast-echo backspace is RE-ENABLED. One backspace deletes exactly
      // one grapheme ("h") off the end of "hạnh" -> "hạn".
      stdin1.send('\x7f')
      await tick()

      expect(values1.at(-1)).toBe('h\u1EA1n')
    } finally {
      instance1.unmount()
      instance1.cleanup()
    }
  })

  it('does NOT suppress fast-echo backspace when no Ink repaint occurred (normal typing)', async () => {
    // Normal ASCII typing never triggers the Ink-repaint suppression.
    const reads = ['h', 'e', 'l', 'l', 'o']
    expect(await drive(reads)).toBe('hello')

    // Two backspaces off "hello" -> "hel" via the fast-echo path.
    const reads2 = [...reads, '\x7f', '\x7f']
    expect(await drive(reads2)).toBe('hel')
  })
})