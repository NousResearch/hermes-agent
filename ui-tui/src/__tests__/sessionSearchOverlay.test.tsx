import { EventEmitter } from 'events'

import { renderSync } from '@hermes/ink'
import { stripAnsi } from '@hermes/shared/ansi'
import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { cleanSnippet, SessionSearchOverlay } from '../components/sessionSearchOverlay.js'
import type { GatewayClient } from '../gatewayClient.js'
import { DEFAULT_THEME } from '../theme.js'

// The /search overlay's client-side contract: debounced type-to-filter with a
// monotonic seq guard (in-flight RPCs cannot be aborted — late arrivals must
// be dropped), snippet cleaning for raw tool-JSON rows, and the guarded
// resume path on Enter.
//
// Only setTimeout/setInterval/Date are faked — setImmediate stays real so
// React's scheduler still commits renders between ticks. Textual assertions
// flatten spaces (ink's writer emits cursor-forward escapes for styled
// space cells) and are limited to lines painted by a full frame; everything
// result-dependent is asserted through the request/onResume mocks instead.

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

// Enough scheduler turns for an RPC promise chain plus the re-render it
// triggers to land — vi.waitFor would spin on faked timers, so poll manually.
const settle = async (rounds = 8) => {
  for (let i = 0; i < rounds; i++) {
    await tick()
  }
}

const DEBOUNCE_MS = 300

/** Space-flattened screen text: single spaces may arrive as cursor-forward escapes. */
const flat = (text: string) => stripAnsi(text).replace(/ /g, '')

const type = async (stdin: FakeTty, text: string) => {
  for (const ch of text) {
    stdin.send(ch)
    await tick()
  }

  // Fast-echo is off, so each keystroke commits synchronously — no 16ms
  // key-burst to flush and no fake-time advance needed here.
  await tick()
}

const result = (id: string, title: string, snippet = '') => ({
  id,
  snippet,
  source: 'cli',
  started_at: Math.floor(Date.now() / 1000) - 3600,
  title
})

const mountOverlay = ({ initialQuery, request }: { initialQuery?: string; request: ReturnType<typeof vi.fn> }) => {
  const stdout = new FakeTty()
  const stdin = new FakeTty()
  const stderr = new FakeTty()
  const onCancel = vi.fn()
  const onResume = vi.fn()

  const instance = renderSync(
    <SessionSearchOverlay
      gw={{ request } as unknown as GatewayClient}
      initialQuery={initialQuery}
      maxWidth={80}
      onCancel={onCancel}
      onResume={onResume}
      t={DEFAULT_THEME}
    />,
    {
      patchConsole: false,
      stderr: stderr as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as unknown as NodeJS.WriteStream
    }
  )

  let done = false

  return {
    cleanup: () => {
      if (!done) {
        done = true
        instance.unmount()
        instance.cleanup()
      }
    },
    onCancel,
    onResume,
    output: () => flat(stdout.chunks.join('')),
    stdin
  }
}

beforeEach(() => {
  // clearTimeout/clearInterval must be faked too: sinon's fake timer handles
  // are opaque to the real clearTimeout, so the debounce's clear-and-re-arm
  // would silently never cancel and every keystroke would fire.
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'] })
  // useStdout() resolves to process.stdout (not the FakeTty passed to
  // renderSync); keep fast-echo off so typed chars render normally.
  ;(process.stdout as { isTTY?: boolean }).isTTY = false
})

afterEach(() => {
  vi.useRealTimers()
  ;(process.stdout as { isTTY?: boolean }).isTTY = undefined
})

describe('cleanSnippet', () => {
  it('collapses whitespace runs in plain transcript text', () => {
    expect(cleanSnippet('foo \n\t bar   baz')).toBe('foo bar baz')
  })

  it('surfaces the output payload from raw tool JSON', () => {
    expect(cleanSnippet('{"output": "=== GITHUB result", "meta": 1}')).toBe('=== GITHUB result')
  })

  it('prefers output over content even when content appears first', () => {
    expect(cleanSnippet('{"content": "C", "output": "O"}')).toBe('O')
  })

  it('finds a preferred string at any depth, including inside arrays', () => {
    expect(cleanSnippet('{"a": {"b": [{"text": "deep"}]}}')).toBe('deep')
  })

  it('keeps the collapsed raw text for broken JSON', () => {
    expect(cleanSnippet('{"output": "unclosed')).toBe('{"output": "unclosed')
  })

  it('keeps the collapsed raw text when no string value exists', () => {
    expect(cleanSnippet('{"count": 3}')).toBe('{"count": 3}')
  })

  it('hard-caps at 160 chars with an ellipsis', () => {
    const cleaned = cleanSnippet('x'.repeat(200))

    expect(cleaned).toHaveLength(160)
    expect(cleaned.endsWith('…')).toBe(true)
  })

  it('handles a JSON array root', () => {
    expect(cleanSnippet('[{"snippet": "S"}, 1]')).toBe('S')
  })
})

describe('SessionSearchOverlay', () => {
  it('fires one immediate search for a prefilled query, without the debounce delay', async () => {
    const request = vi.fn(() => Promise.resolve({ results: [result('sid-1', 'First', 'found it')] }))
    const view = mountOverlay({ initialQuery: 'found', request })

    try {
      // No timer advance yet — the fill must not wait on the debounce.
      await settle()

      expect(request).toHaveBeenCalledTimes(1)
      expect(request).toHaveBeenCalledWith('session.search', { limit: 15, query: 'found' })
      expect(view.output()).toContain('↑↓select·↵open·escclose')
    } finally {
      view.cleanup()
    }
  })

  it('shows the idle hint for an empty query and fires nothing', async () => {
    const request = vi.fn(() => Promise.resolve({ results: [] }))
    const view = mountOverlay({ request })

    try {
      await settle()

      expect(view.output()).toContain('typetosearchsessiontitlesandcontent')
      expect(request).not.toHaveBeenCalled()
    } finally {
      view.cleanup()
    }
  })

  it('debounces: three rapid keystrokes fire a single search once typing pauses', async () => {
    const request = vi.fn(() => Promise.resolve({ results: [] }))
    const view = mountOverlay({ request })

    try {
      await settle()
      await type(view.stdin, 'abc')

      await vi.advanceTimersByTimeAsync(DEBOUNCE_MS - 1)
      expect(request).not.toHaveBeenCalled()

      await vi.advanceTimersByTimeAsync(1)
      await settle()

      expect(request).toHaveBeenCalledTimes(1)
      expect(request).toHaveBeenCalledWith('session.search', { limit: 15, query: 'abc' })
    } finally {
      view.cleanup()
    }
  })

  it('drops a stale response when a newer request was issued', async () => {
    // Both searches stay in flight; the caller resolves them out of order and
    // probes the live result set through Enter after each resolution.
    const pending: Array<(value: unknown) => void> = []

    const request = vi.fn(
      () => new Promise(resolve => pending.push(resolve as (value: unknown) => void))
    )

    const view = mountOverlay({ request })

    try {
      await settle()

      await type(view.stdin, 'a')
      await vi.advanceTimersByTimeAsync(DEBOUNCE_MS)
      await settle()
      expect(request).toHaveBeenCalledTimes(1)

      await type(view.stdin, 'b')
      await vi.advanceTimersByTimeAsync(DEBOUNCE_MS)
      await settle()
      expect(request).toHaveBeenCalledTimes(2)
      expect(request).toHaveBeenLastCalledWith('session.search', { limit: 15, query: 'ab' })

      pending[1]!({ results: [result('sid-new', 'Newer')] })
      await settle()

      view.stdin.send('\r')
      await settle()
      expect(view.onResume).toHaveBeenCalledTimes(1)
      expect(view.onResume).toHaveBeenCalledWith('sid-new')

      // The older ('a') response lands LAST — it must not replace the newer
      // result set the selection is anchored to.
      pending[0]!({ results: [result('sid-old', 'Older')] })
      await settle()

      view.stdin.send('\r')
      await settle()

      expect(view.onResume).toHaveBeenCalledTimes(2)
      expect(view.onResume).toHaveBeenLastCalledWith('sid-new')
    } finally {
      view.cleanup()
    }
  })

  it('Enter resumes the selected result and arrows move the selection', async () => {
    const request = vi.fn(() =>
      Promise.resolve({ results: [result('sid-1', 'First'), result('sid-2', 'Second')] })
    )

    const view = mountOverlay({ initialQuery: 'found', request })

    try {
      await settle()
      expect(request).toHaveBeenCalledTimes(1)

      view.stdin.send('\r')
      await settle()
      expect(view.onResume).toHaveBeenCalledWith('sid-1')

      view.stdin.send('\x1b[B')
      await settle()
      view.stdin.send('\r')
      await settle()
      expect(view.onResume).toHaveBeenLastCalledWith('sid-2')
      expect(view.onCancel).not.toHaveBeenCalled()
    } finally {
      view.cleanup()
    }
  })

  it('Esc closes the overlay and cancels the pending debounce', async () => {
    const request = vi.fn(() => Promise.resolve({ results: [] }))
    const view = mountOverlay({ request })

    try {
      await settle()
      await type(view.stdin, 'ab')

      // A doubled ESC byte: the tokenizer holds a trailing lone escape for
      // sequence-boundary detection, so only the pair emits the keypress.
      view.stdin.send('\x1b\x1b')
      await settle()
      expect(view.onCancel).toHaveBeenCalled()

      // The real mount closes on cancel (overlay flag false unmounts the
      // panel); a TextInput key-burst flushed after that must not arm a
      // search for the closed overlay.
      view.cleanup()

      await vi.advanceTimersByTimeAsync(DEBOUNCE_MS * 2)
      await settle()
      expect(request).not.toHaveBeenCalled()
    } finally {
      view.cleanup()
    }
  })
})
