import { EventEmitter } from 'events'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ActiveSessionSwitcher } from '../components/activeSessionSwitcher.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { SessionActiveItem } from '../gatewayTypes.js'
import { DEFAULT_THEME } from '../theme.js'

// The unified Sessions overlay's `/` key-routing contract: a plain `/` on any
// session row opens the search overlay in place of the switcher, while the
// New row keeps `/` as ordinary prompt input so `/search foo` stays typable
// there. Timers are faked (the 1.5s live-status poll must not re-render
// mid-keystroke); setImmediate stays real so React commits between ticks.
// ink paints per-keystroke deltas, so behaviour is asserted through the prop
// mocks rather than the screen text.

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

const settle = async (rounds = 8) => {
  for (let i = 0; i < rounds; i++) {
    await tick()
  }
}

const type = async (stdin: FakeTty, text: string) => {
  for (const ch of text) {
    stdin.send(ch)
    await tick()
  }

  await tick()
}

const liveSession = (id: string, current = false): SessionActiveItem => ({ current, id, status: 'idle' })

const mountSwitcher = ({ live }: { live: SessionActiveItem[] }) => {
  const stdout = new FakeTty()
  const stdin = new FakeTty()
  const stderr = new FakeTty()
  const onCancel = vi.fn()
  const onNew = vi.fn()
  const onNewPrompt = vi.fn()
  const onSearch = vi.fn()
  const onSelect = vi.fn()

  const request = vi.fn((method: string) => {
    if (method === 'session.active_list') {
      return Promise.resolve({ sessions: live })
    }

    return Promise.resolve({ sessions: [] })
  })

  const instance = renderSync(
    <ActiveSessionSwitcher
      currentSessionId={live.find(s => s.current)?.id ?? null}
      gw={{ request } as unknown as GatewayClient}
      maxWidth={80}
      onCancel={onCancel}
      onClose={async () => null}
      onNew={onNew}
      onNewPrompt={onNewPrompt}
      onResume={vi.fn()}
      onSearch={onSearch}
      onSelect={onSelect}
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
    onNewPrompt,
    onSearch,
    stdin
  }
}

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'] })
  // useStdout() resolves to process.stdout (not the FakeTty passed to
  // renderSync); keep fast-echo off so typed chars render normally.
  ;(process.stdout as { isTTY?: boolean }).isTTY = false
})

afterEach(() => {
  vi.useRealTimers()
  ;(process.stdout as { isTTY?: boolean }).isTTY = undefined
})

describe('ActiveSessionSwitcher `/` key routing', () => {
  it('opens the search overlay on a plain `/` from a session row', async () => {
    const view = mountSwitcher({ live: [liveSession('alpha', true), liveSession('beta')] })

    try {
      // Let the initial load land: selection settles on the current live
      // session (row 1), not the pinned New row.
      await settle()

      view.stdin.send('/')
      await settle()

      expect(view.onSearch).toHaveBeenCalledTimes(1)
      expect(view.onNewPrompt).not.toHaveBeenCalled()
    } finally {
      view.cleanup()
    }
  })

  it('keeps `/` as prompt input on the New row', async () => {
    // No live sessions and no history: selection starts on the New row itself.
    const view = mountSwitcher({ live: [] })

    try {
      await settle()

      await type(view.stdin, '/search foo')
      view.stdin.send('\r')
      await settle()

      expect(view.onSearch).not.toHaveBeenCalled()
      // Submitting echoes the draft back through onNewPrompt: `/` (and the
      // text after it) reached the row's TextInput instead of the search
      // hotkey. (ink paints per-keystroke deltas, so the draft is asserted
      // through the mock, not the screen text.)
      expect(view.onNewPrompt).toHaveBeenCalledWith('/search foo', undefined)
    } finally {
      view.cleanup()
    }
  })
})
