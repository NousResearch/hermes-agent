import { EventEmitter } from 'events'

import React, { useContext, useEffect } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import StdinContext from './components/StdinContext.js'
import Text from './components/Text.js'
import Ink from './ink.js'
import instances from './instances.js'
import { csi } from './termio/csi.js'
import { DEC, DISABLE_MOUSE_TRACKING, enableMouseTrackingFor } from './termio/dec.js'

// DECRQM request for mode 1000 (what the watchdog writes).
const DECRQM_1000 = csi(`?${DEC.MOUSE_NORMAL}$p`)
// DA1 sentinel (what querier.flush() writes).
const DA1_REQUEST = csi('c')
// DECRPM replies (what the terminal answers).
const DECRPM_1000_SET = csi(`?${DEC.MOUSE_NORMAL};1$y`)
const DECRPM_1000_RESET = csi(`?${DEC.MOUSE_NORMAL};2$y`)
const DA1_REPLY = csi('?62c')

// Watchdog cadence (mirrors MOUSE_WATCHDOG_INTERVAL_MS in App.tsx).
const TICK_MS = 2000

class FakeStdout extends EventEmitter {
  chunks: string[] = []
  columns = 80
  rows = 24
  isTTY = true

  write(chunk: string | Uint8Array, cb?: (err?: Error | null) => void): boolean {
    this.chunks.push(typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8'))
    cb?.()

    return true
  }
}

// Stdin fake with a real readable-style buffer so tests can feed terminal
// responses (DECRPM / DA1) the way a live pty would deliver them.
class FakeStdin extends EventEmitter {
  isTTY = true
  isRaw = false
  private buffer: string[] = []

  get readableLength(): number {
    return this.buffer.reduce((n, c) => n + c.length, 0)
  }

  ref(): void {}
  unref(): void {}
  setEncoding(): this {
    return this
  }
  setRawMode(mode: boolean): this {
    this.isRaw = mode

    return this
  }
  read(): string | null {
    return this.buffer.shift() ?? null
  }
  feed(data: string): void {
    this.buffer.push(data)
    this.emit('readable')
  }
}

function RawModeConsumer() {
  const { setRawMode, isRawModeSupported } = useContext(StdinContext)

  useEffect(() => {
    if (!isRawModeSupported) {
      return
    }

    setRawMode(true)

    return () => setRawMode(false)
  }, [isRawModeSupported, setRawMode])

  return React.createElement(Text, null, 'x')
}

type Harness = {
  ink: Ink
  stdout: FakeStdout
  stdin: FakeStdin
  /** Advance one watchdog tick and let the probe write settle. */
  tickWatchdog: () => Promise<void>
  /** Feed a terminal response and let promise resolution settle. */
  answer: (data: string) => Promise<void>
}

const flushMicrotasks = async () => {
  // Real setImmediate turns: lets React flush effects (raw-mode enable),
  // the deferred init writes fire, and querier promise chains settle.
  // Two rounds cover promise → setImmediate interleave.
  await new Promise<void>(resolve => setImmediate(resolve))
  await new Promise<void>(resolve => setImmediate(resolve))
}

async function mount(mouseTracking: 'all' | 'off' = 'all'): Promise<Harness> {
  const stdout = new FakeStdout()
  const stdin = new FakeStdin()
  const stderr = new FakeStdout()

  const ink = new Ink({
    exitOnCtrlC: false,
    patchConsole: false,
    stderr: stderr as unknown as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream
  })

  // Production instances are registered by render.ts; direct construction
  // skips that. The watchdog (like the raw-mode re-assert) resolves its
  // Ink through this map, so mirror production here.
  instances.set(stdout as unknown as NodeJS.WriteStream, ink)

  ink.setAltScreenActive(true, mouseTracking)
  ink.render(React.createElement(RawModeConsumer))
  ink.onRender()
  await flushMicrotasks()

  // The XTVERSION probe from raw-mode entry has its own DA1 sentinel
  // pending. Answer it so the querier queue is empty before tests start.
  stdin.feed(DA1_REPLY)
  await flushMicrotasks()

  stdout.chunks = []

  return {
    ink,
    stdout,
    stdin,
    tickWatchdog: async () => {
      await vi.advanceTimersByTimeAsync(TICK_MS)
    },
    answer: async (data: string) => {
      stdin.feed(data)
      await flushMicrotasks()
    }
  }
}

describe('App mouse-mode watchdog', () => {
  beforeEach(() => {
    // Fake only the interval + Date clock. setImmediate/setTimeout stay
    // real so React effect flushing and Ink's internal scheduling work.
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'Date'] })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('probes DECRQM on the interval and re-asserts tracking when the terminal reports RESET', async () => {
    const h = await mount('all')

    await h.tickWatchdog()

    // Probe went out: DECRQM for mode 1000 + DA1 sentinel.
    const probed = h.stdout.chunks.join('')

    expect(probed).toContain(DECRQM_1000)
    expect(probed).toContain(DA1_REQUEST)

    h.stdout.chunks = []

    // Terminal says mode 1000 is RESET (someone cleared our modes), then
    // answers the sentinel.
    await h.answer(DECRPM_1000_RESET + DA1_REPLY)

    const out = h.stdout.chunks.join('')

    // reassertTerminalModes: DISABLE first, then the full 'all' preset.
    expect(out).toContain(DISABLE_MOUSE_TRACKING)
    expect(out).toContain(enableMouseTrackingFor('all'))

    h.ink.unmount()
  })

  it('does nothing when the terminal reports the mode still SET', async () => {
    const h = await mount('all')

    await h.tickWatchdog()
    h.stdout.chunks = []

    await h.answer(DECRPM_1000_SET + DA1_REPLY)

    expect(h.stdout.chunks.join('')).not.toContain(enableMouseTrackingFor('all'))

    h.ink.unmount()
  })

  it('disables itself permanently when the terminal ignores DECRQM', async () => {
    const h = await mount('all')

    await h.tickWatchdog()
    expect(h.stdout.chunks.join('')).toContain(DECRQM_1000)
    h.stdout.chunks = []

    // Terminal answers only the DA1 sentinel — DECRQM unsupported.
    await h.answer(DA1_REPLY)

    // No re-assert...
    expect(h.stdout.chunks.join('')).not.toContain(enableMouseTrackingFor('all'))

    // ...and no further probes on subsequent ticks.
    await h.tickWatchdog()
    await h.tickWatchdog()
    expect(h.stdout.chunks.join('')).not.toContain(DECRQM_1000)

    h.ink.unmount()
  })

  it('does not probe when mouse tracking is off', async () => {
    const h = await mount('off')

    await h.tickWatchdog()
    await h.tickWatchdog()

    expect(h.stdout.chunks.join('')).not.toContain(DECRQM_1000)

    h.ink.unmount()
  })

  it('skips the probe when a mouse event arrived within the interval', async () => {
    const h = await mount('all')

    // Half a tick in, a live SGR mouse event (wheel-up at 10;5) proves
    // tracking works; the interval fires half a tick later → gap < interval.
    await vi.advanceTimersByTimeAsync(TICK_MS / 2)
    await h.answer(csi('<64;10;5M'))
    await vi.advanceTimersByTimeAsync(TICK_MS / 2)

    expect(h.stdout.chunks.join('')).not.toContain(DECRQM_1000)

    h.ink.unmount()
  })
})
