import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import { stripAnsi } from '@hermes/shared/ansi'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ActiveSessionSwitcher } from '../components/activeSessionSwitcher.js'
import type { GatewayClient } from '../gatewayClient.js'
import { messages } from '../i18n/runtime.js'
import { DEFAULT_THEME } from '../theme.js'

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

const settle = (ms = 40) => new Promise(resolve => setTimeout(resolve, ms))

// Frames land asynchronously (RPC promises, then Ink's render); wait for the
// condition instead of sleeping, so the test holds under a loaded parallel run.
const eventually = (assertion: () => void) => vi.waitFor(assertion, { interval: 10, timeout: 3000 })

const DOWN = '\u001b[B'
const ENTER = '\r'
const ESC = '\u001b'

const history = [
  { id: 'a1', title: 'Fix reconnect', started_at: 4, message_count: 2 },
  { id: 'a2', title: 'Docs cleanup', started_at: 3, message_count: 2 },
  { id: 'a3', title: 'Parser edge cases', started_at: 2, message_count: 2 },
  { id: 'a4', title: 'Release checklist', started_at: 1, message_count: 2 },
  { id: 'b1', title: 'Benchmark run', started_at: 0, message_count: 2 }
]

const projects = [
  { id: 'p_a', label: 'alpha', sessionIds: ['a1', 'a2', 'a3', 'a4'] },
  { id: 'p_b', label: 'beta', sessionIds: ['b1'] }
]

function mount({ tree = true }: { tree?: boolean } = {}) {
  const stdin = new FakeInput()
  const stdout = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 120, isTTY: false, rows: 40 })
  Object.assign(stderr, { columns: 120, isTTY: false, rows: 40 })
  stdout.on('data', chunk => (output += chunk.toString()))

  const gw = {
    request: vi.fn(async (method: string) => {
      if (method === 'session.active_list') {
        return { sessions: [] }
      }

      if (method === 'session.list') {
        return { sessions: history }
      }

      if (method === 'projects.tree' && tree) {
        return { projects }
      }

      throw new Error(`unexpected ${method}`)
    })
  } as unknown as GatewayClient

  const onCancel = vi.fn()
  const onResume = vi.fn()

  const instance = renderSync(
    <ActiveSessionSwitcher
      currentSessionId={null}
      gw={gw}
      onCancel={onCancel}
      onClose={async () => null}
      onNew={vi.fn()}
      onNewPrompt={vi.fn()}
      onResume={onResume}
      onSelect={vi.fn()}
      t={DEFAULT_THEME}
    />,
    {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    }
  )

  // The newest frame: everything after the last time the picker's title was painted.
  const frame = () => stripAnsi(output.split(messages().pickers.session.title).at(-1) ?? '')

  const press = async (...keys: string[]) => {
    for (const key of keys) {
      stdin.send(key)
      // Keys must arrive as separate reads: under load two queued chunks would parse as one input.
      await eventually(() => expect(stdin.chunks).toHaveLength(0))
      await settle(key === ESC ? 120 : 20)
    }
  }

  return { frame, gw, instance, onCancel, onResume, press }
}

let mounted: ReturnType<typeof mount> | null = null

const open = async (options?: Parameters<typeof mount>[0]) => {
  mounted = mount(options)
  await settle(80)

  return mounted
}

afterEach(() => {
  mounted?.instance.unmount()
  mounted?.instance.cleanup()
  mounted = null
})

const shows = (frame: () => string, ...texts: string[]) =>
  eventually(() => texts.forEach(t => expect(frame()).toContain(t)))

const hides = (frame: () => string, ...texts: string[]) =>
  eventually(() => texts.forEach(t => expect(frame()).not.toContain(t)))

describe('ActiveSessionSwitcher project groups', () => {
  it('shows each project with its first 3 sessions by default', async () => {
    const { frame } = await open()

    await shows(frame, '▾ alpha (3/4)', 'Parser edge cases', '▾ beta (1)', 'Benchmark run')
    expect(frame()).not.toContain('Release checklist')
  })

  it('cycles a selected header through all → name only → first 3 with Enter', async () => {
    const { frame, press } = await open()

    await press(DOWN, ENTER)
    await shows(frame, '▼ alpha (4)', 'Release checklist')

    await press(ENTER)
    await shows(frame, 'alpha (4)', 'Benchmark run')
    await hides(frame, 'Fix reconnect')
    expect(frame()).toContain('▸')

    await press(ENTER)
    await shows(frame, '▾ alpha (3/4)', 'Fix reconnect')
  })

  it('g flips to the flat list and back', async () => {
    const { frame, press } = await open()

    await press(DOWN, 'g')
    await shows(frame, 'Release checklist', 'Benchmark run')
    await hides(frame, 'alpha (')

    await press('g')
    await shows(frame, '▾ alpha (3/4)')
  })

  it('resumes the session under the cursor, not the header above it', async () => {
    const { frame, onResume, press } = await open()

    await shows(frame, 'Fix reconnect')
    await press(DOWN, DOWN, ENTER)

    await eventually(() => expect(onResume).toHaveBeenCalledWith('a1'))
  })

  it('falls back to the flat list when the project tree is unavailable', async () => {
    const { frame } = await open({ tree: false })

    await shows(frame, 'Fix reconnect', 'Release checklist', 'Benchmark run')
    expect(frame()).not.toContain('alpha (')
  })
})

describe('ActiveSessionSwitcher search', () => {
  it('/ filters by project name, and Esc clears the filter before closing', async () => {
    const { frame, onCancel, press } = await open()

    await shows(frame, 'Fix reconnect')
    await press(DOWN, '/')
    await press(...'beta')
    await shows(frame, 'Benchmark run')
    await hides(frame, 'Fix reconnect')

    await press(ESC)
    await shows(frame, 'Fix reconnect')
    expect(onCancel).not.toHaveBeenCalled()

    await press(ESC)
    await eventually(() => expect(onCancel).toHaveBeenCalledTimes(1))
  })

  it('matches session text and keeps the filter after Enter', async () => {
    const { frame, press } = await open()

    await shows(frame, 'Fix reconnect')
    await press(DOWN, '/')
    await press(...'parser')
    await press(ENTER)

    await shows(frame, 'Parser edge cases', messages().pickers.session.searchClear.trim())
    await hides(frame, 'Docs cleanup')
  })

  it('says so when nothing matches', async () => {
    const { frame, press } = await open()

    await shows(frame, 'Fix reconnect')
    await press(DOWN, '/')
    await press(...'zzzz')

    await shows(frame, messages().pickers.session.noMatches)
  })
})
