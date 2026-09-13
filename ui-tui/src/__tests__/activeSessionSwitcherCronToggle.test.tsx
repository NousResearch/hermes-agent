import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import { stripAnsi } from '@hermes/shared/ansi'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ActiveSessionSwitcher } from '../components/activeSessionSwitcher.js'
import type { GatewayClient } from '../gatewayClient.js'
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

const mounted: Array<{ cleanup: () => void; unmount: () => void }> = []

afterEach(() => {
  while (mounted.length) {
    const instance = mounted.pop()!
    instance.unmount()
    instance.cleanup()
  }
})

const INTERACTIVE_ROW = { id: 'tui-hist', message_count: 2, started_at: 1, title: 'Interactive work' }
const CRON_ROW = { id: 'cron-hist', message_count: 5, started_at: 1, title: 'Scheduled cron brief' }

interface DeferredList {
  params: { include_cron: boolean; limit: number }
  reject: (err: Error) => void
  resolve: (value: unknown) => void
}

/**
 * Gateway stub whose session.list responses only settle when the test says
 * so, letting a test replay real out-of-order completions (an older filter's
 * response finishing after a newer one).
 */
const deferredGateway = () => {
  const listCalls: DeferredList[] = []

  const gw = {
    request: vi.fn((method: string, params?: Record<string, unknown>) => {
      if (method === 'session.active_list') {
        return Promise.resolve({ sessions: [] })
      }

      return new Promise((resolve, reject) => {
        listCalls.push({ params: params as DeferredList['params'], reject, resolve })
      })
    })
  } as unknown as GatewayClient

  return { gw, listCalls }
}

const mountSwitcher = (gw: GatewayClient) => {
  const stdin = new FakeInput()
  const stdout = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 100, isTTY: false, rows: 40 })
  Object.assign(stderr, { columns: 100, isTTY: false, rows: 40 })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(
    <ActiveSessionSwitcher
      currentSessionId={null}
      gw={gw}
      onCancel={() => {}}
      onClose={async () => null}
      onNew={() => {}}
      onNewPrompt={() => {}}
      onResume={() => {}}
      onSelect={() => {}}
      t={DEFAULT_THEME}
    />,
    {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    }
  )

  mounted.push(instance)

  return { frame: () => stripAnsi(output), stdin }
}

const settle = (ms = 25) => new Promise(resolve => setTimeout(resolve, ms))

/** True when the LAST rendered frame shows the checkbox checked (the raw
 * output accumulates every frame, so plain `toContain` sees stale frames). */
const lastCheckboxChecked = (frame: string) =>
  frame.lastIndexOf('[x] Include cron sessions') > frame.lastIndexOf('[ ] Include cron sessions')

describe('Sessions overlay cron filter toggle', () => {
  it('Alt+C flips the checkbox and re-queries session.list with include_cron', async () => {
    const listParams: Array<{ include_cron: boolean }> = []

    const gw = {
      request: vi.fn((method: string, params?: Record<string, unknown>) => {
        if (method === 'session.active_list') {
          return Promise.resolve({ sessions: [] })
        }

        const p = params as { include_cron: boolean }
        listParams.push(p)

        return Promise.resolve({ sessions: p.include_cron ? [INTERACTIVE_ROW, CRON_ROW] : [INTERACTIVE_ROW] })
      })
    } as unknown as GatewayClient

    const { frame, stdin } = mountSwitcher(gw)

    // Default state: unchecked, cron history hidden, first query opted out.
    await vi.waitFor(() => expect(frame()).toContain('[ ] Include cron sessions'))
    await vi.waitFor(() => expect(frame()).toContain('Interactive work'))
    expect(frame()).not.toContain('Scheduled cron brief')
    expect(listParams[0]).toEqual({ include_cron: false, limit: 200 })

    // Alt+C arrives as ESC-prefixed 'c' in a single chunk (meta keypress).
    stdin.send('\u001bc')

    await vi.waitFor(() => expect(frame()).toContain('[x] Include cron sessions'))
    await vi.waitFor(() => expect(frame()).toContain('Scheduled cron brief'))
    expect(listParams.at(-1)).toEqual({ include_cron: true, limit: 200 })
  })

  it('Alt+C on the "+ new" row toggles the filter without leaking a literal c into the draft', async () => {
    const listParams: Array<{ include_cron: boolean }> = []

    const gw = {
      request: vi.fn((method: string, params?: Record<string, unknown>) => {
        if (method === 'session.active_list') {
          return Promise.resolve({ sessions: [] })
        }

        const p = params as { include_cron: boolean }
        listParams.push(p)

        return Promise.resolve({ sessions: p.include_cron ? [INTERACTIVE_ROW, CRON_ROW] : [INTERACTIVE_ROW] })
      })
    } as unknown as GatewayClient

    const { frame, stdin } = mountSwitcher(gw)

    // With no live sessions the "+ new" row is selected, so the draft
    // TextInput is mounted and listening alongside the switcher.
    await vi.waitFor(() => expect(frame()).toContain('[ ] Include cron sessions'))
    stdin.send('hello')
    await vi.waitFor(() => expect(frame()).toContain('hello'))

    stdin.send('\u001bc')

    // The toggle must consume the shared InputEvent: the filter flips and
    // re-queries, and the draft never sees the trailing 'c'.
    await vi.waitFor(() => expect(lastCheckboxChecked(frame())).toBe(true))
    await vi.waitFor(() => expect(frame()).toContain('Scheduled cron brief'))
    expect(listParams.at(-1)).toEqual({ include_cron: true, limit: 200 })
    expect(frame()).not.toContain('helloc')
    expect(frame()).not.toContain('chello')
  })

  it('rapid Alt+C toggles: a slower opposite-filter response cannot overwrite the newest filter', async () => {
    const { gw, listCalls } = deferredGateway()
    const { frame, stdin } = mountSwitcher(gw)

    await vi.waitFor(() => expect(listCalls).toHaveLength(1))
    listCalls[0]!.resolve({ sessions: [INTERACTIVE_ROW] })
    await vi.waitFor(() => expect(frame()).toContain('Interactive work'))

    // Toggle on, then immediately back off — two full reloads in flight.
    stdin.send('\u001bc')
    await vi.waitFor(() => expect(listCalls).toHaveLength(2))
    stdin.send('\u001bc')
    await vi.waitFor(() => expect(listCalls).toHaveLength(3))
    expect(listCalls.map(c => c.params.include_cron)).toEqual([false, true, false])

    // The newest (cron-off) request completes first…
    listCalls[2]!.resolve({ sessions: [INTERACTIVE_ROW] })
    await vi.waitFor(() => expect(lastCheckboxChecked(frame())).toBe(false))

    // …then the stale cron-on response lands last. It must be dropped: the
    // checkbox says cron is excluded, so no cron row may ever render.
    listCalls[1]!.resolve({ sessions: [INTERACTIVE_ROW, CRON_ROW] })
    await settle()

    expect(frame()).not.toContain('Scheduled cron brief')
    expect(lastCheckboxChecked(frame())).toBe(false)
  })

  it('toggle then Ctrl+R refresh: a stale failure cannot surface an error over the newer result', async () => {
    const { gw, listCalls } = deferredGateway()
    const { frame, stdin } = mountSwitcher(gw)

    await vi.waitFor(() => expect(listCalls).toHaveLength(1))
    listCalls[0]!.resolve({ sessions: [INTERACTIVE_ROW] })
    await vi.waitFor(() => expect(frame()).toContain('Interactive work'))

    // Toggle cron on (response held), then manually refresh with Ctrl+R.
    stdin.send('\u001bc')
    await vi.waitFor(() => expect(listCalls).toHaveLength(2))
    stdin.send('\u0012')
    await vi.waitFor(() => expect(listCalls).toHaveLength(3))
    expect(listCalls.map(c => c.params.include_cron)).toEqual([false, true, true])

    // The refresh succeeds first; the superseded toggle request then fails.
    listCalls[2]!.resolve({ sessions: [INTERACTIVE_ROW, CRON_ROW] })
    await vi.waitFor(() => expect(frame()).toContain('Scheduled cron brief'))
    listCalls[1]!.reject(new Error('gateway went away'))
    await settle()

    // The stale failure must not blank the fresh list or surface an error.
    expect(frame()).not.toContain('could not load resumable sessions')
    expect(frame()).not.toContain('error:')
    expect(lastCheckboxChecked(frame())).toBe(true)
  })
})
