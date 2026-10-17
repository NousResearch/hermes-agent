import { PassThrough, Writable } from 'node:stream'
import { stripVTControlCharacters } from 'node:util'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ActiveSessionSwitcher } from '../components/activeSessionSwitcher.js'
import type { GatewayClient } from '../gatewayClient.js'
import { resetLocale } from '../i18n/runtime.js'
import { DEFAULT_THEME } from '../theme.js'

// Real React effects + real Ink output; only the gateway transport is a fixture.
// No source-text inspection, helper-only render, component mock, or user config.
const NOW = 1_800_000_000_000

const noop = () => {}
type Kind = 'history' | 'live'

beforeEach(() => {
  resetLocale()
  vi.spyOn(Date, 'now').mockReturnValue(NOW)
})
afterEach(() => {
  vi.restoreAllMocks()
  resetLocale()
})

async function withSessionRow(
  kind: Kind,
  source: string | undefined,
  columns: number,
  check: (row: string, frame: string) => void,
  historyFailure = false
) {
  const title = kind === 'history' ? 'Hist title' : 'Live title'

  const item = {
    id: kind === 'history' ? 'hist-001' : 'live-001',
    title,
    preview: 'PREVIEW_NOT_SOURCE',
    started_at: NOW / 1000 - 3600,
    message_count: 7,
    model: 'provider/model-x',
    status: 'idle',
    ...(source === undefined ? {} : { source })
  }

  const request = vi.fn(async (method: string) => {
    if (method === 'session.active_list') {return { sessions: kind === 'live' ? [item] : [] }}

    if (method === 'session.list') {
      if (historyFailure) {throw new Error('History unavailable')}

      return {
        sessions: kind === 'history' ? [item] : [],
        total: kind === 'history' ? 1 : 0
      }
    }

    throw new Error(`Unexpected gateway request: ${method}`)
  })

  const frames: string[] = []

  const stdout = Object.assign(
    new Writable({
      write(chunk, _encoding, callback) {
        // Ink emits cursor-style CSI; Node's stripper leaves that sequence.
        // eslint-disable-next-line no-control-regex
        const text = stripVTControlCharacters(chunk.toString().replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, ''))

        if (text.trim()) {frames.push(text)}
        callback()
      }
    }),
    { columns, rows: 30, isTTY: false }
  )

  const stdin = Object.assign(new PassThrough(), {
    isTTY: true,
    setRawMode: noop,
    ref: noop,
    unref: noop
  })

  const stderr = new PassThrough()

  const view = renderSync(
    <ActiveSessionSwitcher
      currentSessionId={null}
      gw={{ request } as unknown as GatewayClient}
      maxWidth={columns}
      onCancel={noop}
      onClose={async () => null}
      onNew={noop}
      onNewPrompt={noop}
      onResume={noop}
      onSelect={noop}
      t={DEFAULT_THEME}
    />,
    {
      stdout: stdout as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stderr: stderr as unknown as NodeJS.WriteStream,
      patchConsole: false,
      exitOnCtrlC: false
    }
  )

  try {
    // Await hydration of the actual component, not merely resolution of RPCs.
    await vi.waitFor(
      () => {
        expect(request).toHaveBeenCalledWith('session.active_list', {
          current_session_id: null
        })
        expect(request).toHaveBeenCalledWith('session.list', { limit: 200 })
        expect(frames.at(-1)).toContain(title)
      },
      { timeout: 3000, interval: 10 }
    )
    const frame = frames.at(-1)!
    const row = frame.split('\n').find(line => line.includes(title))!
    expect(row).toContain(item.id)
    expect(row).not.toContain('PREVIEW_NOT_SOURCE')
    console.log(`RENDER ${kind} source=${JSON.stringify(source) ?? '<missing>'} columns=${columns}\n${frame}`)
    check(row, frame)
  } finally {
    view.unmount()
    view.cleanup()
    stdin.destroy()
    stdout.destroy()
    stderr.destroy()
  }
}

describe('issue86810: session source is visible in real ActiveSessionSwitcher rows', () => {
  it('shows history source feishu alongside its title at normal width', async () => {
    await withSessionRow('history', 'feishu', 120, row => {
      expect(row).toContain('Hist title')
      expect(row.toLowerCase()).toContain('feishu')
    })
  })

  it('shows live source telegram alongside its title at normal width', async () => {
    await withSessionRow('live', 'telegram', 120, row => {
      expect(row).toContain('Live title')
      expect(row.toLowerCase()).toContain('telegr')
    })
  })

  it('preserves the live source when the independent history request fails', async () => {
    await withSessionRow(
      'live',
      'wecom',
      120,
      (row, frame) => {
        expect(row).toContain('wecom')
        expect(frame).toContain('could not load resumable sessions')
      },
      true
    )
  })

  for (const kind of ['history', 'live'] as const) {
    for (const source of [undefined, '']) {
      it(`preserves ${kind} title with ${source === undefined ? 'missing' : 'empty'} source`, async () => {
        await withSessionRow(kind, source, 120, row => {
          expect(row).not.toMatch(/undefined|null|\[\s*\]/i)
          expect(row).not.toMatch(/feishu|telegram/i)
          expect(row).toContain(kind === 'history' ? 'Hist title' : 'Live title')
        })
      })
    }

    it(`keeps both ${kind} source and short title visible at 64 columns`, async () => {
      const source = kind === 'history' ? 'feishu' : 'telegram'
      await withSessionRow(kind, source, 64, row => {
        expect.soft(row).toContain(kind === 'history' ? 'Hist title' : 'Live title')
        expect.soft(row.toLowerCase()).toContain(source.slice(0, 6))
        expect.soft(Array.from(row).length).toBeLessThanOrEqual(64)
      })
    })
  }
})
