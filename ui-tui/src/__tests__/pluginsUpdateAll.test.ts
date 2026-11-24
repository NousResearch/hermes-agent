import { beforeEach, describe, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { resetOverlayState } from '../app/overlayStore.js'
import { UPDATE_ALL_TIMEOUT_MS } from '../app/slash/commands/pluginsUpdateAll.js'
import { resetUiState } from '../app/uiStore.js'

// `/plugins update` used to ride `slash.exec`, whose worker is killed after 45 s mid-update. It must
// go to the long-budget `plugins.manage update_all` RPC instead, and render that RPC's rows.
describe('/plugins update', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    resetOverlayState()
    resetUiState()
  })

  it('runs the batch over plugins.manage update_all, never the slash worker', async () => {
    const request = vi.fn((method: string) =>
      Promise.resolve(
        method === 'plugins.manage'
          ? {
              results: [
                { name: 'fresh', status: 'updated', detail: '' },
                { name: 'widened', status: 'needs_review', detail: 'adds tools' }
              ],
              skipped: [{ name: 'held', reason: 'held with --ref' }]
            }
          : {}
      )
    )

    const ctx = buildCtx(request)

    expect(createSlashHandler(ctx as never)('/plugins update')).toBe(true)
    await vi.waitFor(() => expect(ctx.transcript.page).toHaveBeenCalled())

    const work = request.mock.calls.filter(([method]) => method !== 'shared_metrics.slash_command')
    expect(work).toEqual([['plugins.manage', { action: 'update_all' }, UPDATE_ALL_TIMEOUT_MS]])
    expect(UPDATE_ALL_TIMEOUT_MS).toBeGreaterThan(45_000)

    const text = ctx.transcript.page.mock.calls[0]![0] as string
    expect(text).toContain('✓ updated  fresh')
    expect(text).toContain('! needs review  widened  (adds tools)')
    expect(text).toContain('skipped held: held with --ref')
  })

  it('keeps other subcommands on the slash worker', async () => {
    const request = vi.fn(() => Promise.resolve({ output: 'ok' }))
    const ctx = buildCtx(request)

    createSlashHandler(ctx as never)('/plugins auto-update on')
    await vi.waitFor(() => expect(request).toHaveBeenCalledWith('slash.exec', expect.anything()))
    expect(request).not.toHaveBeenCalledWith('plugins.manage', expect.anything(), expect.anything())
  })
})

function buildCtx(request: ReturnType<typeof vi.fn>) {
  return {
    slashFlightRef: { current: 0 },
    composer: {
      enqueue: vi.fn(),
      hasSelection: false,
      openEditor: vi.fn(async () => {}),
      paste: vi.fn(),
      queueRef: { current: [] as string[] },
      selection: { copySelection: vi.fn(async () => '') },
      setInput: vi.fn()
    },
    gateway: { gw: { getLogTail: vi.fn(() => ''), kill: vi.fn(), request }, rpc: vi.fn(() => Promise.resolve({})) },
    local: {
      catalog: null,
      getHistoryItems: vi.fn(() => []),
      getLastUserMsg: vi.fn(() => ''),
      maybeWarn: vi.fn()
    },
    session: {
      closeSession: vi.fn(() => Promise.resolve(null)),
      die: vi.fn(),
      guardBusySessionSwitch: vi.fn(() => false),
      newSession: vi.fn(),
      resetVisibleHistory: vi.fn(),
      resumeById: vi.fn(),
      setSessionStartedAt: vi.fn()
    },
    transcript: {
      page: vi.fn(),
      panel: vi.fn(),
      send: vi.fn(),
      setHistoryItems: vi.fn(),
      sys: vi.fn(),
      trimLastExchange: vi.fn((items: unknown[]) => items)
    },
    voice: { setVoiceEnabled: vi.fn() }
  }
}
