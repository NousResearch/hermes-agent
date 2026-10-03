// Regression for #5511: `/keys` must be a TUI-local command that reuses the
// single hotkey table `/help` renders, not a parallel copy or a backend
// fallback (the backend COMMAND_REGISTRY has no `keys` entry, so an
// unrecognized `/keys` would silently dispatch to the slash worker).
import { describe, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { findSlashCommand } from '../app/slash/registry.js'
import { resetUiState } from '../app/uiStore.js'
import { hotkeys } from '../content/hotkeys.js'

describe('/keys', () => {
  it('renders the same hotkey rows /help renders, and never hits the slash worker', () => {
    resetUiState()
    const panel = vi.fn()
    const request = vi.fn(() => Promise.resolve({}))

    const ctx = {
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
        maybeWarn: vi.fn(),
        setCatalog: vi.fn()
      },
      session: {
        closeSession: vi.fn(() => Promise.resolve(null)),
        die: vi.fn(),
        dieWithCode: vi.fn(),
        guardBusySessionSwitch: vi.fn(() => false),
        newLiveSession: vi.fn(),
        newSession: vi.fn(),
        resetVisibleHistory: vi.fn(),
        resumeById: vi.fn(),
        setSessionStartedAt: vi.fn()
      },
      transcript: {
        page: vi.fn(),
        panel,
        send: vi.fn(),
        setHistoryItems: vi.fn(),
        sys: vi.fn(),
        trimLastExchange: vi.fn((items: unknown[]) => items)
      },
      voice: { setVoiceEnabled: vi.fn(), setVoiceRecordKey: vi.fn(), setVoiceTts: vi.fn() }
    }

    expect(createSlashHandler(ctx as never)('/keys')).toBe(true)

    const sections = panel.mock.calls[0]![1] as { rows?: [string, string][]; title?: string }[]
    const rows = sections.flatMap(section => section.rows ?? [])

    // Relation, not a snapshot: the /keys panel and the /help hotkey section
    // read the SAME table, so a binding added to one shows up in the other.
    expect(rows).toEqual(hotkeys())
    expect(rows.length).toBeGreaterThan(0)
    expect(request.mock.calls.filter(([method]) => method !== 'shared_metrics.slash_command')).toEqual([])
    expect(findSlashCommand('keys')?.name).toBe('keys')
  })
})
