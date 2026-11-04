import { PassThrough } from 'node:stream'

import { Box, renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { InputHandlerContext } from '../app/interfaces.js'
import { $overlayState, resetOverlayState } from '../app/overlayStore.js'
import { setSlashCatalog } from '../app/slashCatalog.js'
import { isCommandPaletteShortcut, useInputHandlers } from '../app/useInputHandlers.js'
import {
  CommandPalette,
  commandPaletteEntries,
  type CommandPaletteEntry,
  rankCommandPaletteEntries
} from '../components/commandPalette.js'
import { DEFAULT_THEME } from '../theme.js'

const entries: CommandPaletteEntry[] = [
  { category: 'Session', command: '/compress', description: 'Compress the conversation' },
  { category: 'Model', command: '/model', description: 'Switch model' },
  { category: 'Session', command: '/clear', description: 'Clear the screen' }
]

afterEach(() => {
  setSlashCatalog(null)
  resetOverlayState()
})

function CommandPaletteHotkeyHarness() {
  useInputHandlers({
    actions: {},
    composer: {
      actions: {},
      refs: {
        historyDraftRef: { current: '' },
        historyRef: { current: [] },
        queueEditRef: { current: null },
        queueRef: { current: [] },
        submitRef: { current: () => {} },
        tokensRef: { current: [] }
      },
      state: {
        compIdx: 0,
        compReplace: 0,
        completions: [],
        historyIdx: null,
        input: '',
        inputBuf: [],
        queueEditIdx: null,
        queuedDisplay: [],
        tokens: []
      }
    },
    gateway: { gw: {}, rpc: async () => null },
    terminal: {
      hasSelection: false,
      scrollRef: { current: null },
      scrollWithSelection: () => {},
      selection: {}
    },
    voice: {
      enabled: false,
      recordKey: { key: 'b', modifiers: ['ctrl'] },
      recording: false,
      setProcessing: () => {},
      setRecording: () => {},
      setVoiceEnabled: () => {},
      setVoiceTts: () => {}
    },
    wheelStep: 3
  } as unknown as InputHandlerContext)

  return null
}

describe('command palette ranking', () => {
  it('retains skills that only appear in the catalog pairs list', () => {
    const rows = commandPaletteEntries({
      canon: { '/bg': '/background' },
      categories: [{ name: 'Session', pairs: [['/background', 'Run in background']] }],
      pairs: [['/background', 'Run in background'], ['/my-skill', 'Project skill']],
      skillCount: 1,
      sub: {}
    })

    expect(rows).toEqual([
      { aliases: ['/bg'], category: 'Session', command: '/background', description: 'Run in background' },
      { category: 'Skills & extensions', command: '/my-skill', description: 'Project skill' }
    ])
  })

  it('ranks earlier contiguous command-name matches above scattered hits', () => {
    const ranked = rankCommandPaletteEntries(
      [
        { category: 'Other', command: '/compare', description: 'Scattered cmp hit' },
        { category: 'Other', command: '/cmp', description: 'Exact compact hit' },
        { category: 'Other', command: '/compress', description: 'Contiguous prefix hit' }
      ],
      'cmp'
    )

    expect(ranked.map(row => row.command)).toEqual(['/cmp', '/compare', '/compress'])
  })

  it('keeps command-name matches above description-only matches', () => {
    const ranked = rankCommandPaletteEntries(
      [
        { category: 'Session', command: '/goal', description: 'Show model status' },
        { category: 'Model', command: '/model', description: 'Switch provider' }
      ],
      'model'
    )

    expect(ranked.map(row => row.command)).toEqual(['/model', '/goal'])
  })

  it('matches aliases but returns the canonical command', () => {
    const ranked = rankCommandPaletteEntries(
      [{ aliases: ['/bg'], category: 'Session', command: '/background', description: 'Run in background' }],
      'bg'
    )

    expect(ranked.map(row => row.command)).toEqual(['/background'])
  })
})

describe('CommandPalette interaction', () => {
  const mount = () => {
    setSlashCatalog({
      canon: {},
      categories: [
        { name: 'Session', pairs: [['/compress', 'Compress the conversation'], ['/clear', 'Clear the screen']] },
        { name: 'Model', pairs: [['/model', 'Switch model']] }
      ],
      pairs: entries.map(row => [row.command, row.description]),
      skillCount: 0,
      sub: {}
    })

    const onCancel = vi.fn()
    const onSelect = vi.fn()
    const stdout = Object.assign(new PassThrough(), { columns: 100, rows: 30, isTTY: false })
    const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: () => {}, ref: () => {}, unref: () => {} })
    let output = ''
    stdout.on('data', chunk => {
      output += chunk.toString()
    })
    const view = renderSync(
      <Box height={30}>
        <CommandPalette maxWidth={90} onCancel={onCancel} onSelect={onSelect} t={DEFAULT_THEME} />
      </Box>,
      {
        stdout: stdout as unknown as NodeJS.WriteStream,
        stdin: stdin as unknown as NodeJS.ReadStream,
        stderr: new PassThrough() as unknown as NodeJS.WriteStream,
        patchConsole: false
      }
    )

    return { getOutput: () => output, onCancel, onSelect, stdin, view }
  }

  it('opens from the composer Ctrl+P hotkey', async () => {
    const stdout = Object.assign(new PassThrough(), { columns: 100, rows: 30, isTTY: false })
    const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: () => {}, ref: () => {}, unref: () => {} })
    const view = renderSync(<CommandPaletteHotkeyHarness />, {
      stdout: stdout as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stderr: new PassThrough() as unknown as NodeJS.WriteStream,
      patchConsole: false
    })

    try {
      stdin.write('\x10')
      await vi.waitFor(() => expect($overlayState.get().commandPalette).toBe(true))
    } finally {
      view.unmount()
      view.cleanup()
    }
  })

  it('does not claim modified Ctrl+P variants', () => {
    expect(isCommandPaletteShortcut({ ctrl: true, meta: false, shift: false, super: false }, 'p')).toBe(true)
    expect(isCommandPaletteShortcut({ ctrl: true, meta: false, shift: true, super: false }, 'p')).toBe(false)
    expect(isCommandPaletteShortcut({ ctrl: true, meta: true, shift: false, super: false }, 'p')).toBe(false)
    expect(isCommandPaletteShortcut({ ctrl: true, meta: false, shift: false, super: true }, 'p')).toBe(false)
  })

  it('filters and dispatches the selected command through Enter', async () => {
    const { getOutput, onSelect, stdin, view } = mount()

    try {
      stdin.write('mdl')
      await vi.waitFor(() => expect(getOutput()).toContain('/model'))
      stdin.write('\r')
      await vi.waitFor(() => expect(onSelect).toHaveBeenCalledWith('/model'))
    } finally {
      view.unmount()
      view.cleanup()
    }
  })

  it('navigates with arrows and dismisses with Escape', async () => {
    const { onCancel, onSelect, stdin, view } = mount()

    try {
      stdin.write('\x1b[B')
      await new Promise(resolve => setTimeout(resolve, 20))
      stdin.write('\r')
      await vi.waitFor(() => expect(onSelect).toHaveBeenCalledWith('/model'))

      stdin.write('\x1b')
      await vi.waitFor(() => expect(onCancel).toHaveBeenCalled())
    } finally {
      view.unmount()
      view.cleanup()
    }
  })
})
