import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'

import React from 'react'
import { expect, it, vi } from 'vitest'

import { DEFAULT_THEME } from '../theme.js'

it('an open approval owns Enter: history search neither recalls nor closes under it', async () => {
  const home = mkdtempSync(join(process.env.TMPDIR ?? '.', 'hist-prompt-'))
  const saved = process.env.HERMES_HOME
  process.env.HERMES_HOME = home
  writeFileSync(join(home, '.hermes_history'), '\n# 2026-01-01\n+git push origin main\n')
  vi.resetModules()
  // Fresh modules so the history file is read from the temp HERMES_HOME; Ink and the store come
  // from the same graph the component reads.
  const { renderSync } = await import('@hermes/ink')
  const { HistorySearch } = await import('../components/historySearch.js')
  const { patchOverlayState, resetOverlayState } = await import('../app/overlayStore.js')
  const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: vi.fn(), ref: vi.fn(), unref: vi.fn() })
  const onAccept = vi.fn()
  const onClose = vi.fn()
  resetOverlayState()
  patchOverlayState({ approval: { command: 'rm -rf build', description: 'delete' } as any, historySearch: { query: '' } })

  const instance = renderSync(<HistorySearch onAccept={onAccept} onClose={onClose} t={DEFAULT_THEME} />, {
    stdin: stdin as any,
    stdout: Object.assign(new PassThrough(), { columns: 100, rows: 30, isTTY: false }) as any,
    stderr: new PassThrough() as any,
    patchConsole: false
  })

  try {
    stdin.write('\r')
    await new Promise(resolve => setTimeout(resolve, 50))
    expect(onAccept).not.toHaveBeenCalled()
    expect(onClose).not.toHaveBeenCalled()

    // Once the question is answered the panel takes keys again.
    patchOverlayState({ approval: null })
    await vi.waitFor(() => {
      stdin.write('\r')
      expect(onAccept).toHaveBeenCalledWith('git push origin main')
    })
  } finally {
    instance.unmount()
    resetOverlayState()

    if (saved === undefined) {
      delete process.env.HERMES_HOME
    } else {
      process.env.HERMES_HOME = saved
    }

    rmSync(home, { force: true, recursive: true })
  }
})
