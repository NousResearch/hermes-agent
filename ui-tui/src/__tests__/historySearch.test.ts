import { mkdtempSync, readFileSync, rmSync, statSync, symlinkSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getOverlayState, patchOverlayState, resetFlowOverlays, resetOverlayState } from '../app/overlayStore.js'
import { historyPreview, searchHistory } from '../lib/historySearch.js'

describe('searchHistory', () => {
  it('ranks newest first, dedupes, and needs every term in any order', () => {
    const entries = ['git status', 'deploy staging', 'git push origin main', 'deploy prod', 'git status']

    expect(searchHistory(entries, '')).toEqual(['git status', 'deploy prod', 'git push origin main', 'deploy staging'])
    expect(searchHistory(entries, 'MAIN git')).toEqual(['git push origin main'])
    expect(searchHistory(entries, 'nothing')).toEqual([])
  })

  it('previews a multi-line entry on one line', () => {
    expect(historyPreview('first line\n  second line')).toBe('first line ⏎ second line')
  })

  it('stays open, query intact, across a turn completing', () => {
    patchOverlayState({ historySearch: { query: 'deploy' } })
    resetFlowOverlays()

    expect(getOverlayState().historySearch).toEqual({ query: 'deploy' })
    resetOverlayState()
  })
})

describe('history file', () => {
  let home: string
  const saved = process.env.HERMES_HOME

  beforeEach(() => {
    home = mkdtempSync(join(process.env.TMPDIR ?? '.', 'hist-'))
    process.env.HERMES_HOME = home
    vi.resetModules()
  })

  afterEach(() => {
    process.env.HERMES_HOME = saved
    rmSync(home, { force: true, recursive: true })
  })

  it.skipIf(process.platform === 'win32')('writes .hermes_history owner-only and tightens an older readable file', async () => {
    const file = join(home, '.hermes_history')
    writeFileSync(file, '\n# 2026-01-01\n+old prompt\n', { mode: 0o644 })
    const history = await import('../lib/history.js')

    expect(history.load()).toEqual(['old prompt'])
    history.append('new prompt')

    expect(statSync(file).mode & 0o777).toBe(0o600)
    expect(readFileSync(file, 'utf8')).toContain('+new prompt')
  })

  it('drops a newest record that outgrew the read window instead of recalling its tail', async () => {
    const huge = Array.from({ length: 20_000 }, (_, i) => `+line ${i} ${'x'.repeat(60)}`).join('\n')
    writeFileSync(join(home, '.hermes_history'), `\n# 2026-01-01\n+old prompt\n\n# 2026-01-02\n${huge}\n`)
    const history = await import('../lib/history.js')

    expect(history.load()).toEqual([])
  })

  it.skipIf(process.platform === 'win32')('refuses to follow a symlink planted at the history path', async () => {
    const target = join(home, 'elsewhere')
    writeFileSync(target, '')
    symlinkSync(target, join(home, '.hermes_history'))
    const history = await import('../lib/history.js')

    history.append('secret prompt')

    expect(readFileSync(target, 'utf8')).toBe('')
  })
})
