import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, describe, expect, it, vi } from 'vitest'

const homes: string[] = []

async function freshHistory(temporary: boolean) {
  const home = mkdtempSync(join(tmpdir(), 'hermes-tui-history-'))
  homes.push(home)
  writeFileSync(join(home, '.hermes_history'), '\n# 2026-10-16 10:00:00\n+an older saved prompt\n')
  vi.resetModules()
  vi.stubEnv('HERMES_HOME', home)
  vi.stubEnv('HERMES_TUI_TEMPORARY', temporary ? '1' : '')

  return { file: join(home, '.hermes_history'), history: await import('./history.js') }
}

afterEach(() => {
  vi.unstubAllEnvs()
  homes.splice(0).forEach(home => rmSync(home, { force: true, recursive: true }))
})

describe('input history', () => {
  it('appends prompts to .hermes_history', async () => {
    const { file, history } = await freshHistory(false)

    history.append('a normal prompt')
    expect(readFileSync(file, 'utf8')).toContain('+a normal prompt')
  })

  it('recalls but never writes prompts in a temporary chat', async () => {
    const { file, history } = await freshHistory(true)

    history.append('my secret prompt')
    expect(history.load()).toEqual(['an older saved prompt', 'my secret prompt'])
    expect(readFileSync(file, 'utf8')).not.toContain('secret')
  })
})
