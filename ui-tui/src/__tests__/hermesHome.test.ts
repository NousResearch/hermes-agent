import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { describe, expect, it, vi } from 'vitest'

import { resolveHermesHome } from '../lib/hermesHome.js'

const env = (values: Record<string, string>): NodeJS.ProcessEnv => values as NodeJS.ProcessEnv

describe('resolveHermesHome', () => {
  it('answers what hermes_constants answers: HERMES_HOME, else the platform default', () => {
    // Contract with hermes_constants._get_platform_default_hermes_home() — the
    // Python side of the same home. A TUI that disagrees writes state nobody
    // reads: on native Windows the profile home is %LOCALAPPDATA%\hermes, NOT
    // ~/.hermes (which is only the POSIX default).
    expect(resolveHermesHome(env({ HERMES_HOME: '/tmp/profile-a' }), 'linux', '/home/u')).toBe('/tmp/profile-a')
    expect(resolveHermesHome(env({ HERMES_HOME: '   ' }), 'linux', '/home/u')).toBe(join('/home/u', '.hermes'))
    expect(resolveHermesHome(env({}), 'linux', '/home/u')).toBe(join('/home/u', '.hermes'))
    expect(resolveHermesHome(env({}), 'darwin', '/Users/u')).toBe(join('/Users/u', '.hermes'))

    expect(resolveHermesHome(env({ HERMES_HOME: 'D:\\profiles\\a' }), 'win32', 'C:\\Users\\u')).toBe('D:\\profiles\\a')
    expect(resolveHermesHome(env({ LOCALAPPDATA: 'C:\\Users\\u\\AppData\\Local' }), 'win32', 'C:\\Users\\u')).toBe(
      join('C:\\Users\\u\\AppData\\Local', 'hermes')
    )
    // LOCALAPPDATA unset: still the native home, never ~/.hermes.
    expect(resolveHermesHome(env({}), 'win32', 'C:\\Users\\u')).toBe(join('C:\\Users\\u\\AppData', 'Local', 'hermes'))
  })

  it('routes a TUI artifact into the home the profile actually uses', async () => {
    const home = mkdtempSync(join(tmpdir(), 'hermes-home-'))

    vi.stubEnv('HERMES_HOME', home)
    vi.resetModules()

    try {
      const history = await import('../lib/history.js')

      history.append(`resolved-home-probe-${process.pid}`)

      expect(readFileSync(join(home, '.hermes_history'), 'utf8')).toContain(`resolved-home-probe-${process.pid}`)
    } finally {
      vi.unstubAllEnvs()
      rmSync(home, { force: true, recursive: true })
    }
  })
})
