import { describe, expect, it } from 'vitest'

import { type BootTheme, invalidateBootBackground, seedBootEnvironment } from '../lib/themeBoot.js'
import { defaultTheme } from '../theme.js'

// Review on #20379 (finding 2): the boot cache seeds the previous session's
// background into HERMES_TUI_BACKGROUND, which detectLightMode treats as a
// CURRENT signal. Without provenance, a stale light cache pins a now-dark
// terminal to light indefinitely (the current probe's pure-black answer is
// distrusted, the pure-white foreground is distrusted, and the macOS
// fallback refuses to run while the slot is occupied). These tests cover
// the seed/invalidate contract the gateway handler drives.

const cache = (over: Partial<BootTheme> = {}): BootTheme => ({ theme: defaultTheme, ...over })

describe('seedBootEnvironment', () => {
  it('seeds the cached background when no explicit signal outranks it', () => {
    const env: NodeJS.ProcessEnv = {}

    const seeded = seedBootEnvironment(cache({ background: '#ffffff' }), env)

    expect(env.HERMES_TUI_BACKGROUND).toBe('#ffffff')
    expect(seeded).toEqual({ seededBackground: '#ffffff', seededPin: false })
  })

  it('never seeds over explicit user signals', () => {
    for (const preset of [{ HERMES_TUI_THEME: 'dark' }, { HERMES_TUI_LIGHT: '1' }] as NodeJS.ProcessEnv[]) {
      const env = { ...preset }
      const seeded = seedBootEnvironment(cache({ background: '#ffffff', mode: 'light' }), env)

      expect(env.HERMES_TUI_BACKGROUND).toBeUndefined()
      expect(seeded).toEqual({ seededBackground: null, seededPin: false })
    }

    // A user-exported background keeps its value; only the pin may seed.
    const env: NodeJS.ProcessEnv = { HERMES_TUI_BACKGROUND: '#123456' }

    expect(seedBootEnvironment(cache({ background: '#ffffff' }), env).seededBackground).toBeNull()
    expect(env.HERMES_TUI_BACKGROUND).toBe('#123456')
  })

  it('never seeds the untrusted pure-black fingerprint', () => {
    const env: NodeJS.ProcessEnv = {}

    const seeded = seedBootEnvironment(cache({ background: '#000000' }), env)

    expect(env.HERMES_TUI_BACKGROUND).toBeUndefined()
    expect(seeded.seededBackground).toBeNull()
  })

  it('replays a cached config pin coherently with the physical background', () => {
    // "/theme light" pinned while the physical terminal is dark: the cache
    // stores BOTH — replaying only the background would resolve the first
    // skin dark and recreate the light → dark → light flash.
    const env: NodeJS.ProcessEnv = {}

    const seeded = seedBootEnvironment(cache({ background: '#1e1e1e', mode: 'light' }), env)

    expect(env.HERMES_TUI_THEME).toBe('light')
    expect(env.HERMES_TUI_BACKGROUND).toBe('#1e1e1e')
    expect(seeded).toEqual({ seededBackground: '#1e1e1e', seededPin: true })
  })

  it('replays a pinned dark on a light physical background too', () => {
    const env: NodeJS.ProcessEnv = {}

    const seeded = seedBootEnvironment(cache({ background: '#ffffff', mode: 'dark' }), env)

    expect(env.HERMES_TUI_THEME).toBe('dark')
    expect(env.HERMES_TUI_BACKGROUND).toBe('#ffffff')
    expect(seeded.seededPin).toBe(true)
  })

  it('is a no-op without a cache', () => {
    const env: NodeJS.ProcessEnv = {}

    expect(seedBootEnvironment(null, env)).toEqual({ seededBackground: null, seededPin: false })
    expect(env).toEqual({})
  })
})

describe('invalidateBootBackground', () => {
  it('clears the slot while it still holds the seeded value (stale light cache, current dark terminal)', () => {
    const env: NodeJS.ProcessEnv = {}

    seedBootEnvironment(cache({ background: '#ffffff' }), env)

    // Current terminal answers OSC-11 with distrusted #000000 → the handler
    // invalidates: the slot must clear so foreground / COLORFGBG / macOS
    // appearance / the default get their turn.
    expect(invalidateBootBackground(env)).toBe(true)
    expect(env.HERMES_TUI_BACKGROUND).toBeUndefined()

    // Idempotent: a second distrusted answer has nothing left to clear.
    expect(invalidateBootBackground(env)).toBe(false)
  })

  it('clears a stale dark cache on a current ambiguous terminal the same way', () => {
    const env: NodeJS.ProcessEnv = {}

    seedBootEnvironment(cache({ background: '#1e1e1e' }), env)

    expect(invalidateBootBackground(env)).toBe(true)
    expect(env.HERMES_TUI_BACKGROUND).toBeUndefined()
  })

  it('leaves a trusted OSC answer that overwrote the seed alone', () => {
    const env: NodeJS.ProcessEnv = {}

    seedBootEnvironment(cache({ background: '#ffffff' }), env)

    // A real OSC-11 measurement replaced the hint — it is authoritative.
    env.HERMES_TUI_BACKGROUND = '#282828'

    expect(invalidateBootBackground(env)).toBe(false)
    expect(env.HERMES_TUI_BACKGROUND).toBe('#282828')
  })

  it('is a no-op when nothing was seeded', () => {
    const env: NodeJS.ProcessEnv = { HERMES_TUI_BACKGROUND: '#ffffff' }

    seedBootEnvironment(null, env)

    expect(invalidateBootBackground(env)).toBe(false)
    expect(env.HERMES_TUI_BACKGROUND).toBe('#ffffff')
  })
})
