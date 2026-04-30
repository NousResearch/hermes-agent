import { describe, expect, it } from 'vitest'

const ENV_KEYS = ['COLORTERM', 'FORCE_COLOR', 'HERMES_TUI_TRUECOLOR', 'NO_COLOR', 'TERM', 'TERM_PROGRAM'] as const
let importId = 0

async function withCleanEnv(setup: () => void, body: () => Promise<void>) {
  const saved: Record<string, string | undefined> = {}

  for (const k of ENV_KEYS) {
    saved[k] = process.env[k]
    delete process.env[k]
  }

  try {
    setup()
    await body()
  } finally {
    for (const k of ENV_KEYS) {
      if (saved[k] === undefined) {
        delete process.env[k]
      } else {
        process.env[k] = saved[k]
      }
    }
  }
}

describe('forceTruecolor', () => {
  it('does not force truecolor by default', async () => {
    await withCleanEnv(
      () => {},
      async () => {
        await import('../lib/forceTruecolor.js?t=default-' + importId++)
        expect(process.env.COLORTERM).toBeUndefined()
        expect(process.env.FORCE_COLOR).toBeUndefined()
      }
    )
  })

  it('does not infer truecolor from Apple Terminal on pre-Tahoe macOS', async () => {
    await withCleanEnv(
      () => {
        process.env.TERM_PROGRAM = 'Apple_Terminal'
        process.env.TERM = 'xterm-256color'
      },
      async () => {
        const mod = await import('../lib/forceTruecolor.js?t=apple-' + importId++)
        expect(mod.shouldForceTruecolor({ TERM_PROGRAM: 'Apple_Terminal' })).toBe(false)
        expect(process.env.COLORTERM).toBeUndefined()
        expect(process.env.FORCE_COLOR).toBeUndefined()
      }
    )
  })

  it('sets COLORTERM=truecolor and FORCE_COLOR=3 when explicitly enabled', async () => {
    await withCleanEnv(
      () => {
        process.env.HERMES_TUI_TRUECOLOR = '1'
      },
      async () => {
        await import('../lib/forceTruecolor.js?t=enabled-' + importId++)
        expect(process.env.COLORTERM).toBe('truecolor')
        expect(process.env.FORCE_COLOR).toBe('3')
      }
    )
  })

  it('respects HERMES_TUI_TRUECOLOR=0 opt-out', async () => {
    await withCleanEnv(
      () => {
        process.env.HERMES_TUI_TRUECOLOR = '0'
        process.env.TERM_PROGRAM = 'Apple_Terminal'
      },
      async () => {
        await import('../lib/forceTruecolor.js?t=optout-' + importId++)
        expect(process.env.COLORTERM).toBeUndefined()
        expect(process.env.FORCE_COLOR).toBeUndefined()
      }
    )
  })

  it('respects NO_COLOR', async () => {
    await withCleanEnv(
      () => {
        process.env.NO_COLOR = '1'
        process.env.HERMES_TUI_TRUECOLOR = '1'
      },
      async () => {
        await import('../lib/forceTruecolor.js?t=no-color-' + importId++)
        expect(process.env.COLORTERM).toBeUndefined()
        expect(process.env.FORCE_COLOR).toBeUndefined()
      }
    )
  })

  it('respects existing FORCE_COLOR unless Hermes truecolor is explicit', async () => {
    await withCleanEnv(
      () => {
        process.env.FORCE_COLOR = ''
      },
      async () => {
        const mod = await import('../lib/forceTruecolor.js?t=force-color-' + importId++)
        expect(mod.shouldForceTruecolor(process.env)).toBe(false)
        expect(process.env.COLORTERM).toBeUndefined()
        expect(process.env.FORCE_COLOR).toBe('')
      }
    )
  })

  it('lets explicit Hermes truecolor override existing FORCE_COLOR', async () => {
    await withCleanEnv(
      () => {
        process.env.FORCE_COLOR = '0'
        process.env.HERMES_TUI_TRUECOLOR = '1'
      },
      async () => {
        await import('../lib/forceTruecolor.js?t=explicit-force-' + importId++)
        expect(process.env.COLORTERM).toBe('truecolor')
        expect(process.env.FORCE_COLOR).toBe('3')
      }
    )
  })
})
