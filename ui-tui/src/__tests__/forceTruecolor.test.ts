import { describe, expect, it } from 'vitest'

const ENV_KEYS = ['COLORTERM', 'FORCE_COLOR', 'HERMES_TUI_TRUECOLOR', 'NO_COLOR'] as const

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
  it('sets COLORTERM=truecolor and FORCE_COLOR=3 when unset', async () => {
    await withCleanEnv(
      () => {},
      async () => {
        await import('../lib/forceTruecolor.js?t=' + Date.now())
        expect(process.env.COLORTERM).toBe('truecolor')
        expect(process.env.FORCE_COLOR).toBe('3')
      }
    )
  })

  it('respects HERMES_TUI_TRUECOLOR=0 opt-out', async () => {
    await withCleanEnv(
      () => {
        process.env.HERMES_TUI_TRUECOLOR = '0'
      },
      async () => {
        await import('../lib/forceTruecolor.js?t=optout-' + Date.now())
        expect(process.env.COLORTERM).toBeUndefined()
        expect(process.env.FORCE_COLOR).toBeUndefined()
      }
    )
  })

  it('respects NO_COLOR', async () => {
    await withCleanEnv(
      () => {
        process.env.NO_COLOR = '1'
      },
      async () => {
        await import('../lib/forceTruecolor.js?t=no-color-' + Date.now())
        expect(process.env.COLORTERM).toBeUndefined()
        expect(process.env.FORCE_COLOR).toBeUndefined()
      }
    )
  })
})
