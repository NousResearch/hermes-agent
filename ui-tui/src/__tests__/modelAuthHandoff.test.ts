import { describe, expect, it, vi } from 'vitest'

import { runModelAuthHandoff } from '../lib/modelAuthHandoff.js'

describe('runModelAuthHandoff', () => {
  it('suspends Ink while the interactive model wizard owns the terminal', async () => {
    const calls: string[] = []

    const launcher = vi.fn(async (args: string[]) => {
      calls.push(`launch:${args.join(' ')}`)

      return { code: 0 }
    })

    const suspend = vi.fn(async (run: () => Promise<void>) => {
      calls.push('suspend:start')

      await run()
      calls.push('suspend:end')
    })

    await expect(runModelAuthHandoff({ launcher, suspend })).resolves.toEqual({ code: 0 })
    expect(launcher).toHaveBeenCalledWith(['model'])
    expect(calls).toEqual(['suspend:start', 'launch:model', 'suspend:end'])
  })

  it('returns launcher failures for the picker to surface', async () => {
    const result = { code: null, error: 'spawn failed' }

    await expect(
      runModelAuthHandoff({
        launcher: vi.fn(async () => result),
        suspend: async run => run()
      })
    ).resolves.toEqual(result)
  })
})
