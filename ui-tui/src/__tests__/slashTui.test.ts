import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { coreCommands } from '../app/slash/commands/core.js'

const command = () => {
  const cmd = coreCommands.find(c => c.name === 'tui')

  if (!cmd) {
    throw new Error('missing /tui command')
  }

  return cmd
}

const envKeys = ['HERMES_TUI', 'HERMES_TUI_INLINE'] as const
let savedEnv: Record<(typeof envKeys)[number], string | undefined>

beforeEach(() => {
  savedEnv = Object.fromEntries(envKeys.map(key => [key, process.env[key]])) as Record<
    (typeof envKeys)[number],
    string | undefined
  >

  for (const key of envKeys) {
    delete process.env[key]
  }
})

afterEach(() => {
  for (const key of envKeys) {
    const value = savedEnv[key]

    if (value === undefined) {
      delete process.env[key]
    } else {
      process.env[key] = value
    }
  }
})

const runTui = async (arg = '', sid: null | string = 'session-1') => {
  const messages: string[] = []

  const ctx = {
    gateway: {
      rpc: vi.fn(() => Promise.resolve({ key: 'tui', value: arg.trim().toLowerCase() }))
    },
    guarded: (fn: (value: unknown) => void) => fn,
    guardedErr: (err: unknown) => {
      throw err
    },
    sid,
    transcript: {
      sys: (message: string) => messages.push(message)
    }
  } as any

  command().run(arg, ctx, `/tui${arg ? ` ${arg}` : ''}`)
  await Promise.resolve()

  return messages.join('\n')
}

describe('/tui slash command', () => {
  it('reports fullscreen status by default with concrete relaunch commands', async () => {
    process.env.HERMES_TUI = '1'

    const output = await runTui('status')

    expect(output).toContain('renderer: fullscreen')
    expect(output).toContain('HERMES_TUI=1 hermes -c')
    expect(output).toContain('HERMES_TUI=1 HERMES_TUI_INLINE=1 hermes -c')
    expect(output).toContain('HERMES_TUI=0 hermes -c')
  })

  it('reports inline status when HERMES_TUI_INLINE is enabled', async () => {
    process.env.HERMES_TUI = '1'
    process.env.HERMES_TUI_INLINE = '1'

    const output = await runTui('')

    expect(output).toContain('renderer: inline')
    expect(output).toContain('native terminal scrollback captures transcript rows')
  })

  it('prints focused inline/fullscreen/default instructions', async () => {
    expect(await runTui('inline')).toContain('HERMES_TUI=1 HERMES_TUI_INLINE=1 hermes -c')
    expect(await runTui('fullscreen')).toContain('HERMES_TUI=1 hermes -c')
    expect(await runTui('default')).toContain('export HERMES_TUI=1')
  })

  it('uses plain hermes when no session exists', async () => {
    const output = await runTui('fullscreen', null)

    expect(output).toContain('HERMES_TUI=1 hermes')
    expect(output).not.toContain('hermes -c')
  })

  it('rejects unknown modes with usage', async () => {
    expect(await runTui('sideways')).toContain('usage: /tui [status|fullscreen|inline|classic|default]')
  })
})
