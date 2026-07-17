import { beforeEach, describe, expect, it, vi } from 'vitest'

import { sessionCommands } from '../app/slash/commands/session.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import type { SessionUsageResponse } from '../gatewayTypes.js'

const usageCommand = sessionCommands.find(command => command.name === 'usage')!

describe('/usage state refresh', () => {
  beforeEach(() => resetUiState())

  it('preserves and refreshes runtime identity with the counter snapshot', async () => {
    patchUiState({
      usage: {
        calls: 1,
        credential_label: 'personal',
        input: 10,
        model: 'gpt-primary',
        output: 5,
        total: 15
      }
    })

    const response: SessionUsageResponse = {
      calls: 2,
      credential_label: 'work',
      input: 20,
      model: 'gpt-fallback',
      output: 10,
      total: 30
    }

    const rpc = vi.fn(() => Promise.resolve(response))

    const ctx = {
      gateway: { rpc },
      sid: 'sid-usage',
      stale: () => false,
      transcript: { panel: vi.fn(), sys: vi.fn() }
    }

    usageCommand.run('', ctx as any, 'usage')
    await rpc.mock.results[0]?.value
    await Promise.resolve()

    expect(getUiState().usage).toMatchObject({
      calls: 2,
      credential_label: 'work',
      input: 20,
      model: 'gpt-fallback',
      output: 10,
      total: 30
    })
  })
})