import { beforeEach, describe, expect, it, vi } from 'vitest'

import { resetDelegationState } from '../app/delegationStore.js'
import { opsCommands } from '../app/slash/commands/ops.js'
import { formatDelegationCaps } from '../components/agentsOverlay.js'

const agentsCommand = opsCommands.find(command => command.name === 'agents')!

describe('formatDelegationCaps', () => {
  beforeEach(() => resetDelegationState())

  it('shows a zero depth cap as delegation disabled', () => {
    expect(formatDelegationCaps(0, 3)).toBe('delegation disabled (d0)')
  })

  it('/agents status fetches fresh caps and reports zero depth as disabled', async () => {
    const rpc = vi.fn().mockResolvedValue({ max_concurrent_children: 3, max_spawn_depth: 0, paused: false })
    const sys = vi.fn()
    const guarded = <T,>(fn: (result: T) => void) => (result: null | T) => result && fn(result)

    agentsCommand.run(
      'status',
      {
        gateway: { rpc },
        guarded,
        guardedErr: vi.fn(),
        transcript: { sys }
      } as any,
      'agents'
    )
    await Promise.resolve()
    await Promise.resolve()

    expect(rpc).toHaveBeenCalledWith('delegation.status', {})
    expect(sys).toHaveBeenCalledWith('delegation disabled (d0)')
  })
})
