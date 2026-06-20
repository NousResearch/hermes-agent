import { beforeEach, describe, expect, it } from 'vitest'

import { turnController } from '../app/turnController.js'
import { getTurnState, resetTurnState } from '../app/turnStore.js'

describe('turnController subagent metadata', () => {
  beforeEach(() => {
    resetTurnState()
    turnController.fullReset()
  })

  it('preserves profile, lane, provider, and model across partial subagent events', () => {
    turnController.upsertSubagent(
      {
        goal: 'review patch',
        lane: 'leaf',
        model: 'kimi-k2.6',
        profile: 'reviewer',
        provider: 'opencode-go',
        subagent_id: 'sa_meta',
        task_index: 0,
        toolsets: ['terminal', 'file']
      },
      () => ({ status: 'running' })
    )

    turnController.upsertSubagent(
      {
        goal: 'review patch',
        subagent_id: 'sa_meta',
        task_index: 0,
        tool_count: 1
      },
      current => ({ tools: [...current.tools, 'terminal pytest'] })
    )

    expect(getTurnState().subagents).toHaveLength(1)
    expect(getTurnState().subagents[0]).toMatchObject({
      id: 'sa_meta',
      lane: 'leaf',
      model: 'kimi-k2.6',
      profile: 'reviewer',
      provider: 'opencode-go',
      toolsets: ['terminal', 'file']
    })
  })
})
