import { describe, expect, it } from 'vitest'

import { feedbackStatusLine } from '../components/appChrome.js'

describe('feedbackStatusLine', () => {
  const empty = {
    activity: [],
    outcome: '',
    pendingTools: [],
    todos: [],
    tools: [],
    turnTrail: []
  }

  it('prioritizes active tool feedback over lower-priority progress sources', () => {
    expect(
      feedbackStatusLine({
        ...empty,
        activity: [{ id: 1, text: 'thinking about the next step', tone: 'info' }],
        outcome: 'done',
        pendingTools: ['pending terminal'],
        todos: [{ id: 't1', content: 'finish the task', status: 'in_progress' }],
        tools: [{ id: 'tool-1', name: 'terminal', context: 'pytest tests/run_agent/test_run_agent.py' }],
        turnTrail: ['trail item']
      })
    ).toBe('tool: Terminal("pytest tests/run_agent/test_run_agent.py")')
  })

  it('falls back to todo and outcome when no live feedback exists', () => {
    expect(
      feedbackStatusLine({
        ...empty,
        todos: [{ id: 't1', content: 'verify the status bar feedback HUD', status: 'pending' }]
      })
    ).toBe('todo: verify the status bar feedback HUD')

    expect(feedbackStatusLine({ ...empty, outcome: 'waiting for model output' })).toBe('waiting for model output')
  })
})
