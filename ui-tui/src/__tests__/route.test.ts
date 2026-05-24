import { describe, expect, it } from 'vitest'

import { providerModelLabel, sessionRouteLabel, subagentRouteLabel } from '../lib/route.js'

describe('route labels', () => {
  it('formats provider/model labels compactly', () => {
    expect(providerModelLabel('openai-codex', 'openai/gpt-5.5')).toBe('openai-codex/gpt 5.5')
    expect(providerModelLabel('', 'anthropic/claude-sonnet-4')).toBe('sonnet 4')
  })

  it('formats main session route metadata for the status bar', () => {
    expect(
      sessionRouteLabel({
        model: 'openai/gpt-5.5',
        profile_name: 'orchestrator',
        provider: 'openai-codex',
        reasoning_effort: 'xhigh',
        route: {
          execution_mode: 'inline',
          model: 'openai/gpt-5.5',
          provider: 'openai-codex',
          reason: 'active session profile',
          reasoning_effort: 'xhigh',
          target_profile: 'orchestrator'
        },
        skills: {},
        tools: {}
      })
    ).toBe('route inline→orchestrator · openai-codex/gpt 5.5 · effort xhigh · reason active session profile')
  })

  it('formats delegated child route metadata for the agents overlay', () => {
    expect(
      subagentRouteLabel({
        executionMode: 'delegate_task',
        model: 'deepseek-v4-pro',
        provider: 'deepseek',
        reasoningEffort: 'low',
        role: 'leaf'
      })
    ).toBe('delegate_task · deepseek/deepseek v4 pro · effort low · leaf')
  })
})
