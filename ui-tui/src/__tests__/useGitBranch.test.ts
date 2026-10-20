import { describe, expect, it } from 'vitest'

import { shouldPollGitBranch } from '../hooks/useGitBranch.js'

describe('shouldPollGitBranch', () => {
  it('does not poll when the status bar is hidden', () => {
    expect(shouldPollGitBranch('off', '', null)).toBe(false)
  })

  it('does not poll when an enabled title replaces the cwd label', () => {
    expect(shouldPollGitBranch('top', 'session', null)).toBe(false)
    expect(shouldPollGitBranch('top', 'session', new Set(['title']))).toBe(false)
  })

  it('polls when the cwd label can render', () => {
    expect(shouldPollGitBranch('bottom', '', null)).toBe(true)
    expect(shouldPollGitBranch('bottom', 'session', new Set(['cwd']))).toBe(true)
  })
})
