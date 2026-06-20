import { describe, expect, it } from 'vitest'

import { submittedValueCanUseComposerCompletion } from '../app/useSubmission.js'

describe('submittedValueCanUseComposerCompletion', () => {
  it('ignores stale slash completions when editor submits saved text', () => {
    expect(
      submittedValueCanUseComposerCompletion({
        completionCount: 1,
        composerInput: '/editor',
        value: 'run this saved prompt'
      })
    ).toBe(false)
  })

  it('allows completion only for the current composer draft', () => {
    expect(
      submittedValueCanUseComposerCompletion({
        completionCount: 1,
        composerInput: '/edi',
        value: '/edi'
      })
    ).toBe(true)
  })

  it('does not apply completion when there are no completions', () => {
    expect(
      submittedValueCanUseComposerCompletion({
        completionCount: 0,
        composerInput: '/edi',
        value: '/edi'
      })
    ).toBe(false)
  })
})
