import { describe, expect, it } from 'vitest'

import { promptAcceptedWithoutEvents } from '../app/useSubmission.js'

describe('promptAcceptedWithoutEvents', () => {
  it('detects an accepted prompt that produced no stream events while still busy', () => {
    expect(promptAcceptedWithoutEvents({ busy: true, sid: 's1' }, 's1', 7, 7)).toBe(true)
  })

  it('does not fire after stream events, session changes, or busy clears', () => {
    expect(promptAcceptedWithoutEvents({ busy: true, sid: 's1' }, 's1', 7, 8)).toBe(false)
    expect(promptAcceptedWithoutEvents({ busy: true, sid: 's2' }, 's1', 7, 7)).toBe(false)
    expect(promptAcceptedWithoutEvents({ busy: false, sid: 's1' }, 's1', 7, 7)).toBe(false)
  })
})
