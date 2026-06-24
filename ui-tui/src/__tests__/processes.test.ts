import { describe, expect, it } from 'vitest'

import { countRunningProcesses, visibleBackgroundTaskCount } from '../domain/processes.js'

describe('countRunningProcesses', () => {
  it('counts only running registry entries', () => {
    expect(
      countRunningProcesses([
        { session_id: 'proc-1', status: 'running' },
        { session_id: 'proc-2', status: 'exited' },
        { session_id: 'proc-3', status: 'running' },
        { session_id: 'proc-4', status: 'killed' }
      ])
    ).toBe(2)
  })
})

describe('visibleBackgroundTaskCount', () => {
  it('keeps registry-backed processes visible after the in-memory task set is smaller', () => {
    expect(visibleBackgroundTaskCount(0, 2)).toBe(2)
    expect(visibleBackgroundTaskCount(1, 3)).toBe(3)
  })
})
