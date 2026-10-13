import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $uiState, $uiTheme, getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { GoalBar } from '../components/goalBar.js'

const goalRenderProbe = vi.hoisted(() => vi.fn())

// GoalBar calls this on every render. Count the actual component's render
// path, rather than only testing a computed store that already existed on base.
vi.mock('../app/goalStatus.js', () => ({
  useGoalLine: () => {
    goalRenderProbe()
    return null
  }
}))

afterEach(() => {
  resetUiState()
  goalRenderProbe.mockClear()
})

describe('ui theme subscription', () => {
  it('does not notify the theme projection for unrelated ui-state churn', () => {
    let fullStoreWakeups = 0
    let themeWakeups = 0
    const stopFull = $uiState.listen(() => {
      fullStoreWakeups += 1
    })
    const stopTheme = $uiTheme.listen(() => {
      themeWakeups += 1
    })

    try {
      const fullBaseline = fullStoreWakeups
      const themeBaseline = themeWakeups

      for (let i = 0; i < 20; i++) {
        patchUiState({ status: `streaming-${i}` })
      }

      expect(fullStoreWakeups - fullBaseline).toBe(20)
      expect(themeWakeups - themeBaseline).toBe(0)

      patchUiState({ theme: { ...getUiState().theme } })
      expect(themeWakeups - themeBaseline).toBe(1)
    } finally {
      stopTheme()
      stopFull()
    }
  })

  it('does not rerender the production GoalBar on status updates, but reacts to theme changes', async () => {
    const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 20 })
    const view = renderSync(<GoalBar cols={80} />, {
      patchConsole: false,
      stdin: new PassThrough() as unknown as NodeJS.ReadStream,
      stdout: stdout as unknown as NodeJS.WriteStream,
      stderr: new PassThrough() as unknown as NodeJS.WriteStream
    })

    try {
      await new Promise(resolve => setTimeout(resolve, 0))
      const before = goalRenderProbe.mock.calls.length
      expect(before).toBeGreaterThan(0)

      // Separate ticks so batching cannot mask unnecessary component renders.
      for (let i = 0; i < 8; i++) {
        patchUiState({ status: `render-churn-${i}` })
        await new Promise(resolve => setTimeout(resolve, 0))
      }

      expect(goalRenderProbe).toHaveBeenCalledTimes(before)
      patchUiState({ theme: { ...getUiState().theme } })
      await expect.poll(() => goalRenderProbe.mock.calls.length).toBeGreaterThan(before)
    } finally {
      view.unmount()
    }
  })
})
