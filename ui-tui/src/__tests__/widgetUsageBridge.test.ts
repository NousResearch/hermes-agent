import { afterEach, describe, expect, it } from 'vitest'

import { patchUiState, resetUiState } from '../app/uiStore.js'
import { getUiUsage, widgetSdk } from '../sdk/userWidgets.js'

const usage = (over: Record<string, unknown> = {}) => ({
  calls: 0,
  input: 0,
  output: 0,
  total: 0,
  ...over
})

afterEach(() => resetUiState())

describe('widget usage bridge', () => {
  it('exposes the usage selector + hook on the widget SDK', () => {
    expect(typeof widgetSdk.getUiUsage).toBe('function')
    expect(typeof widgetSdk.useUiUsage).toBe('function')
  })

  it('getUiUsage reflects live usage changes pushed to the ui store', () => {
    expect(getUiUsage()).toEqual(usage())

    patchUiState({ usage: usage({ input: 1200, output: 340, stream_tps: 42.5, avg_latency_s: 1.2 }) })

    expect(getUiUsage()).toMatchObject({
      input: 1200,
      output: 340,
      stream_tps: 42.5,
      avg_latency_s: 1.2
    })
  })
})
