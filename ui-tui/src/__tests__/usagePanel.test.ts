import { describe, expect, it } from 'vitest'

import { planUsageLines } from '../app/slash/usagePanel.js'

const plan = (left_fraction: number) =>
  ({ gauges: [{ label: 'day', left_fraction, style: 'bar', title: 'Today' }], kind: 'usage_window' }) as never

describe('planUsageLines', () => {
  it('a nearly full allowance reads 99% beside its 9 cells, like the CLI and Desktop', () => {
    expect(planUsageLines(plan(0.00004))).toEqual(['Today  [||||||||| ]  99% used'])
  })

  it('100% only once the allowance is spent', () => {
    expect(planUsageLines(plan(0))).toEqual(['Today  [||||||||||]  100% used'])
  })
})
