import { describe, expect, it } from 'vitest'

import { gaugeBarStyle, hasUsageGauge, splitUsageGaugeLine } from '../lib/usageGauge.js'

describe('usage gauge coloring', () => {
  it('maps used fill like the model picker (green / yellow / red)', () => {
    expect(gaugeBarStyle('[          ]')).toBe('green')
    expect(gaugeBarStyle('[|         ]')).toBe('green')
    expect(gaugeBarStyle('[|||||     ]')).toBe('green')
    // Thresholds are `used > 0.5` → yellow and `used > 0.8` → red (curses_ui.py).
    expect(gaugeBarStyle('[||||||    ]')).toBe('yellow')
    expect(gaugeBarStyle('[||||||||  ]')).toBe('yellow')
    expect(gaugeBarStyle('[||||||||| ]')).toBe('red')
    expect(gaugeBarStyle('[||||||||||]')).toBe('red')
  })

  it('recognises a gauge row', () => {
    expect(hasUsageGauge('Today      [||        ]  24% used')).toBe(true)
    expect(hasUsageGauge('Top-tier   [||||||||||]  100% used')).toBe(true)
    expect(hasUsageGauge('plain system text')).toBe(false)
    expect(hasUsageGauge('- [ ] todo item')).toBe(false)
    expect(hasUsageGauge('[]')).toBe(false)
    expect(hasUsageGauge('a [|] pipe or [   ] gap')).toBe(false)
    expect(hasUsageGauge('[|||||||||||]')).toBe(false)
    expect(hasUsageGauge('[ |        ]')).toBe(false)
  })

  it('colors only the bar, leaves the rest of the /usage line alone', () => {
    expect(splitUsageGaugeLine('Today      [||        ]  24% used  ·  resets in 23h')).toEqual([
      { style: null, text: 'Today      ' },
      { style: null, text: '[' },
      { style: 'green', text: '||' },
      { style: null, text: '        ' },
      { style: null, text: ']' },
      { style: null, text: '  24% used  ·  resets in 23h' }
    ])
  })
})
