import { describe, expect, it } from 'vitest'

import { toUsageMeterInfo } from '../app/useUsageMeterPoll.js'

describe('toUsageMeterInfo', () => {
  it('preserves an unavailable reason for the status command', () => {

    const info = toUsageMeterInfo({
      claude: { available: false, windows: [], unavailable_reason: 'Usage endpoint returned HTTP 429.' },
      codex: { available: false, windows: [] }
    } as never)

    expect(info?.claude).toMatchObject({ available: false, unavailableReason: 'Usage endpoint returned HTTP 429.' })
  })
  it('normalizes both providers independently and clamps percentages', () => {
    expect(
      toUsageMeterInfo({
        claude: {
          available: true,
          windows: [
            { label: 'Current session', used_percent: 22.6 },
            { label: 'Current week', used_percent: 140 }
          ]
        },
        codex: { available: false, windows: [] }
      })
    ).toEqual({
      claude: {
        available: true,
        windows: [
          { label: 'Current session', usedPercent: 23 },
          { label: 'Current week', usedPercent: 100 }
        ]
      },
      codex: { available: false, windows: [] }
    })
  })

  it('rejects non-boolean provider availability flags', () => {
    expect(
      toUsageMeterInfo({
        claude: {
          available: 'yes',
          windows: [{ label: 'Current session', used_percent: 12 }]
        },
        codex: { available: false, windows: [] }
      } as never)
    ).toEqual({
      claude: {
        available: false,
        windows: [{ label: 'Current session', usedPercent: 12 }]
      },
      codex: { available: false, windows: [] }
    })
  })
})
