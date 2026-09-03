import { describe, expect, it } from 'vitest'

import { formatResetIn, nextPollDelay, selectQuotaWindows, toAccountUsageInfo } from '../app/useAccountUsagePoll.js'
import { normalizeQuotaDisplay } from '../app/useConfigSync.js'

const NOW = Date.UTC(2025, 0, 1, 12, 0, 0)
const inMinutes = (m: number) => new Date(NOW + m * 60_000).toISOString()

describe('formatResetIn', () => {
  it('renders minutes, hours and days at decreasing resolution', () => {
    expect(formatResetIn(inMinutes(12), NOW)).toBe('12m')
    expect(formatResetIn(inMinutes(165), NOW)).toBe('2h 45m')
    expect(formatResetIn(inMinutes(7380), NOW)).toBe('5d 3h')
  })

  it('clamps an instant already past to now instead of a negative countdown', () => {
    expect(formatResetIn(inMinutes(-90), NOW)).toBe('now')
  })

  it('returns an empty label for a missing or unparseable instant', () => {
    expect(formatResetIn(null, NOW)).toBe('')
    expect(formatResetIn(undefined, NOW)).toBe('')
    expect(formatResetIn('not-a-date', NOW)).toBe('')
  })
})

describe('toAccountUsageInfo', () => {
  it('normalizes a provider snapshot into integer percentages', () => {
    const info = toAccountUsageInfo({
      available: true,
      plan: 'plus',
      provider: 'openai-codex',
      windows: [{ label: 'Session', reset_at: inMinutes(165), used_percent: 57.4 }]
    })

    expect(info?.provider).toBe('openai-codex')
    expect(info?.plan).toBe('plus')
    expect(info?.windows).toHaveLength(1)
    expect(info?.windows[0]?.usedPercent).toBe(57)
    expect(info?.windows[0]?.remainingPercent).toBe(43)
  })

  it('drops windows the provider left unreported', () => {
    const info = toAccountUsageInfo({
      available: true,
      provider: 'openai-codex',
      windows: [
        { label: 'Session', used_percent: 10 },
        { label: 'Weekly', used_percent: null }
      ]
    })

    expect(info?.windows.map(w => w.label)).toEqual(['Session'])
  })

  it('resolves to null when nothing is displayable', () => {
    expect(toAccountUsageInfo(null)).toBeNull()
    expect(toAccountUsageInfo({ available: false })).toBeNull()
    expect(toAccountUsageInfo({ available: true, windows: [] })).toBeNull()
    expect(toAccountUsageInfo({ available: true, windows: [{ label: 'Weekly', used_percent: null }] })).toBeNull()
  })

  it('clamps an out-of-range percentage into 0-100', () => {
    const info = toAccountUsageInfo({
      available: true,
      windows: [{ label: 'Session', used_percent: 130 }]
    })

    expect(info?.windows[0]?.usedPercent).toBe(100)
    expect(info?.windows[0]?.remainingPercent).toBe(0)
  })
})

describe('nextPollDelay', () => {
  it('retries briefly while no snapshot has landed yet', () => {
    expect(nextPollDelay(false, 6)).toBe(5_000)
    expect(nextPollDelay(false, 1)).toBe(5_000)
  })

  it('settles onto the steady cadence once a snapshot lands', () => {
    expect(nextPollDelay(true, 6)).toBe(60_000)
  })

  it('settles onto the steady cadence when the warm-up budget runs out', () => {
    expect(nextPollDelay(false, 0)).toBe(60_000)
  })
})

describe('normalizeQuotaDisplay', () => {
  it('defaults to the session window — the cap that bites next', () => {
    expect(normalizeQuotaDisplay(undefined)).toBe('session')
    expect(normalizeQuotaDisplay(true)).toBe('session')
    expect(normalizeQuotaDisplay('session')).toBe('session')
    expect(normalizeQuotaDisplay('5h')).toBe('session')
  })

  it('accepts the off switch as a boolean or a name', () => {
    expect(normalizeQuotaDisplay(false)).toBe('off')
    expect(normalizeQuotaDisplay('off')).toBe('off')
    expect(normalizeQuotaDisplay('none')).toBe('off')
    expect(normalizeQuotaDisplay(' HIDDEN ')).toBe('off')
  })

  it('reads the other selections', () => {
    expect(normalizeQuotaDisplay('both')).toBe('both')
    expect(normalizeQuotaDisplay('all')).toBe('both')
    expect(normalizeQuotaDisplay('weekly')).toBe('weekly')
    expect(normalizeQuotaDisplay('week')).toBe('weekly')
    expect(normalizeQuotaDisplay('tightest')).toBe('tightest')
  })

  it('falls back to the default rather than hiding on an unknown value', () => {
    expect(normalizeQuotaDisplay('nonsense')).toBe('session')
    expect(normalizeQuotaDisplay(42)).toBe('session')
  })
})

describe('selectQuotaWindows', () => {
  const session = { label: 'Session', remainingPercent: 100, resetAt: null, resetIn: '2h 13m', usedPercent: 0 }
  const weekly = { label: 'Weekly', remainingPercent: 81, resetAt: null, resetIn: '5d 0h', usedPercent: 19 }
  const windows = [session, weekly]

  it('shows the session window by default, even when the weekly cap is tighter', () => {
    expect(selectQuotaWindows(windows, 'session').map(w => w.label)).toEqual(['Session'])
  })

  it('puts the weekly window last when both are asked for', () => {
    expect(selectQuotaWindows(windows, 'both').map(w => w.label)).toEqual(['Session', 'Weekly'])
  })

  it('keeps session first regardless of the order the provider reports', () => {
    expect(selectQuotaWindows([weekly, session], 'both').map(w => w.label)).toEqual(['Session', 'Weekly'])
  })

  it('shows only the weekly window when that is what was asked for', () => {
    expect(selectQuotaWindows(windows, 'weekly').map(w => w.label)).toEqual(['Weekly'])
  })

  it('still supports tracking whichever window is tightest', () => {
    expect(selectQuotaWindows(windows, 'tightest').map(w => w.label)).toEqual(['Weekly'])
  })

  it('falls back to the tightest when the provider labels windows differently', () => {
    const odd = [{ ...weekly, label: 'Monthly allowance' }]

    expect(selectQuotaWindows(odd, 'session').map(w => w.label)).toEqual(['Monthly allowance'])
  })

  it('selects nothing when the read-out is switched off or there is no data', () => {
    expect(selectQuotaWindows(windows, 'off')).toEqual([])
    expect(selectQuotaWindows([], 'session')).toEqual([])
  })
})
