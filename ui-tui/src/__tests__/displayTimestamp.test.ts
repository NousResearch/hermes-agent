import { formatDisplayTimestamp } from '@hermes/shared/display-timestamp'
import { describe, expect, it } from 'vitest'

describe('formatDisplayTimestamp', () => {
  it('preserves the configured Python strftime date-time format', () => {
    const local = new Date(2026, 7, 8, 15, 4, 5)

    expect(
      formatDisplayTimestamp(local, {
        enabled: true,
        format: '%Y-%m-%d %H:%M:%S'
      })
    ).toBe('2026-08-08 15:04:05')
  })

  it('returns no label when timestamps are disabled', () => {
    const local = new Date(2026, 7, 8, 15, 4, 5)

    expect(formatDisplayTimestamp(local, { enabled: false, format: '%Y-%m-%d %H:%M:%S' })).toBe('')
  })

  it('preserves literal percent escapes', () => {
    const local = new Date(2026, 7, 8, 15, 4, 5)

    expect(formatDisplayTimestamp(local, { enabled: true, format: '%% %H:%M' })).toBe('% 15:04')
  })

  it('treats numeric message timestamps as Unix seconds', () => {
    const local = new Date(2026, 7, 8, 15, 4, 5)

    expect(formatDisplayTimestamp(local.getTime() / 1000, { enabled: true, format: '%Y-%m-%d %H:%M:%S' })).toBe(
      '2026-08-08 15:04:05'
    )
  })

  it('supports Python composite directives accepted by the CLI', () => {
    const local = new Date(2026, 7, 8, 15, 4, 5)

    expect(formatDisplayTimestamp(local, { enabled: true, format: '%F|%T|%D|%R|%r' })).toBe(
      '2026-08-08|15:04:05|08/08/26|15:04|03:04:05 PM'
    )
  })

  it('supports Python weekday, week, and locale directives', () => {
    const local = new Date(2026, 7, 8, 15, 4, 5)

    expect(
      formatDisplayTimestamp(local, {
        enabled: true,
        format: '%a|%A|%w|%u|%e|%j|%U|%W|%V|%G|%g|%h|%C|%c|%x|%X'
      })
    ).toBe('Sat|Saturday|6|6| 8|220|31|31|32|2026|26|Aug|20|Sat Aug  8 15:04:05 2026|08/08/26|15:04:05')
  })

  it('supports Unix epoch seconds and newline/tab escapes', () => {
    const local = new Date(2026, 7, 8, 15, 4, 5)
    const expectedEpoch = String(Math.floor(local.getTime() / 1000))

    expect(formatDisplayTimestamp(local, { enabled: true, format: '%s' })).toBe(expectedEpoch)
    expect(formatDisplayTimestamp(local, { enabled: true, format: '%n%t%H:%M' })).toBe('\n\t15:04')
  })

  it('matches Python week and ISO-year rollover semantics', () => {
    const cases = [
      [new Date(2025, 11, 29), '52|52|01|2026|26'],
      [new Date(2026, 0, 1), '00|00|01|2026|26'],
      [new Date(2026, 0, 5), '01|01|02|2026|26'],
      [new Date(2027, 0, 1), '00|00|53|2026|26']
    ] as const

    for (const [date, expected] of cases) {
      expect(formatDisplayTimestamp(date, { enabled: true, format: '%U|%W|%V|%G|%g' })).toBe(expected)
    }
  })
})
