import { stringWidth } from '@hermes/ink'
import { describe, expect, it } from 'vitest'

import {
  BUSY_STATUS_PAD_LEN,
  BUSY_STATUS_TEXT,
  busyIndicatorWidth,
  busyStatusShimmerSegments,
  MAX_DURATION_WIDTH,
  padBusyStatusText
} from '../components/appChrome.js'
import { VERBS } from '../content/verbs.js'

describe('FaceTicker busy status text', () => {
  it('uses a static working label padded to the legacy rotating-verb slot', () => {
    const padded = padBusyStatusText()
    const legacyRotatingVerbWidth = VERBS.reduce((max, verb) => Math.max(max, verb.length), 0) + 1

    expect(BUSY_STATUS_TEXT).toBe('working')
    expect(padded).toHaveLength(BUSY_STATUS_PAD_LEN)
    expect(padded.trimEnd()).toBe('working')
    expect(padded).not.toContain('…')
    expect(BUSY_STATUS_PAD_LEN).toBeGreaterThanOrEqual(legacyRotatingVerbWidth)
  })

  it('moves a shimmer window across the stable working letters', () => {
    const frameAtFirstLetter = busyStatusShimmerSegments(2)
    const frameAtSecondLetter = busyStatusShimmerSegments(3)

    expect(frameAtFirstLetter.map(segment => segment.char).join('')).toBe('working')
    expect(frameAtSecondLetter.map(segment => segment.char).join('')).toBe('working')
    expect(frameAtFirstLetter.map(segment => segment.tone)).not.toEqual(frameAtSecondLetter.map(segment => segment.tone))
    expect(frameAtFirstLetter.findIndex(segment => segment.tone === 'peak')).toBe(0)
    expect(frameAtSecondLetter.findIndex(segment => segment.tone === 'peak')).toBe(1)
    expect(frameAtFirstLetter[1]?.tone).toBe('soft')
  })

  it('preserves text-hidden unicode style and spinner/duration width reservations', () => {
    expect(busyIndicatorWidth('unicode', false)).toBe(1)
    expect(busyIndicatorWidth('unicode', true)).toBe(1 + stringWidth(' · ') + MAX_DURATION_WIDTH)
    expect(busyIndicatorWidth('ascii', false)).toBe(1 + 1 + BUSY_STATUS_PAD_LEN)
    expect(busyIndicatorWidth('ascii', true)).toBe(1 + 1 + BUSY_STATUS_PAD_LEN + stringWidth(' · ') + MAX_DURATION_WIDTH)
  })
})
