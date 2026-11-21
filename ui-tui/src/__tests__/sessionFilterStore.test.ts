import type { Key } from '@hermes/ink'
import { afterEach, expect, it } from 'vitest'

import type { OverlayState } from '../app/interfaces.js'
import { $sessionFilterHoldsEsc, sessionsEscCloses } from '../app/sessionFilterStore.js'

const esc = { escape: true } as unknown as Key
const other = { escape: false } as unknown as Key
const open = { sessions: true } as OverlayState
const closed = { sessions: false } as OverlayState

afterEach(() => $sessionFilterHoldsEsc.set(false))

it('lets the global Esc close the switcher only while its filter does not hold Esc', () => {
  expect(sessionsEscCloses(esc, open)).toBe(true)
  expect(sessionsEscCloses(other, open)).toBe(false)
  expect(sessionsEscCloses(esc, closed)).toBe(false)

  $sessionFilterHoldsEsc.set(true)
  expect(sessionsEscCloses(esc, open)).toBe(false)
})
