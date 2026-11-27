import type { SessionActiveItem } from '@hermes/shared/gateway-types'
import { afterEach, describe, expect, it } from 'vitest'

import { applyLiveSessionRoster } from '../app/liveSessions.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'

const roster = (n: number) => Array.from({ length: n }, (_, i) => ({ id: `s${i}` }) as SessionActiveItem)

describe('applyLiveSessionRoster under display.tui_quiet_idle', () => {
  afterEach(() => {
    resetUiState()
  })

  it('updates the live count while idle by default', () => {
    patchUiState({ busy: false, liveSessionCount: 3 })
    applyLiveSessionRoster(roster(5))

    expect(getUiState().liveSessionCount).toBe(5)
  })

  it('holds the count while idle, then catches up once a turn runs', () => {
    patchUiState({ busy: false, liveSessionCount: 3, quietIdle: true })
    applyLiveSessionRoster(roster(5))

    expect(getUiState().liveSessionCount).toBe(3)

    patchUiState({ busy: true })
    applyLiveSessionRoster(roster(5))

    expect(getUiState().liveSessionCount).toBe(5)
  })

  it('always lands the initial hydrate', () => {
    patchUiState({ busy: false, liveSessionCount: 0, quietIdle: true })
    applyLiveSessionRoster(roster(4))

    expect(getUiState().liveSessionCount).toBe(4)
  })
})
