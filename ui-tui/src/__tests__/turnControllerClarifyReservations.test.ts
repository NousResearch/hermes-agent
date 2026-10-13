import { beforeEach, describe, expect, it } from 'vitest'

import { turnController } from '../app/turnController.js'
import { toolTrailLabel } from '../lib/text.js'

describe('clarify trail reservations', () => {
  beforeEach(() => {
    turnController.fullReset()
  })

  it('releases only the request that owns a shared clarify label', () => {
    const label = toolTrailLabel('clarify')

    turnController.reservePersistedToolLabel(label, 'request-old')
    turnController.reservePersistedToolLabel(label, 'request-new')
    turnController.releasePersistedToolLabel(label, 'request-old')

    expect(turnController.persistedToolLabels.has(label)).toBe(true)

    turnController.releasePersistedToolLabel(label, 'request-new')

    expect(turnController.persistedToolLabels.has(label)).toBe(false)
  })

  it('does not let a late release consume a new turn reservation', () => {
    const label = toolTrailLabel('clarify')

    turnController.reservePersistedToolLabel(label, 'request-old')
    turnController.clearPersistedToolLabels()
    turnController.reservePersistedToolLabel(label, 'request-new')
    turnController.releasePersistedToolLabel(label, 'request-old')

    expect(turnController.persistedToolLabels.has(label)).toBe(true)
  })

  it('does not let a late completion consume a new turn reservation', () => {
    const label = toolTrailLabel('clarify')

    turnController.recordToolStart('tool-old', 'clarify', '')
    turnController.startMessage()
    turnController.reservePersistedToolLabel(label, 'request-new')
    turnController.recordToolComplete('tool-old', 'clarify')

    expect(turnController.persistedToolLabels.has(label)).toBe(true)
  })
})
