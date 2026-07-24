import { beforeEach, describe, expect, it } from 'vitest'

import { turnController } from '../app/turnController.js'
import { getTurnState, resetTurnState } from '../app/turnStore.js'
import { parseToolTrailResultLine } from '../lib/text.js'

describe('TurnController tool narratives', () => {
  beforeEach(() => {
    turnController.reset()
    resetTurnState()
  })

  it('preserves reason target and concise result through completion', () => {
    turnController.recordToolStart('read-1', 'read_file', 'src/auth.ts', undefined, 'Inspect the current auth flow')
    turnController.recordToolComplete('read-1', 'read_file', undefined, 'read_file: 8 lines', 0.94)

    expect(parseToolTrailResultLine(getTurnState().streamPendingTools[0]!)).toEqual({
      call: 'Read File · Inspect the current auth flow (0.9s)',
      detail: 'src/auth.ts → 8 lines',
      mark: '✓'
    })
  })
})
