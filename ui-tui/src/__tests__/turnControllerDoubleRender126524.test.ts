import { beforeEach, describe, expect, it } from 'vitest'

import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'

describe('issue 126524: interim-sealed lead-in must not render twice', () => {
  beforeEach(() => {
    resetUiState()
    resetTurnState()
    turnController.fullReset()
    patchUiState({ showReasoning: true })
  })

  it('strips interim-sealed text from the final when response_previewed is absent', () => {
    const LEAD = 'Lead-in line here.'
    const TAIL = 'Final answer body.'
    const FULL = `${LEAD}\n\n${TAIL}`

    turnController.recordMessageDelta({ text: LEAD })
    turnController.recordInterimMessage(LEAD)
    turnController.recordToolStart('t1', 'read_file', 'example.txt')
    turnController.recordToolComplete('t1', 'read_file', '', undefined, undefined, 'example content')
    turnController.recordMessageDelta({ text: TAIL })

    const { finalMessages } = turnController.recordMessageComplete({ text: FULL } as any)

    const assistantTexts = finalMessages
      .filter(m => m.role === 'assistant' && typeof m.text === 'string')
      .map(m => m.text as string)

    const joined = assistantTexts.join('\n\n')
    const occurrences = joined.split(LEAD).length - 1
    expect(occurrences).toBe(1)
    expect(joined).toContain(TAIL)
  })

  it('keeps a verbatim interim-sealed reply as its own bubble', () => {
    const SAME = 'Same reply, no extension.'

    turnController.recordMessageDelta({ text: SAME })
    turnController.recordInterimMessage(SAME)

    const { finalMessages } = turnController.recordMessageComplete({ text: SAME } as any)

    const assistantTexts = finalMessages
      .filter(m => m.role === 'assistant' && typeof m.text === 'string')
      .map(m => m.text as string)

    // A verbatim repeat is pinned as two messages (#65919): the sealed bubble
    // stays, and the final does not collapse into an empty duplicate.
    expect(assistantTexts.filter(t => t.trim() === SAME)).toHaveLength(2)
  })
})
