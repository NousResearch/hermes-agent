import { beforeEach, describe, expect, it, vi } from 'vitest'

import { submitPrompt, type SubmitPromptDeps } from '../app/submissionCore.js'
import { turnController } from '../app/turnController.js'
import { getTurnState, patchTurnState, resetTurnState } from '../app/turnStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import type { GatewayClient } from '../gatewayClient.js'

// A busy-input row must land INSIDE the live turn, at the point it was sent
// (#136670): the gateway holds a mid-turn prompt.submit until the next tool
// result, so appending the row to committed history renders it above tool rows
// that already ran. These tests pin the placement decision, the send-order
// flush, the turn-end commit order, and the failed-turn salvage.
describe('busy input renders inside the live turn (#136670)', () => {
  beforeEach(() => {
    resetUiState()
    resetTurnState()
    turnController.fullReset()
    patchUiState({ sid: 'sess-1' })
  })

  function makeDeps(): SubmitPromptDeps {
    const gw = {
      request: vi.fn((method: string) => {
        if (method === 'input.detect_drop') {
          return Promise.resolve({ matched: false })
        }

        return Promise.resolve({ status: 'streaming' })
      })
    } as unknown as GatewayClient

    return {
      appendMessage: vi.fn(),
      enqueue: vi.fn(),
      expand: (t: string) => t,
      gw,
      setLastUserMsg: vi.fn(),
      sys: vi.fn()
    }
  }

  async function flushMicrotasks() {
    await Promise.resolve()
    await Promise.resolve()
    await Promise.resolve()
  }

  function seedLiveToolSegment() {
    const segment = { kind: 'trail' as const, role: 'system' as const, text: '', tools: ['Terminal("sleep 30")'] }

    turnController.segmentMessages = [segment]
    patchTurnState({ streamSegments: [segment] })
  }

  it('routes the user row into the turn segments when the live turn has content', async () => {
    seedLiveToolSegment()

    const deps = makeDeps()
    submitPrompt('also write BANANE', deps)
    await flushMicrotasks()

    expect(deps.appendMessage).not.toHaveBeenCalled()

    const segments = getTurnState().streamSegments
    const userRow = segments.find(msg => msg.role === 'user')

    expect(userRow?.text).toBe('also write BANANE')
    // After the tool row that already ran before the message was sent.
    expect(segments.indexOf(userRow!)).toBe(1)
  })

  it('keeps the plain history append when the live area is still empty', async () => {
    const deps = makeDeps()
    submitPrompt('first prompt', deps)
    await flushMicrotasks()

    expect(deps.appendMessage).toHaveBeenCalledWith({ role: 'user', text: 'first prompt' })
    expect(getTurnState().streamSegments).toEqual([])
  })

  it('routes the user row into the turn during the first thinking phase', async () => {
    // Before any tool or text: reasoning alone is live-area content, so a send
    // mid-thinking must land inside the turn, not in history above it.
    patchTurnState({ reasoningActive: true, reasoning: 'hmm…' })

    const deps = makeDeps()
    submitPrompt('also write BANANE', deps)
    await flushMicrotasks()

    expect(deps.appendMessage).not.toHaveBeenCalled()

    const userRow = getTurnState().streamSegments.find(msg => msg.role === 'user')

    expect(userRow?.text).toBe('also write BANANE')
  })

  it('flushes the streaming tail before the user row so the send order survives', () => {
    turnController.bufRef = 'mid-turn narration'
    turnController.pushUserSegment('steer')

    expect(turnController.segmentMessages.map(msg => [msg.role, msg.text])).toEqual([
      ['assistant', 'mid-turn narration'],
      ['user', 'steer']
    ])
    expect(turnController.bufRef).toBe('')
  })

  it('commits the user row between the segments around it on message.complete', () => {
    seedLiveToolSegment()
    turnController.pushUserSegment('also write BANANE')

    const before = getTurnState().streamSegments.length
    const laterTool = { kind: 'trail' as const, role: 'system' as const, text: '', tools: ['ProcessManage("wait")'] }

    turnController.segmentMessages = [...turnController.segmentMessages, laterTool]

    const { finalMessages } = turnController.recordMessageComplete({ text: 'FERTIG' })

    const label = (msg: (typeof finalMessages)[number]) => `${msg.role}:${msg.text || msg.tools?.join(',')}`

    const rows = finalMessages.map(label)

    expect(rows.indexOf('system:Terminal("sleep 30")')).toBeLessThan(rows.indexOf('user:also write BANANE'))
    expect(rows.indexOf('user:also write BANANE')).toBeLessThan(rows.indexOf('system:ProcessManage("wait")'))
    expect(rows.indexOf('user:also write BANANE')).toBeLessThan(rows.indexOf('assistant:FERTIG'))
    expect(before).toBe(2)
  })

  it('salvages the user row when a failed turn discards its segments', () => {
    seedLiveToolSegment()
    turnController.pushUserSegment('also write BANANE')

    const salvage = vi.fn()
    turnController.recordError(salvage)

    expect(salvage).toHaveBeenCalledTimes(1)
    expect(salvage).toHaveBeenCalledWith({ role: 'user', text: 'also write BANANE' })
    expect(turnController.segmentMessages).toEqual([])
    expect(getTurnState().streamSegments).toEqual([])
  })
})
