import { beforeEach, describe, expect, it } from 'vitest'

import { turnController } from '../app/turnController.js'
import { getTurnState, resetTurnState } from '../app/turnStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'

describe('turnController reasoning visibility (#22894)', () => {
  beforeEach(() => {
    resetUiState()
    resetTurnState()
    turnController.fullReset()
  })

  describe('display.show_reasoning = false', () => {
    beforeEach(() => {
      patchUiState({ showReasoning: false, streaming: true })
    })

    it('does not store streamed <think> content into turn state on flush', () => {
      turnController.startMessage()
      turnController.recordMessageDelta({ text: '<think>secret plan</think>visible reply' })
      turnController.flushStreamingSegment()

      expect(getTurnState().reasoning).toBe('')
      expect(getTurnState().reasoningTokens).toBe(0)
      expect(turnController.reasoningText).toBe('')
    })

    it('does not attach thinking when payload.text carries <think> tags', () => {
      turnController.startMessage()
      const result = turnController.recordMessageComplete({ text: '<think>secret plan</think>visible reply' })

      expect(result.finalText).toBe('visible reply')
      expect(result.finalMessages).toHaveLength(1)
      expect(result.finalMessages[0]).toMatchObject({ role: 'assistant', text: 'visible reply' })
      expect(result.finalMessages[0]?.thinking).toBeUndefined()
      expect(result.finalMessages[0]?.thinkingTokens).toBeUndefined()
    })

    it('drops payload.reasoning on message.complete', () => {
      turnController.startMessage()

      const result = turnController.recordMessageComplete({
        reasoning: 'gateway sent reasoning anyway',
        text: 'visible reply'
      })

      expect(result.finalMessages[0]).toMatchObject({ role: 'assistant', text: 'visible reply' })
      expect(result.finalMessages[0]?.thinking).toBeUndefined()
    })

    it('does not attach thinking when payload.rendered carries <think> tags', () => {
      turnController.startMessage()
      const result = turnController.recordMessageComplete({ rendered: '<think>secret plan</think>visible reply' })

      expect(result.finalText).toBe('visible reply')
      expect(result.finalMessages[0]).toMatchObject({ role: 'assistant', text: 'visible reply' })
      expect(result.finalMessages[0]?.thinking).toBeUndefined()
      expect(result.finalMessages[0]?.thinkingTokens).toBeUndefined()
    })

    it('drops reasoning captured before reasoning display is toggled off', () => {
      turnController.startMessage()
      patchUiState({ showReasoning: true })
      turnController.recordMessageDelta({ text: '<think>earlier visible reasoning</think>partial reply' })
      turnController.flushStreamingSegment()

      patchUiState({ showReasoning: false })
      const result = turnController.recordMessageComplete({ text: 'final visible reply' })

      expect(result.finalMessages.map(msg => msg.text)).toEqual(['partial reply', 'final visible reply'])
      expect(result.finalMessages.every(msg => msg.thinking === undefined)).toBe(true)
      expect(result.finalMessages.every(msg => msg.thinkingTokens === undefined)).toBe(true)
    })

    it('preserves visible response text untouched', () => {
      turnController.startMessage()
      turnController.recordMessageDelta({ text: '<think>x</think>plain answer' })
      turnController.flushStreamingSegment()

      expect(getTurnState().streamSegments[0]).toMatchObject({ role: 'assistant', text: 'plain answer' })
      expect(getTurnState().streamSegments[0]?.thinking).toBeUndefined()
    })
  })

  describe('display.show_reasoning = true', () => {
    beforeEach(() => {
      patchUiState({ showReasoning: true, streaming: true })
    })

    it('stores streamed <think> content into turn state on flush', () => {
      turnController.startMessage()
      turnController.recordMessageDelta({ text: '<think>secret plan</think>visible reply' })
      turnController.flushStreamingSegment()

      expect(getTurnState().reasoning).toBe('secret plan')
      expect(turnController.reasoningText).toBe('secret plan')
    })

    it('attaches thinking when payload.text carries <think> tags', () => {
      turnController.startMessage()
      const result = turnController.recordMessageComplete({ text: '<think>secret plan</think>visible reply' })

      expect(result.finalMessages).toHaveLength(1)
      expect(result.finalMessages[0]).toMatchObject({
        role: 'assistant',
        text: 'visible reply',
        thinking: 'secret plan'
      })
      expect(result.finalMessages[0]?.thinkingTokens).toBeGreaterThan(0)
    })

    it('preserves payload.reasoning on message.complete', () => {
      turnController.startMessage()

      const result = turnController.recordMessageComplete({
        reasoning: 'thought process',
        text: 'visible reply'
      })

      expect(result.finalMessages[0]).toMatchObject({
        role: 'assistant',
        text: 'visible reply',
        thinking: 'thought process'
      })
    })
  })
})
