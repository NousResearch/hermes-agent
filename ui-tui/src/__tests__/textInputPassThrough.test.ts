import { describe, expect, it } from 'vitest'

import { rebaseAsyncPasteResult, shouldPassThroughToGlobalHandler } from '../components/textInput.js'
import { DEFAULT_VOICE_RECORD_KEY, parseVoiceRecordKey } from '../lib/platform.js'

const key = (overrides: Record<string, unknown> = {}) =>
  ({ ctrl: false, meta: false, ...overrides }) as any

describe('shouldPassThroughToGlobalHandler', () => {
  it('passes through the configured voice shortcut while composer is focused', () => {
    expect(
      shouldPassThroughToGlobalHandler('o', key({ ctrl: true }), parseVoiceRecordKey('ctrl+o'))
    ).toBe(true)
    expect(
      shouldPassThroughToGlobalHandler('r', key({ meta: true }), parseVoiceRecordKey('alt+r'))
    ).toBe(true)
    expect(
      shouldPassThroughToGlobalHandler(' ', key({ ctrl: true }), parseVoiceRecordKey('ctrl+space'))
    ).toBe(true)
    expect(
      shouldPassThroughToGlobalHandler('', key({ ctrl: true, return: true }), parseVoiceRecordKey('ctrl+enter'))
    ).toBe(true)
  })

  it('keeps the legacy default pass-through when no custom key is provided', () => {
    expect(shouldPassThroughToGlobalHandler('b', key({ ctrl: true }), DEFAULT_VOICE_RECORD_KEY)).toBe(true)
    expect(shouldPassThroughToGlobalHandler('b', key({ ctrl: true }))).toBe(true)
  })

  it('does not swallow ordinary typing keys', () => {
    expect(shouldPassThroughToGlobalHandler('h', key(), parseVoiceRecordKey('ctrl+o'))).toBe(false)
    expect(shouldPassThroughToGlobalHandler('o', key(), parseVoiceRecordKey('ctrl+o'))).toBe(false)
  })

  it('always passes through non-voice global control keys', () => {
    expect(shouldPassThroughToGlobalHandler('c', key({ ctrl: true }))).toBe(true)
    expect(shouldPassThroughToGlobalHandler('x', key({ ctrl: true }))).toBe(true)
    expect(shouldPassThroughToGlobalHandler('', key({ escape: true }))).toBe(true)
    expect(shouldPassThroughToGlobalHandler('', key({ tab: true }))).toBe(true)
    expect(shouldPassThroughToGlobalHandler('', key({ pageUp: true }))).toBe(true)
    expect(shouldPassThroughToGlobalHandler('', key({ pageDown: true }))).toBe(true)
  })
})

describe('rebaseAsyncPasteResult', () => {
  it('keeps an async paste at its original position when typing continues', () => {
    expect(
      rebaseAsyncPasteResult({
        currentValue: 'typed after paste',
        result: { cursor: 5, value: 'PATH-' },
        startCursor: 0,
        startValue: ''
      })
    ).toEqual({
      cursor: 5,
      value: 'PATH-typed after paste'
    })
  })

  it('rebases transformed paste text before edits made at the same cursor', () => {
    expect(
      rebaseAsyncPasteResult({
        currentValue: 'prefix user suffix',
        result: { cursor: 18, value: 'prefix [[paste]] suffix' },
        startCursor: 'prefix '.length,
        startValue: 'prefix suffix'
      })
    ).toEqual({
      cursor: 'prefix [[paste]] '.length,
      value: 'prefix [[paste]] user suffix'
    })
  })

  it('drops stale async paste results when the original anchor changed', () => {
    expect(
      rebaseAsyncPasteResult({
        currentValue: 'changed suffix',
        result: { cursor: 18, value: 'prefix [[paste]] suffix' },
        startCursor: 'prefix '.length,
        startValue: 'prefix suffix'
      })
    ).toBeNull()
  })

  it('drops stale async paste results when the suffix mutated after the original cursor', () => {
    // Prefix is intact but the original suffix was edited in flight: the
    // captured anchor no longer exists, so the async paste must be dropped
    // rather than inserted at a stale position.
    expect(
      rebaseAsyncPasteResult({
        currentValue: 'prefix changed',
        result: { cursor: 18, value: 'prefix [[paste]] suffix' },
        startCursor: 'prefix '.length,
        startValue: 'prefix suffix'
      })
    ).toBeNull()
  })
})
