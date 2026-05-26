import { describe, expect, it } from 'vitest'

import { shouldPassThroughToGlobalHandler } from '../components/textInput.js'
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
