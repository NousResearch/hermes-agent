import { describe, expect, it } from 'vitest'

import { INITIAL_STATE, parseMultipleKeypresses } from '../../packages/hermes-ink/src/ink/parse-keypress.js'
import {
  isCtrlJNewline,
  shouldPassThroughToGlobalHandler,
  shouldPreserveCtrlJNewline
} from '../components/textInput.js'
import { DEFAULT_VOICE_RECORD_KEY, parseVoiceRecordKey } from '../lib/platform.js'

const key = (overrides: Record<string, unknown> = {}) => ({ ctrl: false, meta: false, ...overrides }) as any

describe('isCtrlJNewline', () => {
  it.each([
    ['Kitty CSI-u', '\x1b[106;5u'],
    ['modifyOtherKeys', '\x1b[27;5;106~']
  ])('recognizes Ctrl+J parsed from %s', (_label, sequence) => {
    const [[parsed]] = parseMultipleKeypresses(INITIAL_STATE, sequence)

    if (parsed?.kind !== 'key') {
      throw new Error('expected a parsed key event')
    }

    expect(parsed).toMatchObject({ ctrl: true, name: 'j' })
    expect(isCtrlJNewline(parsed.name ?? '', parsed)).toBe(true)
  })

  it('does not consume plain J or chords with additional modifiers', () => {
    expect(isCtrlJNewline('j', key())).toBe(false)
    expect(isCtrlJNewline('j', key({ ctrl: true, meta: true }))).toBe(false)
    expect(isCtrlJNewline('j', key({ ctrl: true, shift: true }))).toBe(false)
  })
})

describe('shouldPreserveCtrlJNewline', () => {
  it('preserves Ctrl+J as newline in Ghostty even when tmux masks TERM/TERM_PROGRAM', () => {
    expect(
      shouldPreserveCtrlJNewline({
        GHOSTTY_RESOURCES_DIR: '/usr/share/ghostty',
        TERM: 'tmux-256color',
        TERM_PROGRAM: 'tmux'
      })
    ).toBe(true)
  })

  it('preserves Ctrl+J as newline inside tmux, which forwards it as a bare LF', () => {
    expect(shouldPreserveCtrlJNewline({ TERM: 'tmux-256color', TMUX: '/tmp/tmux-1000/default,1234,0' })).toBe(true)
  })

  it('keeps bare local POSIX LF-compatible prompts submitting on Ctrl+J', () => {
    expect(shouldPreserveCtrlJNewline({ TERM: 'xterm-256color' })).toBe(false)
  })
})

describe('shouldPassThroughToGlobalHandler', () => {
  it('passes through the configured voice shortcut while composer is focused', () => {
    expect(shouldPassThroughToGlobalHandler('o', key({ ctrl: true }), parseVoiceRecordKey('ctrl+o'))).toBe(true)
    expect(shouldPassThroughToGlobalHandler('r', key({ meta: true }), parseVoiceRecordKey('alt+r'))).toBe(true)
    expect(shouldPassThroughToGlobalHandler(' ', key({ ctrl: true }), parseVoiceRecordKey('ctrl+space'))).toBe(true)
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
    expect(shouldPassThroughToGlobalHandler('o', key({ ctrl: true }))).toBe(true)
    expect(shouldPassThroughToGlobalHandler('', key({ escape: true }))).toBe(true)
    expect(shouldPassThroughToGlobalHandler('', key({ tab: true }))).toBe(true)
    expect(shouldPassThroughToGlobalHandler('', key({ pageUp: true }))).toBe(true)
    expect(shouldPassThroughToGlobalHandler('', key({ pageDown: true }))).toBe(true)
  })
})
