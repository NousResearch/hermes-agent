import { describe, expect, it } from 'vitest'

import { shouldPassThroughToGlobalHandler, shouldPreserveCtrlJNewline } from '../components/textInput.js'
import { DEFAULT_VOICE_RECORD_KEY, isMac, parseVoiceRecordKey } from '../lib/platform.js'

const key = (overrides: Record<string, unknown> = {}) => ({ ctrl: false, meta: false, ...overrides }) as any

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

  it('passes through the Ctrl+L redraw key so the composer never inserts a stray "l"', () => {
    // Raw Ctrl+L — the byte every terminal (macOS included) sends for the
    // dashboard redraw — must pass through on ALL platforms, so pin the
    // assertion to the raw-ctrl shape rather than isAction, whose action
    // modifier means Cmd on darwin.
    expect(shouldPassThroughToGlobalHandler('l', key({ ctrl: true }))).toBe(true)
    // A plain (unmodified) "l" must still reach the composer as typing.
    expect(shouldPassThroughToGlobalHandler('l', key())).toBe(false)
  })

  it('passes through Cmd+L on macOS-shaped events without touching plain Alt+L', () => {
    // Cmd surfaces as meta (legacy terminals) or super (kitty protocol);
    // isAction passes it through only where the action modifier is Cmd
    // (darwin). Elsewhere meta means plain Alt, and Alt+L must stay
    // composer input. Keyed to isMac — the same constant the
    // implementation keys on — so the test is platform-independent.
    for (const mod of [{ meta: true }, { super: true }]) {
      expect(shouldPassThroughToGlobalHandler('l', key(mod))).toBe(isMac)
    }
  })
})
