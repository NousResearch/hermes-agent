import { describe, expect, it, beforeEach } from 'vitest'

import {
  getComposerDraftCursor,
  rememberComposerDraftCursor,
  requestComposerCursorRestore,
  resetComposerDraftCursorStateForTests,
  setComposerDraftCursorFrozen,
  setComposerDraftCursorSuppress,
  takeComposerCursorRestore
} from '../lib/composerDraftCursor.js'
import { isShiftArrowChord, shouldStopVisualRowNavPropagation } from '../components/textInput.js'

describe('composerDraftCursor', () => {
  beforeEach(() => {
    resetComposerDraftCursorStateForTests()
  })

  it('remembers edit cursor and ignores updates while suppressed (Up-climb)', () => {
    rememberComposerDraftCursor(12)
    expect(getComposerDraftCursor()).toBe(12)

    setComposerDraftCursorSuppress(true)
    rememberComposerDraftCursor(0)
    expect(getComposerDraftCursor()).toBe(12)

    setComposerDraftCursorSuppress(false)
    rememberComposerDraftCursor(4)
    expect(getComposerDraftCursor()).toBe(4)
  })

  it('freezes memory while browsing history so entry cursors cannot clobber it', () => {
    rememberComposerDraftCursor(10)
    setComposerDraftCursorFrozen(true)
    rememberComposerDraftCursor(3)
    expect(getComposerDraftCursor()).toBe(10)

    setComposerDraftCursorFrozen(false)
    rememberComposerDraftCursor(6)
    expect(getComposerDraftCursor()).toBe(6)
  })

  it('hands a pending restore to TextInput on the next external value change', () => {
    requestComposerCursorRestore(7)
    expect(takeComposerCursorRestore(99)).toBe(7)
    expect(takeComposerCursorRestore(99)).toBe(99)
  })
})

describe('isShiftArrowChord', () => {
  const up = { downArrow: false, shift: false, upArrow: true }
  const shiftUp = { downArrow: false, shift: true, upArrow: true }

  it('honors key.shift when the terminal reports it', () => {
    expect(isShiftArrowChord(shiftUp)).toBe(true)
    expect(shouldStopVisualRowNavPropagation(shiftUp)).toBe(false)
  })

  it('detects CSI shift-arrow sequences when key.shift is missing', () => {
    expect(isShiftArrowChord(up, { keypress: { raw: '\x1b[1;2A' } })).toBe(true)
    expect(isShiftArrowChord(up, { keypress: { raw: '\x1b[A' } })).toBe(false)
  })
})
