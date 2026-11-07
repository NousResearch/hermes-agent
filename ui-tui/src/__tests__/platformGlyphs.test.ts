import { describe, expect, it } from 'vitest'

import { isLegacyWindowsConsole, promptGlyph } from '../lib/platform.js'

describe('legacy Windows console detection (#67151)', () => {
  it('is true on win32 without Windows Terminal markers', () => {
    expect(isLegacyWindowsConsole('win32', {})).toBe(true)
  })

  it('Windows Terminal markers disarm it', () => {
    expect(isLegacyWindowsConsole('win32', { WT_SESSION: 'guid' })).toBe(false)
    expect(isLegacyWindowsConsole('win32', { WT_PROFILE_ID: 'profile' })).toBe(false)
  })

  it('never fires off Windows', () => {
    expect(isLegacyWindowsConsole('darwin', {})).toBe(false)
    expect(isLegacyWindowsConsole('linux', { WT_SESSION: 'guid' })).toBe(false)
  })
})

describe('promptGlyph (#67151)', () => {
  it('falls back to ASCII under a legacy Windows console', () => {
    expect(promptGlyph('win32', {})).toBe('>')
  })

  it('keeps U+276F elsewhere, including Windows Terminal', () => {
    expect(promptGlyph('darwin', {})).toBe('❯')
    expect(promptGlyph('linux', {})).toBe('❯')
    expect(promptGlyph('win32', { WT_SESSION: 'guid' })).toBe('❯')
  })
})
