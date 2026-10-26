import { describe, expect, it } from 'vitest'

import { isLineKillModifier } from '../components/textInput.js'

// Cmd+Backspace should kill to the line boundary, but the modifier it is
// distinguished by matters a great deal. `isActionMod` is the wrong test:
//   - on macOS it accepts `key.meta`, and hermes-ink reports Option as
//     `meta` — so Option+Backspace (delete-word, the macOS standard) would
//     silently become "delete the whole line".
//   - on Linux/Windows it is `key.ctrl`, and Ctrl+Backspace is delete-word
//     there in readline, VS Code, browsers, and Windows Terminal.
// Only the kitty CSI-u / modifyOtherKeys `super` bit means Cmd.

const key = (over: Partial<{ ctrl: boolean; meta: boolean; super: boolean }> = {}) => ({
  ctrl: false,
  meta: false,
  super: false,
  ...over
})

describe('isLineKillModifier', () => {
  it('accepts the super bit (Cmd via kitty CSI-u / modifyOtherKeys)', () => {
    expect(isLineKillModifier(key({ super: true }))).toBe(true)
  })

  it('accepts super even when the terminal also sets a benign ctrl bit', () => {
    // VS Code/Cursor forward Cmd chords as CSI-u with super + ctrl set.
    expect(isLineKillModifier(key({ ctrl: true, super: true }))).toBe(true)
  })

  it('rejects meta so Option+Backspace stays delete-word on macOS', () => {
    expect(isLineKillModifier(key({ meta: true }))).toBe(false)
  })

  it('rejects ctrl so Ctrl+Backspace stays delete-word on Linux/Windows', () => {
    expect(isLineKillModifier(key({ ctrl: true }))).toBe(false)
  })

  it('rejects an unmodified keypress', () => {
    expect(isLineKillModifier(key())).toBe(false)
  })

  it('treats a missing super field as absent rather than truthy', () => {
    expect(isLineKillModifier({ ctrl: false, meta: false })).toBe(false)
  })
})
