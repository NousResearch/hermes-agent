import { describe, expect, it } from 'vitest'

import { INITIAL_STATE, parseMultipleKeypresses } from './parse-keypress.js'
import { PASTE_END, PASTE_START } from './termio/csi.js'

describe('parseMultipleKeypresses bracketed paste recovery', () => {
  it('emits empty bracketed pastes when the terminal sends both markers', () => {
    const [keys, state] = parseMultipleKeypresses(INITIAL_STATE, PASTE_START + PASTE_END)

    expect(keys).toHaveLength(1)
    expect(keys[0]).toMatchObject({ isPasted: true, raw: '' })
    expect(state.mode).toBe('NORMAL')
  })

  it('flushes unterminated paste content back to normal input mode', () => {
    const [pendingKeys, pendingState] = parseMultipleKeypresses(INITIAL_STATE, PASTE_START + 'hello')

    expect(pendingKeys).toEqual([])
    expect(pendingState.mode).toBe('IN_PASTE')

    const [keys, state] = parseMultipleKeypresses(pendingState, null)

    expect(keys).toHaveLength(1)
    expect(keys[0]).toMatchObject({ isPasted: true, raw: 'hello' })
    expect(state.mode).toBe('NORMAL')
    expect(state.pasteBuffer).toBe('')
  })

  it('resets an empty unterminated paste start instead of staying stuck', () => {
    const [pendingKeys, pendingState] = parseMultipleKeypresses(INITIAL_STATE, PASTE_START)

    expect(pendingKeys).toEqual([])
    expect(pendingState.mode).toBe('IN_PASTE')

    const [keys, state] = parseMultipleKeypresses(pendingState, null)

    expect(keys).toEqual([])
    expect(state.mode).toBe('NORMAL')
    expect(state.pasteBuffer).toBe('')
  })
})

describe('parseMultipleKeypresses CSI u (Kitty keyboard protocol)', () => {
  it('parses Shift+Enter (CSI 13;2u) as return with shift=true', () => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, '\x1b[13;2u')

    expect(keys).toHaveLength(1)
    expect(keys[0]).toMatchObject({
      kind: 'key',
      name: 'return',
      shift: true,
      ctrl: false,
      meta: false,
      super: false
    })
  })

  it('parses Ctrl+Enter (CSI 13;5u) as return with ctrl=true', () => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, '\x1b[13;5u')

    expect(keys).toHaveLength(1)
    expect(keys[0]).toMatchObject({
      kind: 'key',
      name: 'return',
      shift: false,
      ctrl: true,
      meta: false,
      super: false
    })
  })

  it('parses Cmd+Enter (CSI 13;9u) as return with super=true', () => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, '\x1b[13;9u')

    expect(keys).toHaveLength(1)
    expect(keys[0]).toMatchObject({
      kind: 'key',
      name: 'return',
      shift: false,
      ctrl: false,
      meta: false,
      super: true
    })
  })

  it('parses plain Enter (CSI 13u) as return with no modifiers', () => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, '\x1b[13u')

    expect(keys).toHaveLength(1)
    expect(keys[0]).toMatchObject({
      kind: 'key',
      name: 'return',
      shift: false,
      ctrl: false,
      meta: false,
      super: false
    })
  })
})
