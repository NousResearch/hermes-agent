import type { Key } from '@hermes/ink'
import { describe, expect, it } from 'vitest'

import { InputEvent } from '../../packages/hermes-ink/src/ink/events/input-event.js'
import { INITIAL_STATE, parseMultipleKeypresses } from '../../packages/hermes-ink/src/ink/parse-keypress.js'
import { approvalAction } from '../components/prompts.js'
import { shouldPassThroughToGlobalHandler } from '../components/textInput.js'
import { isListDown, isListUp } from '../lib/listNavigation.js'

describe('Ctrl+N/P list navigation contract', () => {
  it.each([
    ['\x0e', 1], ['\x10', -1], ['\x1b[110;5u', 1], ['\x1b[112;5u', -1]
  ] as const)('decodes terminal sequence %j into navigation', (sequence, direction) => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, sequence)
    expect(keys).toHaveLength(1)
    const parsed = keys[0]!

    if (parsed.kind !== 'key') {
      throw new Error('Expected a keypress')
    }

    const event = new InputEvent(parsed)
    expect(event.key.ctrl).toBe(true)
    expect(isListDown(event.input, event.key)).toBe(direction === 1)
    expect(isListUp(event.input, event.key)).toBe(direction === -1)
    expect(approvalAction(event.input, event.key, 1)).toEqual({ kind: 'move', delta: direction })
  })

  it.each(['n', '\x0e'])('moves down with Ctrl+%s without choosing', ch => {
    expect(approvalAction(ch, { ctrl: true }, 1)).toEqual({ kind: 'move', delta: 1 })
  })
  it.each(['p', '\x10'])('moves up with Ctrl+%s without choosing', ch => {
    expect(approvalAction(ch, { ctrl: true }, 1)).toEqual({ kind: 'move', delta: -1 })
  })
  it.each(['n', 'p', 'j', 'k'])('leaves ordinary %s alone', ch => {
    expect(approvalAction(ch, {}, 1)).toEqual({ kind: 'noop' })
  })
  it.each([
    { ctrl: true, shift: true },
    { ctrl: true, meta: true },
    { ctrl: true, super: true }
  ])('ignores modified chords %j', key => {
    expect(approvalAction('n', key, 1)).toEqual({ kind: 'noop' })
    expect(approvalAction('p', key, 1)).toEqual({ kind: 'noop' })
  })
  it.each(['\x1b[110;13u', '\x1b[112;13u'])('rejects decoded Ctrl+Super CSI-u sequence %j', sequence => {
    const [keys] = parseMultipleKeypresses(INITIAL_STATE, sequence)
    const parsed = keys[0]!

    if (parsed.kind !== 'key') {
      throw new Error('Expected a keypress')
    }

    const event = new InputEvent(parsed)
    expect(event.key.ctrl && event.key.super).toBe(true)
    expect(isListDown(event.input, event.key)).toBe(false)
    expect(isListUp(event.input, event.key)).toBe(false)
    expect(approvalAction(event.input, event.key, 1)).toEqual({ kind: 'noop' })
  })
  it('reserves shifted arrows for transcript scrolling', () => {
    expect(approvalAction('', { upArrow: true, shift: true }, 1)).toEqual({ kind: 'noop' })
    expect(approvalAction('', { downArrow: true, shift: true }, 1)).toEqual({ kind: 'noop' })
  })
  it('clamps both ends', () => {
    expect(approvalAction('p', { ctrl: true }, 0)).toEqual({ kind: 'noop' })
    expect(approvalAction('n', { ctrl: true }, 3)).toEqual({ kind: 'noop' })
  })
  it.each(['n', '\x0e', 'p', '\x10'])('passes Ctrl+%s through the editor', ch => {
    expect(shouldPassThroughToGlobalHandler(ch, { ctrl: true } as Key)).toBe(true)
  })
})
