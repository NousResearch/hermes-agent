import { describe, expect, it } from 'vitest'

import {
  sessionPickerHintText,
  sessionPickerKeyAction,
  sessionPickerModeAfterKey,
  sessionSearchText,
  shouldAppendSessionFilterChar
} from '../components/sessionPicker.js'

const key = (overrides: Record<string, unknown> = {}) =>
  ({ ctrl: false, meta: false, ...overrides }) as any

describe('sessionSearchText', () => {
  it('indexes id, title, preview, source, and message count for picker filtering', () => {
    const text = sessionSearchText({
      id: '20260509_001122_abc123',
      message_count: 42,
      preview: 'Need to create a Hermes TUI session picker PR',
      source: 'telegram',
      started_at: 1778265600,
      title: 'Question mark session search'
    })

    expect(text).toContain('20260509_001122_abc123')
    expect(text).toContain('question mark session search')
    expect(text).toContain('hermes tui session picker')
    expect(text).toContain('telegram')
    expect(text).toContain('42')
  })
})

describe('session picker filter mode', () => {
  it('enters filter mode only from bare ? so browse shortcuts keep working', () => {
    expect(sessionPickerModeAfterKey('browse', '?', key())).toBe('filter')
    expect(sessionPickerModeAfterKey('browse', 'a', key())).toBe('browse')
    expect(sessionPickerModeAfterKey('browse', '?', key({ ctrl: true }))).toBe('browse')
    expect(sessionPickerModeAfterKey('filter', '', key({ escape: true }))).toBe('browse')
  })

  it('only appends printable search text while filter mode is active', () => {
    expect(shouldAppendSessionFilterChar('q', key(), 'browse')).toBe(false)
    expect(shouldAppendSessionFilterChar('d', key(), 'browse')).toBe(false)
    expect(shouldAppendSessionFilterChar('q', key(), 'filter')).toBe(true)
    expect(shouldAppendSessionFilterChar('d', key(), 'filter')).toBe(true)
    expect(shouldAppendSessionFilterChar('d', key({ ctrl: true }), 'filter')).toBe(false)
  })

  it('shows the ? filter shortcut in browse footer and type-to-filter in filter footer', () => {
    expect(sessionPickerHintText({ filterMode: false, hasQuery: false })).toContain('? filter')
    expect(sessionPickerHintText({ filterMode: false, hasQuery: false })).toContain('1-9 quick')
    expect(sessionPickerHintText({ filterMode: true, hasQuery: true })).toContain('type to filter')
    expect(sessionPickerHintText({ filterMode: true, hasQuery: true })).not.toContain('1-9 quick')
  })

  it('keeps /resume browse shortcuts out of filter text handling', () => {
    expect(sessionPickerKeyAction({ ch: '1', key: key(), mode: 'browse', query: '', visibleCount: 3 })).toEqual({
      index: 0,
      type: 'quick-select'
    })
    expect(sessionPickerKeyAction({ ch: 'q', key: key(), mode: 'browse', query: '', visibleCount: 3 })).toEqual({
      type: 'cancel'
    })
    expect(sessionPickerKeyAction({ ch: 'd', key: key({ ctrl: true }), mode: 'browse', query: '', visibleCount: 3 })).toEqual({
      type: 'delete'
    })
    expect(sessionPickerKeyAction({ ch: '?', key: key(), mode: 'browse', query: '', visibleCount: 3 })).toEqual({
      type: 'enter-filter'
    })
  })

  it('treats q, d, and digits as search text only after ? enters filter mode', () => {
    expect(sessionPickerKeyAction({ ch: 'q', key: key(), mode: 'filter', query: '', visibleCount: 3 })).toEqual({
      ch: 'q',
      type: 'append-filter'
    })
    expect(sessionPickerKeyAction({ ch: 'd', key: key(), mode: 'filter', query: '', visibleCount: 3 })).toEqual({
      ch: 'd',
      type: 'append-filter'
    })
    expect(sessionPickerKeyAction({ ch: '1', key: key(), mode: 'filter', query: '', visibleCount: 3 })).toEqual({
      ch: '1',
      type: 'append-filter'
    })
    expect(sessionPickerKeyAction({ ch: '', key: key({ escape: true }), mode: 'filter', query: 'abc', visibleCount: 3 })).toEqual({
      type: 'exit-filter'
    })
  })
})
