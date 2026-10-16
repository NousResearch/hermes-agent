import { describe, expect, it } from 'vitest'

import { switcherKeyIntent, type SwitcherKeyState } from '../domain/switcherKeys.js'

const state = (over: Partial<SwitcherKeyState> = {}): SwitcherKeyState => ({
  busy: false,
  canMoveDown: true,
  canMoveUp: true,
  confirmingDelete: false,
  draftHasText: false,
  hasQuery: false,
  kind: 'history',
  searching: false,
  ...over
})

describe('switcherKeyIntent', () => {
  it('ignores everything while a sub-picker or delete is in flight', () => {
    expect(switcherKeyIntent(state({ busy: true }), 'g', {})).toBe('none')
  })

  it('lets an armed delete prompt capture the next key: d confirms, anything else dismisses', () => {
    const armed = state({ confirmingDelete: true })

    expect(switcherKeyIntent(armed, 'd', {})).toBe('delete-confirm')
    expect(switcherKeyIntent(armed, 'x', {})).toBe('delete-dismiss')
    expect(switcherKeyIntent(armed, undefined, { escape: true })).toBe('delete-dismiss')
  })

  it('gives the search box every key, leaving only Esc and the arrows', () => {
    const searching = state({ searching: true })

    expect(switcherKeyIntent(searching, 'g', {})).toBe('none')
    expect(switcherKeyIntent(searching, 'd', {})).toBe('none')
    expect(switcherKeyIntent(searching, undefined, { escape: true })).toBe('search-cancel')
    expect(switcherKeyIntent(searching, undefined, { upArrow: true })).toBe('search-up')
    expect(switcherKeyIntent(searching, undefined, { downArrow: true })).toBe('search-down')
  })

  it('Esc clears an active filter before it cancels the switcher', () => {
    expect(switcherKeyIntent(state({ hasQuery: true }), undefined, { escape: true })).toBe('clear-query')
    expect(switcherKeyIntent(state(), undefined, { escape: true })).toBe('cancel')
  })

  it('maps Ctrl+N / Ctrl+R / Ctrl+D, accepting the raw control character too', () => {
    expect(switcherKeyIntent(state(), 'n', { ctrl: true })).toBe('new')
    expect(switcherKeyIntent(state(), '\u000e', { ctrl: true })).toBe('new')
    expect(switcherKeyIntent(state(), 'r', { ctrl: true })).toBe('refresh')
    expect(switcherKeyIntent(state({ kind: 'live' }), 'd', { ctrl: true })).toBe('close-live')
    expect(switcherKeyIntent(state({ kind: 'history' }), 'd', { ctrl: true })).toBe('none')
  })

  it('arms delete only on a history row, and only for plain d', () => {
    expect(switcherKeyIntent(state({ kind: 'history' }), 'd', {})).toBe('arm-delete')
    expect(switcherKeyIntent(state({ kind: 'live' }), 'd', {})).toBe('none')
    expect(switcherKeyIntent(state({ kind: 'header' }), 'd', {})).toBe('none')
  })

  it('opens search and toggles groups from any row except New, where letters type into the prompt', () => {
    for (const kind of ['header', 'history', 'live'] as const) {
      expect(switcherKeyIntent(state({ kind }), '/', {})).toBe('open-search')
      expect(switcherKeyIntent(state({ kind }), 'g', {})).toBe('toggle-group')
    }

    expect(switcherKeyIntent(state({ kind: 'new' }), '/', {})).toBe('none')
    expect(switcherKeyIntent(state({ kind: 'new' }), 'g', {})).toBe('none')
    expect(switcherKeyIntent(state({ kind: 'history' }), 'g', { ctrl: true })).toBe('none')
  })

  it('Tab picks a draft model only on the New row', () => {
    expect(switcherKeyIntent(state({ kind: 'new' }), undefined, { tab: true })).toBe('pick-model')
    expect(switcherKeyIntent(state({ kind: 'history' }), undefined, { tab: true })).toBe('none')
  })

  it('leaves arrows and Enter to the prompt while a draft has text, and respects list bounds', () => {
    const drafting = state({ draftHasText: true, kind: 'new' })

    expect(switcherKeyIntent(drafting, undefined, { downArrow: true })).toBe('none')
    expect(switcherKeyIntent(drafting, undefined, { return: true })).toBe('none')
    expect(switcherKeyIntent(state({ canMoveUp: false }), undefined, { upArrow: true })).toBe('none')
    expect(switcherKeyIntent(state({ canMoveDown: false }), undefined, { downArrow: true })).toBe('none')
    expect(switcherKeyIntent(state(), undefined, { downArrow: true })).toBe('down')
    expect(switcherKeyIntent(state(), undefined, { return: true })).toBe('activate')
  })
})
