import { describe, expect, it } from 'vitest'

import { copyOnSelectEnabled, normalizeCopyOnSelect } from './copyOnSelect.js'

describe('normalizeCopyOnSelect', () => {
  it('defaults to auto when absent or unrecognized', () => {
    expect(normalizeCopyOnSelect({})).toBe('auto')
    expect(normalizeCopyOnSelect({ copy_on_select: null })).toBe('auto')
    expect(normalizeCopyOnSelect({ copy_on_select: 'maybe' })).toBe('auto')
  })

  it('normalizes booleans and truthy/falsey strings', () => {
    expect(normalizeCopyOnSelect({ copy_on_select: true })).toBe('on')
    expect(normalizeCopyOnSelect({ copy_on_select: false })).toBe('off')
    expect(normalizeCopyOnSelect({ copy_on_select: 'On' })).toBe('on')
    expect(normalizeCopyOnSelect({ copy_on_select: 'no' })).toBe('off')
  })
})

describe('copyOnSelectEnabled', () => {
  it('keeps the local-mac behavior', () => {
    expect(copyOnSelectEnabled({}, true)).toBe(true)
  })

  it('enables copy-on-select for hosted-over-SSH sessions', () => {
    expect(copyOnSelectEnabled({ SSH_CONNECTION: '192.168.1.2 51000 10.0.0.5 22' }, false)).toBe(true)
  })

  it('stays off for local non-mac sessions (existing behavior)', () => {
    expect(copyOnSelectEnabled({}, false)).toBe(false)
  })

  it('ignores an empty SSH_CONNECTION', () => {
    expect(copyOnSelectEnabled({ SSH_CONNECTION: '' }, false)).toBe(false)
  })

  it('honors an explicit setting regardless of platform', () => {
    expect(copyOnSelectEnabled({}, false, 'on')).toBe(true)
    expect(copyOnSelectEnabled({ SSH_CONNECTION: '10.0.0.1 1 10.0.0.2 22' }, true, 'off')).toBe(false)
  })

  it('the env override beats the config setting', () => {
    expect(copyOnSelectEnabled({ HERMES_TUI_COPY_ON_SELECT: 'off' }, true, 'on')).toBe(false)
  })

  it('honors HERMES_TUI_COPY_ON_SELECT=on on a local non-mac host', () => {
    expect(copyOnSelectEnabled({ HERMES_TUI_COPY_ON_SELECT: 'on' }, false)).toBe(true)
  })

  it('honors HERMES_TUI_COPY_ON_SELECT=off over SSH and on mac', () => {
    expect(
      copyOnSelectEnabled({ SSH_CONNECTION: '10.0.0.1 5000 10.0.0.2 22', HERMES_TUI_COPY_ON_SELECT: 'off' }, false)
    ).toBe(false)
    expect(copyOnSelectEnabled({ HERMES_TUI_COPY_ON_SELECT: 'false' }, true)).toBe(false)
  })

  it('ignores an unrecognized override value (falls back to the heuristic)', () => {
    expect(copyOnSelectEnabled({ HERMES_TUI_COPY_ON_SELECT: 'maybe' }, false)).toBe(false)
    expect(
      copyOnSelectEnabled({ HERMES_TUI_COPY_ON_SELECT: 'maybe', SSH_CONNECTION: '10.0.0.1 1 10.0.0.2 22' }, false)
    ).toBe(true)
  })
})
