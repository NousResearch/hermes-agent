import { describe, expect, it } from 'vitest'

import { applyCompletion, completionToApplyOnSubmit } from '../domain/slash.js'

describe('applyCompletion', () => {
  it('replaces from compReplace and drops the leading slash from the row', () => {
    // The gateway's slash completer returns bare command names with
    // replace_from = 1 (after the leading "/").
    expect(applyCompletion('/ex', 'exit', 1)).toBe('/exit')
  })

  it('keeps the leading slash when the row carries one and input does not', () => {
    expect(applyCompletion('ex', '/exit', 0)).toBe('/exit')
  })

  it('replaces an argument token after a space (subcommand completion)', () => {
    expect(applyCompletion('/cron ad', 'add', 6)).toBe('/cron add')
  })
})

describe('completionToApplyOnSubmit', () => {
  it('accepts a completion that finishes a partial command name', () => {
    // "/ex" -> "/exit": a real token change, so Enter accepts it.
    expect(completionToApplyOnSubmit('/ex', 'exit', 1)).toBe('/exit')
  })

  it('does NOT swallow Enter when the completion only adds a trailing space', () => {
    // This is the bug: once "/exit" is fully typed, the gateway returns the
    // command with a trailing space ("exit ") so the classic-CLI dropdown
    // stays open. In the TUI that must NOT eat the Enter — the command is
    // already complete, so Enter should submit.
    expect(completionToApplyOnSubmit('/exit', 'exit ', 1)).toBeNull()
  })

  it('does not swallow Enter when applying the row is a no-op', () => {
    expect(completionToApplyOnSubmit('/exit', 'exit', 1)).toBeNull()
  })

  it('still accepts a real argument completion (no trailing-space false positive)', () => {
    expect(completionToApplyOnSubmit('/cron ad', 'add', 6)).toBe('/cron add')
  })

  it('submits (no accept) once an argument is fully typed and only a space is added', () => {
    expect(completionToApplyOnSubmit('/cron add', 'add ', 6)).toBeNull()
  })

  it('returns null when there is no row text', () => {
    expect(completionToApplyOnSubmit('/exit', undefined, 1)).toBeNull()
    expect(completionToApplyOnSubmit('/exit', '', 1)).toBeNull()
  })
})
