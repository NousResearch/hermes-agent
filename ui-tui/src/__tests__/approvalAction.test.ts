import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { DARK_THEME } from '../theme.js'
import {
  ApprovalPanel,
  approvalAction,
  approvalFullReviewMessage,
  approvalOptions,
  approvalOverflowMessage,
  approvalPayloadPage
} from '../components/prompts.js'

const theme = DARK_THEME

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

describe('approvalAction — pure key dispatch for ApprovalPrompt', () => {
  it('maps Esc to deny — parity with global Ctrl+C cancellation', () => {
    expect(approvalAction('', { escape: true }, 0)).toEqual({ kind: 'choose', choice: 'deny' })
    expect(approvalAction('', { escape: true }, 2)).toEqual({ kind: 'choose', choice: 'deny' })
  })

  it('maps number keys 1..4 to once/session/always/deny in registration order', () => {
    expect(approvalAction('1', {}, 0)).toEqual({ kind: 'choose', choice: 'once' })
    expect(approvalAction('2', {}, 0)).toEqual({ kind: 'choose', choice: 'session' })
    expect(approvalAction('3', {}, 0)).toEqual({ kind: 'choose', choice: 'always' })
    expect(approvalAction('4', {}, 0)).toEqual({ kind: 'choose', choice: 'deny' })
  })

  it('ignores out-of-range numbers', () => {
    expect(approvalAction('0', {}, 1)).toEqual({ kind: 'noop' })
    expect(approvalAction('5', {}, 1)).toEqual({ kind: 'noop' })
    expect(approvalAction('9', {}, 1)).toEqual({ kind: 'noop' })
  })

  it('confirms the current selection on Enter', () => {
    expect(approvalAction('', { return: true }, 0)).toEqual({ kind: 'choose', choice: 'once' })
    expect(approvalAction('', { return: true }, 3)).toEqual({ kind: 'choose', choice: 'deny' })
  })

  it('moves selection up/down within bounds', () => {
    expect(approvalAction('', { upArrow: true }, 2)).toEqual({ kind: 'move', delta: -1 })
    expect(approvalAction('', { downArrow: true }, 1)).toEqual({ kind: 'move', delta: 1 })
  })

  it('clamps selection movement at the edges', () => {
    expect(approvalAction('', { upArrow: true }, 0)).toEqual({ kind: 'noop' })
    expect(approvalAction('', { downArrow: true }, 3)).toEqual({ kind: 'noop' })
  })

  it('offers a full-payload review row only when the command preview overflows', () => {
    expect(approvalAction('v', {}, 0)).toEqual({ kind: 'noop' })
    expect(approvalAction('5', {}, 0)).toEqual({ kind: 'noop' })
    expect(approvalAction('', { downArrow: true }, 3)).toEqual({ kind: 'noop' })

    expect(approvalAction('v', {}, 0, undefined, true)).toEqual({ kind: 'toggleFull' })
    expect(approvalAction('V', {}, 0, undefined, true)).toEqual({ kind: 'toggleFull' })
    expect(approvalAction('5', {}, 0, undefined, true)).toEqual({ kind: 'toggleFull' })
    expect(approvalAction('', { return: true }, 4, undefined, true)).toEqual({ kind: 'toggleFull' })
    expect(approvalAction('', { downArrow: true }, 3, undefined, true)).toEqual({ kind: 'move', delta: 1 })
    expect(approvalAction('', { downArrow: true }, 4, undefined, true)).toEqual({ kind: 'noop' })
  })

  it('uses approval-local overflow copy instead of the old scrollback fallback', () => {
    expect(approvalOverflowMessage(1)).toBe('… +1 more line hidden - select View full payload to review here')
    expect(approvalOverflowMessage(3)).toBe('… +3 more lines hidden - select View full payload to review here')
    expect(approvalOverflowMessage(3)).not.toContain('full text above')
    expect(approvalFullReviewMessage(0, 10, 23)).toBe('Reviewing full payload: lines 1-10 of 23 (j/k scroll)')
  })

  it('pages the full payload with reversible j/k page indexes', () => {
    const lines = Array.from({ length: 23 }, (_, index) => `line ${index + 1}`)

    expect(approvalPayloadPage(lines, 0)).toEqual({ end: 10, index: 0, lines: lines.slice(0, 10), start: 0 })
    expect(approvalPayloadPage(lines, 1)).toEqual({ end: 20, index: 1, lines: lines.slice(10, 20), start: 10 })
    expect(approvalPayloadPage(lines, 2)).toEqual({ end: 23, index: 2, lines: lines.slice(20), start: 20 })
    expect(approvalPayloadPage(lines, 3)).toEqual({ end: 23, index: 2, lines: lines.slice(20), start: 20 })

    const lastPage = approvalPayloadPage(lines, 2)
    const previousPage = approvalPayloadPage(lines, lastPage.index - 1)

    expect(previousPage.start).toBe(10)
    expect(approvalPayloadPage(lines, previousPage.index + 1)).toEqual(lastPage)

    expect(approvalAction('j', {}, 0, undefined, true, true)).toEqual({ kind: 'pagePayload', delta: 1 })
    expect(approvalAction('k', {}, 0, undefined, true, true)).toEqual({ kind: 'pagePayload', delta: -1 })
    expect(approvalAction('', { pageDown: true }, 0, undefined, true, true)).toEqual({ kind: 'noop' })
    expect(approvalAction('', { pageUp: true }, 0, undefined, true, true)).toEqual({ kind: 'noop' })
  })

  it('renders a bounded full-payload review with approval controls in a compact viewport', async () => {
    const rawLines = Array.from({ length: 40 }, (_, index) => `payload line ${index + 1}`)
    const stdout = new PassThrough()
    const stdin = new PassThrough()
    const stderr = new PassThrough()
    let output = ''

    Object.assign(stdout, { columns: 80, isTTY: false, rows: 21 })
    Object.assign(stdin, { isTTY: false })
    Object.assign(stderr, { isTTY: false })
    stdout.on('data', chunk => {
      output += chunk.toString()
    })

    const instance = renderSync(
      React.createElement(ApprovalPanel, {
        description: 'review this command',
        opts: approvalOptions({ allowPermanent: true, command: 'ignored', description: 'review this command' }),
        rawLines,
        reviewPageIndex: 3,
        sel: 0,
        showFull: true,
        t: theme
      }),
      {
        patchConsole: false,
        stderr: stderr as NodeJS.WriteStream,
        stdin: stdin as NodeJS.ReadStream,
        stdout: stdout as NodeJS.WriteStream
      }
    )

    try {
      await delay(20)

      // Strip ANSI so assertions operate on the 21-row terminal frame.
      // eslint-disable-next-line no-control-regex
      const frame = output.replace(/\u001b\[[0-9;]*m/g, '')

      expect(frame).toContain('payload line 31')
      expect(frame).toContain('payload line 40')
      expect(frame).not.toContain('payload line 30')
      expect(frame).toContain('1. Allow once')
      expect(frame).toContain('4. Deny')
      expect(frame).toContain('5. Return to preview')
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })

  it('Esc beats numeric/return — denying is always the first interpretation', () => {
    // If a terminal somehow delivers Esc + a digit in the same event, deny
    // wins.  Documents the precedence so a future refactor doesn't flip it.
    expect(approvalAction('1', { escape: true }, 0, undefined, true)).toEqual({ kind: 'choose', choice: 'deny' })
    expect(approvalAction('v', { escape: true }, 0, undefined, true)).toEqual({ kind: 'choose', choice: 'deny' })
    expect(approvalAction('', { escape: true, return: true }, 1, undefined, true)).toEqual({
      kind: 'choose',
      choice: 'deny'
    })
  })

  it('returns noop for unrelated keystrokes (printable letters etc.)', () => {
    expect(approvalAction('a', {}, 0)).toEqual({ kind: 'noop' })
    expect(approvalAction(' ', {}, 0)).toEqual({ kind: 'noop' })
  })

  it('respects a reduced option set when permanent allow is disabled', () => {
    // tirith content-security warning present → no "always"; the 3-item set is
    // once/session/deny, so 3 maps to deny and 4 is out of range.
    const opts = ['once', 'session', 'deny'] as const

    expect(approvalAction('3', {}, 0, opts)).toEqual({ kind: 'choose', choice: 'deny' })
    expect(approvalAction('4', {}, 0, opts)).toEqual({ kind: 'noop' })
    expect(approvalAction('', { downArrow: true }, 2, opts)).toEqual({ kind: 'noop' })
    expect(approvalAction('', { return: true }, 2, opts)).toEqual({ kind: 'choose', choice: 'deny' })

    expect(approvalAction('4', {}, 0, opts, true)).toEqual({ kind: 'toggleFull' })
    expect(approvalAction('', { return: true }, 3, opts, true)).toEqual({ kind: 'toggleFull' })
  })

  it('offers only once and deny for Smart DENY owner override', () => {
    const opts = approvalOptions({
      allowPermanent: true,
      command: 'rm -rf /',
      description: 'blocked',
      smartDenied: true
    })

    expect(opts).toEqual(['once', 'deny'])
    expect(approvalAction('2', {}, 0, opts)).toEqual({ kind: 'choose', choice: 'deny' })
    expect(approvalAction('3', {}, 0, opts)).toEqual({ kind: 'noop' })
  })

  it('uses explicit gateway choices as the prompt contract', () => {
    expect(
      approvalOptions({
        allowPermanent: true,
        choices: ['once', 'deny'],
        command: 'rm -rf /',
        description: 'blocked'
      })
    ).toEqual(['once', 'deny'])
  })
})
