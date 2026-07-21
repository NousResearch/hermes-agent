import { describe, expect, it } from 'vitest'

import { classifyAgentsListKey, rankSubagentRows } from '../components/agentsOverlay.js'
import { clampIndex } from '../components/overlayControls.js'
import type { SubagentNode } from '../types.js'

// Minimal node factory — rankSubagentRows only reads item.goal / id / model.
const node = (id: string, goal: string, model?: string): SubagentNode =>
  ({ aggregate: {}, children: [], item: { goal, id, model } }) as unknown as SubagentNode

describe('agents overlay fuzzy filtering', () => {
  const rows = [
    node('a1', 'review the auth module', 'opus'),
    node('a2', 'write tests for billing', 'sonnet'),
    node('a3', 'refactor the parser', 'haiku')
  ]

  it('keeps the full row list for an empty query', () => {
    expect(rankSubagentRows(rows, '')).toEqual(rows)
    expect(rankSubagentRows(rows, '   ')).toEqual(rows)
  })

  it('filters rows by goal text', () => {
    expect(rankSubagentRows(rows, 'auth').map(n => n.item.id)).toEqual(['a1'])
    expect(rankSubagentRows(rows, 'parser').map(n => n.item.id)).toEqual(['a3'])
  })

  it('matches on subagent id and model too', () => {
    expect(rankSubagentRows(rows, 'a2').map(n => n.item.id)).toContain('a2')
    expect(rankSubagentRows(rows, 'sonnet').map(n => n.item.id)).toContain('a2')
  })

  it('drops rows that do not match', () => {
    expect(rankSubagentRows(rows, 'zzzz')).toEqual([])
  })
})

describe('agents overlay list-mode key gating', () => {
  it('fires reserved single-key shortcuts only when no filter is active', () => {
    expect(classifyAgentsListKey('q', {}, false, true)).toEqual({ kind: 'close' })
    expect(classifyAgentsListKey('s', {}, false, true)).toEqual({ kind: 'cycleSort' })
    expect(classifyAgentsListKey('f', {}, false, true)).toEqual({ kind: 'cycleFilter' })
    expect(classifyAgentsListKey('x', {}, false, true)).toEqual({ kind: 'killOne' })
    expect(classifyAgentsListKey('X', {}, false, true)).toEqual({ kind: 'killSubtree' })
    expect(classifyAgentsListKey('p', {}, false, true)).toEqual({ kind: 'pause' })
    expect(classifyAgentsListKey('g', {}, false, true)).toEqual({ kind: 'cursorTop' })
    expect(classifyAgentsListKey('G', {}, false, true)).toEqual({ kind: 'cursorBottom' })
    expect(classifyAgentsListKey('[', {}, false, true)).toEqual({ kind: 'historyOlder' })
    expect(classifyAgentsListKey(']', {}, false, true)).toEqual({ kind: 'historyNewer' })
  })

  it('types the same reserved keys into the filter once it is active', () => {
    expect(classifyAgentsListKey('s', {}, true, true)).toEqual({ ch: 's', kind: 'filterAppend' })
    expect(classifyAgentsListKey('q', {}, true, true)).toEqual({ ch: 'q', kind: 'filterAppend' })
    expect(classifyAgentsListKey('x', {}, true, true)).toEqual({ ch: 'x', kind: 'filterAppend' })
  })

  it('starts a filter when a non-reserved printable key is pressed', () => {
    expect(classifyAgentsListKey('z', {}, false, true)).toEqual({ ch: 'z', kind: 'filterAppend' })
  })

  it('does not start a filter on a leading space, but allows spaces mid-query', () => {
    expect(classifyAgentsListKey(' ', {}, false, true)).toEqual({ kind: 'ignore' })
    expect(classifyAgentsListKey(' ', {}, true, true)).toEqual({ ch: ' ', kind: 'filterAppend' })
  })

  it('clamps the cursor into the filtered rows when the list shrinks (clampIndex)', () => {
    // The list-mode cursor clamp uses clampIndex(cursor, visibleRows.length).
    expect(clampIndex(7, 3)).toBe(2)
    expect(clampIndex(2, 3)).toBe(2)
    expect(clampIndex(1, 0)).toBe(0)
  })

  it('clears a non-empty filter on Esc before closing the overlay', () => {
    expect(classifyAgentsListKey('', { escape: true }, true, true)).toEqual({ kind: 'filterClear' })
    expect(classifyAgentsListKey('', { escape: true }, false, true)).toEqual({ kind: 'close' })
  })

  it('edits the filter with Backspace / Ctrl+U only while filtering', () => {
    expect(classifyAgentsListKey('', { backspace: true }, true, true)).toEqual({ kind: 'filterBackspace' })
    expect(classifyAgentsListKey('', { backspace: true }, false, true)).toEqual({ kind: 'ignore' })
    expect(classifyAgentsListKey('u', { ctrl: true }, true, true)).toEqual({ kind: 'filterClear' })
  })

  it('keeps arrows / wheel / Enter navigating regardless of filter state', () => {
    expect(classifyAgentsListKey('', { upArrow: true }, true, true)).toEqual({ kind: 'cursorUp' })
    expect(classifyAgentsListKey('', { downArrow: true }, true, true)).toEqual({ kind: 'cursorDown' })
    expect(classifyAgentsListKey('', { wheelUp: true }, true, true)).toEqual({ kind: 'cursorUp' })
    expect(classifyAgentsListKey('', { return: true }, true, true)).toEqual({ kind: 'open' })
  })

  it('routes vim-style j/k/l only when not filtering', () => {
    expect(classifyAgentsListKey('j', {}, false, true)).toEqual({ kind: 'cursorDown' })
    expect(classifyAgentsListKey('k', {}, false, true)).toEqual({ kind: 'cursorUp' })
    expect(classifyAgentsListKey('l', {}, false, true)).toEqual({ kind: 'open' })
    expect(classifyAgentsListKey('j', {}, true, true)).toEqual({ ch: 'j', kind: 'filterAppend' })
  })

  it('does not open on Enter when nothing is selected', () => {
    expect(classifyAgentsListKey('', { return: true }, false, false)).toEqual({ kind: 'ignore' })
  })

  it('ignores modified chords as filter input', () => {
    expect(classifyAgentsListKey('a', { meta: true }, true, true)).toEqual({ kind: 'ignore' })
  })
})
