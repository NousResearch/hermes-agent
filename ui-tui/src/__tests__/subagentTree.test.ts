import { describe, expect, it } from 'vitest'

import {
  buildSubagentTree,
  descendantIds,
  flattenTree,
  fmtCost,
  fmtDuration,
  fmtTokens,
  formatSummary,
  hotnessBucket,
  peakHotness,
  sparkline,
  topLevelSubagents,
  treeTotals,
  widthByDepth
} from '../lib/subagentTree.js'
import type { SubagentProgress } from '../types.js'

const makeItem = (overrides: Partial<SubagentProgress> & Pick<SubagentProgress, 'id' | 'index'>): SubagentProgress => ({
  depth: 0,
  goal: overrides.id,
  notes: [],
  parentId: null,
  status: 'running',
  taskCount: 1,
  thinking: [],
  toolCount: 0,
  tools: [],
  ...overrides
})

describe('aggregate: tokens, cost, files, hotness', () => {
  it('sums tokens and cost across subtree', () => {
    const items = [
      makeItem({ costUsd: 0.01, id: 'p', index: 0, inputTokens: 1000, outputTokens: 500 }),
      makeItem({
        costUsd: 0.005,
        depth: 1,
        id: 'c1',
        index: 0,
        inputTokens: 500,
        outputTokens: 100,
        parentId: 'p'
      }),
      makeItem({
        costUsd: 0.008,
        depth: 1,
        id: 'c2',
        index: 1,
        inputTokens: 300,
        outputTokens: 200,
        parentId: 'p'
      })
    ]

    const tree = buildSubagentTree(items)
    expect(tree[0]!.aggregate).toMatchObject({
      costUsd: 0.023,
      inputTokens: 1800,
      outputTokens: 800
    })
  })

  it('counts files read + written across subtree', () => {
    const items = [
      makeItem({ filesRead: ['a.ts', 'b.ts'], id: 'p', index: 0 }),
      makeItem({ depth: 1, filesWritten: ['c.ts'], id: 'c', index: 0, parentId: 'p' })
    ]

    const tree = buildSubagentTree(items)
    expect(tree[0]!.aggregate.filesTouched).toBe(3)
  })

  it('hotness = totalTools / totalDuration', () => {
    const items = [
      makeItem({
        durationSeconds: 10,
        id: 'p',
        index: 0,
        status: 'completed',
        toolCount: 20
      })
    ]

    const tree = buildSubagentTree(items)
    expect(tree[0]!.aggregate.hotness).toBeCloseTo(2)
  })

  it('hotness is zero when duration is zero', () => {
    const items = [makeItem({ id: 'p', index: 0, toolCount: 10 })]
    const tree = buildSubagentTree(items)
    expect(tree[0]!.aggregate.hotness).toBe(0)
  })
})

describe('hotnessBucket + peakHotness', () => {
  it('peakHotness walks subtree', () => {
    const items = [
      makeItem({ durationSeconds: 100, id: 'p', index: 0, status: 'completed', toolCount: 1 }),
      makeItem({
        depth: 1,
        durationSeconds: 1,
        id: 'c',
        index: 0,
        parentId: 'p',
        status: 'completed',
        toolCount: 5
      })
    ]

    const tree = buildSubagentTree(items)
    expect(peakHotness(tree)).toBeGreaterThan(2)
  })

  it('hotnessBucket clamps and normalizes', () => {
    expect(hotnessBucket(0, 10, 4)).toBe(0)
    expect(hotnessBucket(10, 10, 4)).toBe(3)
    expect(hotnessBucket(5, 10, 4)).toBe(2)
    expect(hotnessBucket(100, 10, 4)).toBe(3) // clamped
    expect(hotnessBucket(5, 0, 4)).toBe(0) // guard against divide-by-zero
  })
})

describe('fmtCost + fmtTokens', () => {
  it('fmtCost handles ranges', () => {
    expect(fmtCost(0)).toBe('')
    expect(fmtCost(0.001)).toBe('<$0.01')
    expect(fmtCost(0.42)).toBe('$0.42')
    expect(fmtCost(1.23)).toBe('$1.23')
    expect(fmtCost(12.5)).toBe('$12.5')
  })

  it('fmtTokens handles ranges', () => {
    expect(fmtTokens(0)).toBe('0')
    expect(fmtTokens(542)).toBe('542')
    expect(fmtTokens(1234)).toBe('1.2k')
    expect(fmtTokens(45678)).toBe('46k')
  })
})

describe('formatSummary with tokens + cost', () => {
  it('includes token + cost when present', () => {
    expect(
      formatSummary({
        activeCount: 0,
        costUsd: 0.42,
        descendantCount: 3,
        filesTouched: 0,
        hotness: 0,
        inputTokens: 8000,
        maxDepthFromHere: 2,
        outputTokens: 2000,
        totalDuration: 30,
        totalTools: 14
      })
    ).toBe('d2 · 3 agents · 14 tools · 30s · 10k tok · $0.42')
  })
})

describe('buildSubagentTree', () => {
  it('returns empty list for empty input', () => {
    expect(buildSubagentTree([])).toEqual([])
  })

  it('treats flat list as top-level when no parentId is given', () => {
    const items = [makeItem({ id: 'a', index: 0 }), makeItem({ id: 'b', index: 1 }), makeItem({ id: 'c', index: 2 })]

    const tree = buildSubagentTree(items)
    expect(tree).toHaveLength(3)
    expect(tree.map(n => n.item.id)).toEqual(['a', 'b', 'c'])
    expect(tree.every(n => n.children.length === 0)).toBe(true)
  })

  it('nests children under their parent by subagent_id', () => {
    const items = [
      makeItem({ id: 'parent', index: 0 }),
      makeItem({ depth: 1, id: 'child-1', index: 0, parentId: 'parent' }),
      makeItem({ depth: 1, id: 'child-2', index: 1, parentId: 'parent' })
    ]

    const tree = buildSubagentTree(items)
    expect(tree).toHaveLength(1)
    expect(tree[0]!.children).toHaveLength(2)
    expect(tree[0]!.children.map(n => n.item.id)).toEqual(['child-1', 'child-2'])
  })

  it('builds multi-level nesting', () => {
    const items = [
      makeItem({ id: 'p', index: 0 }),
      makeItem({ depth: 1, id: 'c', index: 0, parentId: 'p' }),
      makeItem({ depth: 2, id: 'gc', index: 0, parentId: 'c' })
    ]

    const tree = buildSubagentTree(items)
    expect(tree[0]!.children[0]!.children[0]!.item.id).toBe('gc')
    expect(tree[0]!.aggregate.maxDepthFromHere).toBe(2)
    expect(tree[0]!.aggregate.descendantCount).toBe(2)
  })

  it('promotes orphaned children (missing parent) to top level', () => {
    const items = [makeItem({ id: 'a', index: 0 }), makeItem({ depth: 1, id: 'orphan', index: 1, parentId: 'ghost' })]

    const tree = buildSubagentTree(items)
    expect(tree).toHaveLength(2)
    expect(tree.map(n => n.item.id)).toEqual(['a', 'orphan'])
  })

  it('stable sort: children ordered by (depth, index) not insert order', () => {
    const items = [
      makeItem({ id: 'p', index: 0 }),
      makeItem({ depth: 1, id: 'c3', index: 2, parentId: 'p' }),
      makeItem({ depth: 1, id: 'c1', index: 0, parentId: 'p' }),
      makeItem({ depth: 1, id: 'c2', index: 1, parentId: 'p' })
    ]

    const tree = buildSubagentTree(items)
    expect(tree[0]!.children.map(n => n.item.id)).toEqual(['c1', 'c2', 'c3'])
  })
})

describe('aggregate', () => {
  it('sums tool counts and durations across subtree', () => {
    const items = [
      makeItem({ durationSeconds: 10, id: 'p', index: 0, status: 'completed', toolCount: 5 }),
      makeItem({ depth: 1, durationSeconds: 4, id: 'c1', index: 0, parentId: 'p', status: 'completed', toolCount: 3 }),
      makeItem({ depth: 1, durationSeconds: 2, id: 'c2', index: 1, parentId: 'p', status: 'completed', toolCount: 1 })
    ]

    const tree = buildSubagentTree(items)
    expect(tree[0]!.aggregate).toMatchObject({
      activeCount: 0,
      descendantCount: 2,
      totalDuration: 16,
      totalTools: 9
    })
  })

  it('counts queued + running as active', () => {
    const items = [
      makeItem({ id: 'p', index: 0, status: 'running' }),
      makeItem({ depth: 1, id: 'c1', index: 0, parentId: 'p', status: 'queued' }),
      makeItem({ depth: 1, id: 'c2', index: 1, parentId: 'p', status: 'completed' })
    ]

    const tree = buildSubagentTree(items)
    expect(tree[0]!.aggregate.activeCount).toBe(2)
  })
})

describe('widthByDepth', () => {
  it('returns empty array for empty tree', () => {
    expect(widthByDepth([])).toEqual([])
  })

  it('tallies nodes at each depth', () => {
    const items = [
      makeItem({ id: 'p1', index: 0 }),
      makeItem({ id: 'p2', index: 1 }),
      makeItem({ depth: 1, id: 'c1', index: 0, parentId: 'p1' }),
      makeItem({ depth: 1, id: 'c2', index: 1, parentId: 'p1' }),
      makeItem({ depth: 1, id: 'c3', index: 0, parentId: 'p2' }),
      makeItem({ depth: 2, id: 'gc1', index: 0, parentId: 'c1' })
    ]

    expect(widthByDepth(buildSubagentTree(items))).toEqual([2, 3, 1])
  })
})

describe('treeTotals', () => {
  it('folds a full tree into a single rollup', () => {
    const items = [
      makeItem({ id: 'p1', index: 0, toolCount: 5 }),
      makeItem({ id: 'p2', index: 1, toolCount: 2 }),
      makeItem({ depth: 1, id: 'c', index: 0, parentId: 'p1', toolCount: 3 })
    ]

    const totals = treeTotals(buildSubagentTree(items))
    expect(totals.descendantCount).toBe(3)
    expect(totals.totalTools).toBe(10)
    expect(totals.maxDepthFromHere).toBe(2)
  })

  it('returns zeros for empty tree', () => {
    expect(treeTotals([])).toEqual({
      activeCount: 0,
      costUsd: 0,
      descendantCount: 0,
      filesTouched: 0,
      hotness: 0,
      inputTokens: 0,
      maxDepthFromHere: 0,
      outputTokens: 0,
      totalDuration: 0,
      totalTools: 0
    })
  })
})

describe('flattenTree + descendantIds', () => {
  const items = [
    makeItem({ id: 'p', index: 0 }),
    makeItem({ depth: 1, id: 'c1', index: 0, parentId: 'p' }),
    makeItem({ depth: 2, id: 'gc', index: 0, parentId: 'c1' }),
    makeItem({ depth: 1, id: 'c2', index: 1, parentId: 'p' })
  ]

  it('flattens in visit order (depth-first, pre-order)', () => {
    const tree = buildSubagentTree(items)
    expect(flattenTree(tree).map(n => n.item.id)).toEqual(['p', 'c1', 'gc', 'c2'])
  })

  it('collects descendant ids excluding the node itself', () => {
    const tree = buildSubagentTree(items)
    expect(descendantIds(tree[0]!)).toEqual(['c1', 'gc', 'c2'])
  })
})

describe('sparkline', () => {
  it('returns empty string for empty input', () => {
    expect(sparkline([])).toBe('')
  })

  it('renders zeroes as spaces (not bottom glyph)', () => {
    expect(sparkline([0, 0])).toBe('  ')
  })

  it('scales to the max value', () => {
    const out = sparkline([1, 8])
    expect(out).toHaveLength(2)
    expect(out[1]).toBe('█')
  })

  it('sparse widths render as expected', () => {
    const out = sparkline([2, 3, 7, 4])
    expect(out).toHaveLength(4)
    expect([...out].every(ch => /[\s▁-█]/.test(ch))).toBe(true)
  })
})

describe('formatSummary', () => {
  const emptyTotals = {
    activeCount: 0,
    costUsd: 0,
    descendantCount: 0,
    filesTouched: 0,
    hotness: 0,
    inputTokens: 0,
    maxDepthFromHere: 0,
    outputTokens: 0,
    totalDuration: 0,
    totalTools: 0
  }

  it('collapses zero-valued components', () => {
    expect(formatSummary({ ...emptyTotals, descendantCount: 1 })).toBe('d0 · 1 agent')
  })

  it('emits rich summary with all pieces', () => {
    expect(
      formatSummary({
        ...emptyTotals,
        activeCount: 2,
        descendantCount: 7,
        maxDepthFromHere: 3,
        totalDuration: 134,
        totalTools: 124
      })
    ).toBe('d3 · 7 agents · 124 tools · 2m 14s · ⚡2')
  })
})

describe('fmtDuration', () => {
  it('formats under a minute as plain seconds', () => {
    expect(fmtDuration(0)).toBe('0s')
    expect(fmtDuration(42)).toBe('42s')
    expect(fmtDuration(59.4)).toBe('59s')
  })

  it('formats whole minutes without trailing seconds', () => {
    expect(fmtDuration(60)).toBe('1m')
    expect(fmtDuration(180)).toBe('3m')
  })

  it('mixes minutes and seconds', () => {
    expect(fmtDuration(134)).toBe('2m 14s')
    expect(fmtDuration(605)).toBe('10m 5s')
  })
})

describe('topLevelSubagents', () => {
  it('returns items with no parent', () => {
    const items = [makeItem({ id: 'a', index: 0 }), makeItem({ id: 'b', index: 1 })]
    expect(topLevelSubagents(items).map(s => s.id)).toEqual(['a', 'b'])
  })

  it('excludes children whose parent is present', () => {
    const items = [
      makeItem({ id: 'p', index: 0 }),
      makeItem({ depth: 1, id: 'c', index: 0, parentId: 'p' })
    ]

    expect(topLevelSubagents(items).map(s => s.id)).toEqual(['p'])
  })

  it('promotes orphans whose parent is missing', () => {
    const items = [makeItem({ id: 'a', index: 0 }), makeItem({ depth: 1, id: 'orphan', index: 1, parentId: 'ghost' })]
    expect(topLevelSubagents(items).map(s => s.id)).toEqual(['a', 'orphan'])
  })
})
