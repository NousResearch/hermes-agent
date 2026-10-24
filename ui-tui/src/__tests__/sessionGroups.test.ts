import type { ProjectTreeNode } from '@hermes/shared/gateway-events'
import { describe, expect, it } from 'vitest'

import {
  buildSwitcherRows,
  deriveSwitcherRows,
  groupSessionsByProject,
  nextGroupView,
  switcherRowKey
} from '../domain/sessionGroups.js'

const project = (id: string, label: string, sessionIds: string[]): ProjectTreeNode => ({ id, label, sessionIds })
const rows = (...ids: string[]) => ids.map(id => ({ id }))

describe('groupSessionsByProject', () => {
  const projects = [project('p_a', 'alpha', ['a1', 'a2']), project('p_b', 'beta', ['b1'])]

  it('buckets rows by owning project in tree order, keeping row order inside a group', () => {
    const groups = groupSessionsByProject(rows('b1', 'a2', 'a1'), projects, 'Other')

    expect(groups.map(g => [g.label, g.rows.map(r => r.id)])).toEqual([
      ['alpha', ['a2', 'a1']],
      ['beta', ['b1']]
    ])
  })

  it('drops projects with no visible rows and trails unclaimed rows under the fallback label', () => {
    const groups = groupSessionsByProject(rows('zz', 'b1'), projects, 'Other')

    expect(groups.map(g => [g.id, g.label])).toEqual([
      ['p_b', 'beta'],
      [null, 'Other']
    ])
    expect(groups.flatMap(g => g.rows.map(r => r.id))).toEqual(['b1', 'zz'])
  })

  it('lets the first project that claims a session own it', () => {
    const groups = groupSessionsByProject(
      rows('x'),
      [project('p_a', 'alpha', ['x']), project('p_b', 'beta', ['x'])],
      'Other'
    )

    expect(groups.map(g => g.label)).toEqual(['alpha'])
  })
})

describe('nextGroupView', () => {
  it('cycles collapsed → preview → all → collapsed', () => {
    expect(nextGroupView('collapsed', 10)).toBe('preview')
    expect(nextGroupView('preview', 10)).toBe('all')
    expect(nextGroupView('all', 10)).toBe('collapsed')
  })

  it('skips `all` when the group fits in the preview', () => {
    expect(nextGroupView('preview', 3)).toBe('collapsed')
  })
})

describe('buildSwitcherRows', () => {
  const live = rows('L1')
  const hist = rows('a1', 'a2', 'a3', 'a4', 'b1')

  const groups = groupSessionsByProject(
    hist,
    [project('p_a', 'alpha', ['a1', 'a2', 'a3', 'a4']), project('p_b', 'beta', ['b1'])],
    'Other'
  )

  const keys = (views: Record<string, 'all' | 'collapsed' | 'preview'>) =>
    buildSwitcherRows(live, hist, groups, views).map(switcherRowKey)

  it('defaults every group to its first 3 sessions under a selectable header', () => {
    expect(keys({})).toEqual(['live:L1', 'group:p_a', 'hist:a1', 'hist:a2', 'hist:a3', 'group:p_b', 'hist:b1'])
  })

  it('shows only the header when collapsed and every session when expanded', () => {
    expect(keys({ p_a: 'collapsed', p_b: 'collapsed' })).toEqual(['live:L1', 'group:p_a', 'group:p_b'])
    expect(keys({ p_a: 'all' })).toContain('hist:a4')
  })

  it('reports how many sessions each header currently shows', () => {
    const header = buildSwitcherRows(live, hist, groups, {}).find(r => r.kind === 'header')

    expect(header).toMatchObject({ shown: 3, view: 'preview' })
  })

  it('stays flat (no headers) when ungrouped', () => {
    expect(buildSwitcherRows(live, hist, [], {}).map(switcherRowKey)).toEqual([
      'live:L1',
      'hist:a1',
      'hist:a2',
      'hist:a3',
      'hist:a4',
      'hist:b1'
    ])
  })
})

describe('deriveSwitcherRows (search)', () => {
  const hist = [
    { id: 'a1', title: 'Fix reconnect' },
    { id: 'a2', title: 'Docs' },
    { id: 'b1', title: 'Fix parser' }
  ]

  const live = [{ id: 'L1', title: 'Live fix' }]

  const groups = groupSessionsByProject(
    hist,
    [project('p_a', 'alpha', ['a1', 'a2']), project('p_b', 'beta', ['b1'])],
    'Other'
  )

  const keys = (query: string, grouped = true, views = {}) =>
    deriveSwitcherRows(live, hist, groups, { grouped, query, views }).map(switcherRowKey)

  it('is the plain grouped view when the query is blank', () => {
    expect(keys('  ')).toEqual(buildSwitcherRows(live, hist, groups, {}).map(switcherRowKey))
  })

  it('filters sessions by title and drops groups with no match', () => {
    expect(keys('fix')).toEqual(['live:L1', 'group:p_a', 'hist:a1', 'group:p_b', 'hist:b1'])
  })

  it('keeps a whole project when its label matches, ignoring a collapsed view', () => {
    expect(keys('alpha', true, { p_a: 'collapsed' })).toEqual(['group:p_a', 'hist:a1', 'hist:a2'])
  })

  it('requires every token, across the project label and the session text', () => {
    expect(keys('beta fix')).toEqual(['group:p_b', 'hist:b1'])
    expect(keys('alpha parser')).toEqual([])
  })

  it('still matches project names in the flat list', () => {
    expect(keys('beta', false)).toEqual(['hist:b1'])
  })
})
