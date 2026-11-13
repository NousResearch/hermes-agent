import type { SessionListRow } from '@hermes/shared/gateway-events'
import type { SessionActiveItem } from '@hermes/shared/gateway-types'
import { describe, expect, it } from 'vitest'

import { buildSessionTree } from '../components/sessionSidebar.js'

const live = (id: string): SessionActiveItem => ({ id, status: 'idle', title: id })

describe('buildSessionTree', () => {
  it('nests live continuations under a listed lineage root', () => {
    const tree = buildSessionTree(
      [{ id: 'child', _lineage_root_id: 'root' } as SessionListRow],
      [live('child'), live('root')]
    )

    expect(tree).toEqual([{ session: live('root'), children: [{ session: live('child'), children: [] }] }])
  })

  it('keeps orphaned sessions flat and in live-list order', () => {
    const tree = buildSessionTree(
      [{ id: 'orphan', _lineage_root_id: 'missing' } as SessionListRow],
      [live('orphan'), live('later'), live('earlier')]
    )

    expect(tree.map(node => node.session.id)).toEqual(['orphan', 'later', 'earlier'])
    expect(tree.every(node => node.children.length === 0)).toBe(true)
  })

  it('returns an empty tree for empty input', () => {
    expect(buildSessionTree([], [])).toEqual([])
  })

  it('deduplicates live ids while keeping the first item and position', () => {
    const duplicate = { ...live('same'), status: 'working' as const }
    const tree = buildSessionTree([], [live('same'), live('other'), duplicate])

    expect(tree.map(node => node.session.id)).toEqual(['same', 'other'])
    expect(tree[0]?.session.status).toBe('idle')
  })
})
