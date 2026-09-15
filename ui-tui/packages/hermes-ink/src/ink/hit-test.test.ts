import { describe, expect, it } from 'vitest'

import { appendChildNode, createNode } from './dom.js'
import { dispatchClick, hitTest } from './hit-test.js'
import { nodeCache } from './node-cache.js'

const rect = (node: ReturnType<typeof createNode>, x: number, y: number, width: number, height: number) => {
  nodeCache.set(node, { x, y, width, height })
}

describe('hit-test', () => {
  it('hits absolutely positioned children that paint outside their parent rect', () => {
    const root = createNode('ink-root')
    const parent = createNode('ink-box')
    const wrapper = createNode('ink-box')
    const overlay = createNode('ink-box')
    const row = createNode('ink-box')
    const seen: string[] = []

    appendChildNode(root, parent)
    appendChildNode(parent, wrapper)
    appendChildNode(wrapper, overlay)
    appendChildNode(overlay, row)

    overlay.style.position = 'absolute'
    row._eventHandlers = { onClick: () => seen.push('row') }

    rect(root, 0, 0, 120, 40)
    rect(parent, 0, 30, 120, 1)
    rect(wrapper, 0, 30, 120, 1)
    rect(overlay, 0, 20, 96, 6)
    rect(row, 1, 22, 80, 1)

    expect(hitTest(root, 2, 22)).toBe(row)
    expect(dispatchClick(root, 2, 22)).toBe(true)
    expect(seen).toEqual(['row'])
  })
})
