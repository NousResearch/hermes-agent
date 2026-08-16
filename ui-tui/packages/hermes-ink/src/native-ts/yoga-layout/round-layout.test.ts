import { describe, expect, it } from 'vitest'

import Yoga, { FlexDirection, getYogaCounters, type Node } from './index.js'

const snapshot = (node: Node): number[] => {
  const result = [node.getComputedLeft(), node.getComputedTop(), node.getComputedWidth(), node.getComputedHeight()]

  for (let index = 0; index < node.getChildCount(); index++) {
    result.push(...snapshot(node.getChild(index)))
  }

  return result
}

const buildTree = (rootWidth: number, widths: number[], scale: number) => {
  const config = Yoga.Config.create()
  config.setPointScaleFactor(scale)
  const root = Yoga.Node.create(config)
  root.setFlexDirection(FlexDirection.Column)
  root.setWidth(rootWidth)
  root.setHeight(20)
  const leaves: Node[] = []

  for (let groupIndex = 0; groupIndex < 4; groupIndex++) {
    const group = Yoga.Node.create(config)
    group.setFlexDirection(FlexDirection.Row)
    group.setHeight(3.125)
    root.insertChild(group, groupIndex)

    for (let leafIndex = 0; leafIndex < 8; leafIndex++) {
      const leaf = Yoga.Node.create(config)
      leaf.setWidth(widths[groupIndex * 8 + leafIndex]!)
      leaf.setHeight(1.125 + (leafIndex % 3) * 0.25)
      group.insertChild(leaf, leafIndex)
      leaves.push(leaf)
    }
  }

  return { config, leaves, root }
}

describe('incremental layout rounding', () => {
  it('skips an unchanged transcript subtree when only the clock changes', () => {
    const config = Yoga.Config.create()
    config.setPointScaleFactor(2)

    const root = Yoga.Node.create(config)
    root.setWidth(80)
    root.setHeight(40)

    const transcript = Yoga.Node.create(config)
    transcript.setHeight(39)
    root.insertChild(transcript, 0)

    for (let index = 0; index < 500; index++) {
      const row = Yoga.Node.create(config)
      row.setWidth(20.25)
      row.setHeight(0.25)
      transcript.insertChild(row, index)
    }

    const clock = Yoga.Node.create(config)
    clock.setWidth(5.25)
    clock.setHeight(1)
    root.insertChild(clock, 1)

    root.calculateLayout(80, 40)
    const transcriptWidth = transcript.getComputedWidth()

    clock.setWidth(6.25)
    root.calculateLayout(80, 40)

    const counters = getYogaCounters()
    expect(clock.getComputedWidth()).toBe(6.5)
    expect(transcript.getComputedWidth()).toBe(transcriptWidth)
    expect(counters.rounded).toBeLessThanOrEqual(4)
    expect(counters.roundSkips).toBe(1)

    root.freeRecursive()
    Yoga.Config.destroy(config)
  })

  it('re-rounds cached raw geometry when the point scale changes', () => {
    const config = Yoga.Config.create()
    config.setPointScaleFactor(2)

    const root = Yoga.Node.create(config)
    root.setWidth(20)
    root.setHeight(10)

    const child = Yoga.Node.create(config)
    child.setWidth(10.25)
    child.setHeight(1)
    root.insertChild(child, 0)

    root.calculateLayout(20, 10)
    expect(child.getComputedWidth()).toBe(10.5)

    config.setPointScaleFactor(4)
    child.setWidth(10.125)
    root.calculateLayout(20, 10)

    expect(child.getComputedWidth()).toBe(10.25)

    config.setPointScaleFactor(0)
    root.calculateLayout(20, 10)

    expect(child.getComputedWidth()).toBe(10.125)

    root.freeRecursive()
    Yoga.Config.destroy(config)
  })

  it('matches a fresh full layout across leaf, root, and scale changes', () => {
    const widths = Array.from({ length: 32 }, (_, index) => 1.125 + (index % 5) * 0.375)
    let rootWidth = 40.25
    let scale = 2
    const incremental = buildTree(rootWidth, widths, scale)

    for (let step = 0; step < 24; step++) {
      if (step % 6 === 0) {
        scale = scale === 2 ? 4 : 2
        incremental.config.setPointScaleFactor(scale)
      } else if (step % 5 === 0) {
        rootWidth += 0.375
        incremental.root.setWidth(rootWidth)
      } else {
        const leafIndex = (step * 7) % widths.length
        widths[leafIndex]! += 0.125
        incremental.leaves[leafIndex]!.setWidth(widths[leafIndex]!)
      }

      incremental.root.calculateLayout(rootWidth, 20)
      const fresh = buildTree(rootWidth, widths, scale)
      fresh.root.calculateLayout(rootWidth, 20)

      expect(snapshot(incremental.root), `step ${step}`).toEqual(snapshot(fresh.root))

      fresh.root.freeRecursive()
      Yoga.Config.destroy(fresh.config)
    }

    incremental.root.freeRecursive()
    Yoga.Config.destroy(incremental.config)
  })
})
