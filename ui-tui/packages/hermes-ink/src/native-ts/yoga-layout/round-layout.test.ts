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

interface NodeSpec {
  w?: number
  h?: number
  pad?: number
  mar?: number
  row: boolean
  grow?: number
  kids: NodeSpec[]
}

const buildMutationTree = (spec: NodeSpec) => {
  const all: Node[] = []
  const makeNode = (value: NodeSpec): Node => {
    const node = Yoga.Node.create()

    if (value.w !== undefined) node.setWidth(value.w)
    if (value.h !== undefined) node.setHeight(value.h)
    if (value.pad !== undefined) node.setPadding(1, value.pad)
    if (value.mar !== undefined) node.setMargin(1, value.mar)
    if (value.row) node.setFlexDirection(FlexDirection.Row)
    if (value.grow !== undefined) node.setFlexGrow(value.grow)

    all.push(node)
    value.kids.forEach((child, index) => node.insertChild(makeNode(child), index))

    return node
  }

  return { all, root: makeNode(spec) }
}

describe('incremental layout rounding', () => {
  it('keeps rounding work flat when only the clock changes', () => {
    const results = [50, 500, 5000].map(rowCount => {
      const config = Yoga.Config.create()
      config.setPointScaleFactor(2)

      const root = Yoga.Node.create(config)
      root.setWidth(80)
      root.setHeight(40)

      const transcript = Yoga.Node.create(config)
      transcript.setHeight(39)
      root.insertChild(transcript, 0)

      for (let index = 0; index < rowCount; index++) {
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
      root.freeRecursive()
      Yoga.Config.destroy(config)

      return { rounded: counters.rounded, roundSkips: counters.roundSkips }
    })

    expect(results).toEqual([
      { rounded: 2, roundSkips: 1 },
      { rounded: 2, roundSkips: 1 },
      { rounded: 2, roundSkips: 1 }
    ])
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

  it('rounds a fractional child after its whole-pixel parent hits the layout cache', () => {
    const root = Yoga.Node.create()
    root.setWidth(120)
    root.setHeight(40)
    const row = Yoga.Node.create()
    row.setWidth(120)
    row.setHeight(2)
    const leaf = Yoga.Node.create()
    leaf.setWidth(10.4)
    leaf.setHeight(1.6)
    row.insertChild(leaf, 0)
    root.insertChild(row, 0)
    root.calculateLayout(120, 40)

    leaf.setWidth(11.4)
    leaf.setHeight(2.6)
    root.calculateLayout(120, 40)

    expect(leaf.getComputedLayout()).toMatchObject({ width: 11, height: 3 })
    root.freeRecursive()
  })

  it('does not mix cached rounded coordinates with raw dimensions', () => {
    const spec: NodeSpec = {
      h: 4.917283,
      pad: 1.934923,
      row: false,
      kids: [
        {
          w: 11.007846,
          row: false,
          kids: [
            {
              mar: 1.306334,
              row: true,
              kids: [
                {
                  mar: 0.199071,
                  row: true,
                  grow: 1.522763,
                  kids: [{ w: 4.6117, h: 5.852709, row: false, kids: [] }]
                }
              ]
            },
            {
              w: 40.14272,
              pad: 1.043468,
              mar: 1.495136,
              row: false,
              kids: [
                {
                  w: 8.335441,
                  row: false,
                  kids: [
                    { h: 5.249866, mar: 0.997316, row: true, grow: 1.554555, kids: [] },
                    { h: 6.335704, mar: 0.817136, row: false, grow: 0.079169, kids: [] }
                  ]
                },
                {
                  w: 32.935694,
                  row: false,
                  grow: 1.781838,
                  kids: [
                    { w: 38.068879, pad: 0.696862, row: true, kids: [] },
                    { pad: 1.66287, row: false, kids: [] }
                  ]
                }
              ]
            }
          ]
        }
      ]
    }
    const mutations = [
      { index: 10, kind: 'height', value: 2.823024 },
      { index: 6, kind: 'margin', value: 0.858021 },
      { index: 3, kind: 'grow', value: 1.321691 },
      { index: 9, kind: 'grow', value: 0.022362 },
      { index: 6, kind: 'grow', value: 0.812981 },
      { index: 7, kind: 'width', value: 5.644585 },
      { index: 2, kind: 'width', value: 20.498063 },
      { index: 4, kind: 'grow', value: 1.653743 },
      { index: 1, kind: 'height', value: 2.709275 },
      { index: 1, kind: 'margin', value: 2.411063 }
    ] as const
    const applyMutation = (all: Node[], mutation: (typeof mutations)[number]) => {
      const node = all[mutation.index]!

      if (mutation.kind === 'width') node.setWidth(mutation.value)
      else if (mutation.kind === 'height') node.setHeight(mutation.value)
      else if (mutation.kind === 'margin') node.setMargin(1, mutation.value)
      else node.setFlexGrow(mutation.value)
    }
    const incremental = buildMutationTree(spec)
    incremental.root.calculateLayout(96, 5)

    for (const mutation of mutations) {
      applyMutation(incremental.all, mutation)
      incremental.root.calculateLayout(96, 5)
    }

    const fresh = buildMutationTree(spec)
    mutations.forEach(mutation => applyMutation(fresh.all, mutation))
    fresh.root.calculateLayout(96, 5)

    expect(incremental.all[11]!.getComputedHeight()).toBe(fresh.all[11]!.getComputedHeight())
    expect(fresh.all[11]!.getComputedHeight()).toBe(2)
    incremental.root.freeRecursive()
    fresh.root.freeRecursive()
  })
})
