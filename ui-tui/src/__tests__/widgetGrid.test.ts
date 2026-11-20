import { describe, expect, it } from 'vitest'

import { type GridTrackSize, layoutGridAreas, layoutWidgetGrid, resolveGridTracks } from '../lib/widgetGrid.js'

describe('layoutWidgetGrid', () => {
  it('falls back to a single column on narrow widths', () => {
    const layout = layoutWidgetGrid({
      items: [{ id: 'a' }, { id: 'b' }],
      maxColumns: 3,
      minColumnWidth: 40,
      width: 35
    })

    expect(layout.columnCount).toBe(1)
    expect(layout.columns).toEqual([35])
    expect(layout.rows).toEqual([[{ col: 0, id: 'a', span: 1, width: 35 }], [{ col: 0, id: 'b', span: 1, width: 35 }]])
  })

  it('packs spans left-to-right and wraps to the next row', () => {
    const layout = layoutWidgetGrid({
      gap: 2,
      items: [
        { id: 'a', span: 1 },
        { id: 'b', span: 2 },
        { id: 'c', span: 1 }
      ],
      maxColumns: 3,
      minColumnWidth: 30,
      width: 100
    })

    expect(layout.columnCount).toBe(3)
    expect(layout.columns).toEqual([32, 32, 32])
    expect(layout.rows).toEqual([
      [
        { col: 0, id: 'a', span: 1, width: 32 },
        { col: 1, id: 'b', span: 2, width: 66 }
      ],
      [{ col: 0, id: 'c', span: 1, width: 32 }]
    ])
  })

  it('clamps spans to available columns', () => {
    const layout = layoutWidgetGrid({
      gap: 1,
      items: [{ id: 'huge', span: 9 }],
      maxColumns: 2,
      minColumnWidth: 20,
      width: 50
    })

    expect(layout.columnCount).toBe(2)
    expect(layout.rows[0]?.[0]).toEqual({
      col: 0,
      id: 'huge',
      span: 2,
      width: 50
    })
  })

  it('honors an exact column count when the grid has room', () => {
    const layout = layoutWidgetGrid({
      columns: 4,
      gap: 1,
      items: Array.from({ length: 8 }, (_, idx) => ({ id: `cell-${idx}` })),
      width: 43
    })

    expect(layout.columnCount).toBe(4)
    expect(layout.columns).toEqual([10, 10, 10, 10])
    expect(layout.rows).toHaveLength(2)
  })

  it('renders sparse explicit starts without collapsing holes', () => {
    const layout = layoutWidgetGrid({
      columns: 4,
      gap: 1,
      items: [
        { colStart: 0, id: 'a' },
        { colStart: 2, id: 'b' },
        { colStart: 3, id: 'c' }
      ],
      width: 43
    })

    expect(layout.rows).toEqual([
      [
        { col: 0, id: 'a', span: 1, width: 10 },
        { col: 2, id: 'b', span: 1, width: 10 },
        { col: 3, id: 'c', span: 1, width: 10 }
      ]
    ])
  })
})

describe('resolveGridTracks', () => {
  it('splits equal fr tracks with the remainder spread left-to-right', () => {
    expect(resolveGridTracks(10, 0, [{ fr: 1 }, { fr: 1 }, { fr: 1 }])).toEqual([4, 3, 3])
  })

  it('gives fixed tracks their size and shares the leftover by weight', () => {
    // usable = 40 - 2*1 = 38; fixed 10; leftover 28 over 1fr+2fr.
    expect(resolveGridTracks(40, 1, [10, { fr: 1 }, { fr: 2 }])).toEqual([10, 10, 18])
  })

  it('pins tracks that fall below their min and re-solves the rest', () => {
    expect(resolveGridTracks(20, 0, [{ fr: 1 }, { fr: 1, min: 15 }])).toEqual([5, 15])
  })

  it('never lets the sum exceed the usable axis when tracks fit at their minimums', () => {
    const cases: Array<{ gap: number; total: number; tracks: GridTrackSize[] }> = [
      { gap: 1, total: 80, tracks: [12, { fr: 1 }, { fr: 3, min: 20 }, 6] },
      { gap: 2, total: 33, tracks: [{ fr: 1 }, { fr: 1 }, { fr: 1 }, { fr: 1 }] },
      { gap: 0, total: 9, tracks: [4, 4, { fr: 1 }] }
    ]

    for (const { gap, total, tracks } of cases) {
      const sizes = resolveGridTracks(total, gap, tracks)
      const usable = total - gap * (tracks.length - 1)

      expect(sizes.reduce((acc, s) => acc + s, 0)).toBeLessThanOrEqual(Math.max(tracks.length, usable))
      expect(sizes.every(s => s >= 1)).toBe(true)
    }
  })

  it('shaves overflow off trailing tracks when fixed sizes exceed the axis', () => {
    const sizes = resolveGridTracks(20, 0, [15, 15, { fr: 1 }])

    expect(sizes.reduce((acc, s) => acc + s, 0)).toBeLessThanOrEqual(20)
    expect(sizes[0]).toBe(15)
    expect(sizes.every(s => s >= 1)).toBe(true)
  })
})

describe('layoutGridAreas', () => {
  it('solves a rowSpan cell so it occupies both rows and blocks placement beneath it', () => {
    const layout = layoutGridAreas({
      columns: 2,
      gap: 0,
      height: 10,
      items: [{ id: 'tall', rowSpan: 2 }, { id: 'b' }, { id: 'c' }],
      rowGap: 0,
      width: 20
    })

    const [tall, b, c] = layout.cells

    expect(tall).toMatchObject({ col: 0, height: 10, row: 0, rowSpan: 2, width: 10, x: 0, y: 0 })
    expect(b).toMatchObject({ col: 1, row: 0, x: 10, y: 0 })
    // c cannot go under `tall` — it lands in the second row's free column.
    expect(c).toMatchObject({ col: 1, row: 1, x: 10, y: 5 })
  })

  it('fills holes densely (auto-flow dense) after a colSpan pushes a wrap', () => {
    const layout = layoutGridAreas({
      columns: 3,
      gap: 0,
      height: 6,
      items: [{ id: 'a' }, { colSpan: 2, id: 'wide' }, { id: 'filler' }],
      width: 30
    })

    const byId = Object.fromEntries(layout.cells.map(cell => [cell.id, cell]))

    expect(byId['a']).toMatchObject({ col: 0, row: 0 })
    expect(byId['wide']).toMatchObject({ col: 1, colSpan: 2, row: 0 })
    expect(byId['filler']).toMatchObject({ col: 0, row: 1 })
  })

  it('spans include the gaps they bridge', () => {
    const layout = layoutGridAreas({
      columns: [{ fr: 1 }, { fr: 1 }, { fr: 1 }],
      gap: 2,
      height: 11,
      items: [{ colSpan: 3, id: 'full' }, { id: 'a' }, { id: 'b' }, { rowSpan: 2, id: 'tall' }],
      rowGap: 1,
      width: 32
    })

    const byId = Object.fromEntries(layout.cells.map(cell => [cell.id, cell]))

    // 3 tracks over usable 28 → [10, 9, 9]; full spans all three + two gaps.
    expect(layout.columnSizes).toEqual([10, 9, 9])
    expect(byId['full']!.width).toBe(32)
    // 3 rows over usable 9 → [3, 3, 3]; tall spans rows 2-3 + one rowGap.
    expect(layout.rowSizes).toEqual([3, 3, 3])
    expect(byId['tall']!.height).toBe(7)
    expect(byId['tall']!.y).toBe(4)
  })

  it('honors explicit col/row pins and grows implicit rows beneath explicit tracks', () => {
    const layout = layoutGridAreas({
      columns: 2,
      gap: 0,
      height: 12,
      items: [{ col: 1, id: 'pinned', row: 2 }, { id: 'auto-a' }, { id: 'auto-b' }],
      rowGap: 0,
      rows: 2,
      width: 10
    })

    const byId = Object.fromEntries(layout.cells.map(cell => [cell.id, cell]))

    // The pin forces a third row beyond the two explicit tracks.
    expect(layout.rowCount).toBe(3)
    expect(byId['pinned']).toMatchObject({ col: 1, row: 2 })
    expect(byId['auto-a']).toMatchObject({ col: 0, row: 0 })
    expect(byId['auto-b']).toMatchObject({ col: 1, row: 0 })
  })

  it('supports weighted and fixed row tracks', () => {
    const layout = layoutGridAreas({
      columns: 1,
      gap: 0,
      height: 20,
      items: [{ id: 'header' }, { id: 'body' }, { id: 'footer' }],
      rowGap: 0,
      rows: [3, { fr: 1 }, 3],
      width: 40
    })

    expect(layout.rowSizes).toEqual([3, 14, 3])

    const byId = Object.fromEntries(layout.cells.map(cell => [cell.id, cell]))

    expect(byId['header']!.height).toBe(3)
    expect(byId['body']).toMatchObject({ height: 14, y: 3 })
    expect(byId['footer']).toMatchObject({ height: 3, y: 17 })
  })

  it('clamps colSpan to the column count', () => {
    const layout = layoutGridAreas({
      columns: 2,
      gap: 0,
      height: 4,
      items: [{ colSpan: 9, id: 'huge' }],
      width: 10
    })

    expect(layout.cells[0]).toMatchObject({ col: 0, colSpan: 2, width: 10 })
  })
})
