export interface WidgetGridItem {
  colSpan?: number
  colStart?: number
  id: string
  span?: number
}

export interface WidgetGridCell {
  col: number
  id: string
  span: number
  width: number
}

export interface WidgetGridLayout {
  columnCount: number
  columns: number[]
  rows: WidgetGridCell[][]
}

export interface WidgetGridLayoutOptions {
  /**
   * Explicit column count (equal shares) or a grid-template-style track list
   * (fixed cell counts / weighted `fr` shares). Omitted: auto from width.
   */
  columns?: GridTrackSize[] | number
  gap?: number
  items: WidgetGridItem[]
  maxColumns?: number
  minColumnWidth?: number
  width: number
}

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value))

const toInt = (value: number, fallback: number) => {
  if (!Number.isFinite(value)) {
    return fallback
  }

  return Math.trunc(value)
}

const columnCountForWidth = (width: number, minColumnWidth: number, gap: number, maxColumns: number) => {
  const safeWidth = Math.max(1, toInt(width, 1))
  const safeMinWidth = Math.max(1, toInt(minColumnWidth, 1))
  const safeGap = Math.max(0, toInt(gap, 0))
  const safeMaxColumns = Math.max(1, toInt(maxColumns, 1))
  const count = Math.floor((safeWidth + safeGap) / (safeMinWidth + safeGap))

  return clamp(count || 1, 1, safeMaxColumns)
}

const buildColumnWidths = (width: number, columnCount: number, gap: number) =>
  resolveGridTracks(
    width,
    gap,
    Array.from({ length: Math.max(1, toInt(columnCount, 1)) }, () => ({ fr: 1 }))
  )

const spanWidth = (columns: number[], colStart: number, span: number, gap: number) => {
  const end = Math.min(columns.length, colStart + span)
  const width = columns.slice(colStart, end).reduce((acc, value) => acc + value, 0)
  const safeGap = Math.max(0, toInt(gap, 0))

  return width + safeGap * Math.max(0, end - colStart - 1)
}

export const widgetGridSpanWidth = spanWidth

const itemSpan = (item: WidgetGridItem, columnCount: number) =>
  clamp(toInt(item.colSpan ?? item.span ?? 1, 1), 1, columnCount)

const itemColStart = (item: WidgetGridItem, columnCount: number, span: number) => {
  if (item.colStart === undefined) {
    return null
  }

  return clamp(toInt(item.colStart, 0), 0, Math.max(0, columnCount - span))
}

const rangeIsFree = (occupied: boolean[], colStart: number, span: number) => {
  for (let col = colStart; col < colStart + span; col++) {
    if (occupied[col]) {
      return false
    }
  }

  return true
}

const occupyRange = (occupied: boolean[], colStart: number, span: number) => {
  for (let col = colStart; col < colStart + span; col++) {
    occupied[col] = true
  }
}

const firstFreeCol = (occupied: boolean[], span: number) => {
  for (let col = 0; col <= occupied.length - span; col++) {
    if (rangeIsFree(occupied, col, span)) {
      return col
    }
  }

  return null
}

const sortRow = (row: WidgetGridCell[]) => row.sort((a, b) => a.col - b.col)

export function layoutWidgetGrid({
  columns: requestedColumns,
  gap = 1,
  items,
  maxColumns = 3,
  minColumnWidth = 28,
  width
}: WidgetGridLayoutOptions): WidgetGridLayout {
  const safeGap = Math.max(0, toInt(gap, 1))
  const safeWidth = Math.max(1, toInt(width, 1))
  const maxDrawableColumns = safeGap > 0 ? Math.max(1, Math.floor((safeWidth + safeGap) / (safeGap + 1))) : safeWidth

  const trackList = Array.isArray(requestedColumns) && requestedColumns.length ? requestedColumns : null

  const columnCount = trackList
    ? trackList.length
    : requestedColumns === undefined || Array.isArray(requestedColumns)
      ? columnCountForWidth(safeWidth, minColumnWidth, safeGap, maxColumns)
      : clamp(toInt(requestedColumns, 1), 1, maxDrawableColumns)

  const columns = trackList
    ? resolveGridTracks(safeWidth, safeGap, trackList)
    : buildColumnWidths(width, columnCount, safeGap)

  const rows: WidgetGridCell[][] = []
  let row: WidgetGridCell[] = []
  let occupied = Array.from({ length: columnCount }, () => false)

  const pushRow = () => {
    rows.push(sortRow(row))
    row = []
    occupied = Array.from({ length: columnCount }, () => false)
  }

  for (const item of items) {
    const wantedSpan = itemSpan(item, columnCount)
    const explicitCol = itemColStart(item, columnCount, wantedSpan)
    let col = explicitCol ?? firstFreeCol(occupied, wantedSpan)

    if (col === null || (explicitCol !== null && !rangeIsFree(occupied, explicitCol, wantedSpan))) {
      if (row.length > 0) {
        pushRow()
      }

      col = explicitCol ?? 0
    }

    row.push({
      col,
      id: item.id,
      span: wantedSpan,
      width: spanWidth(columns, col, wantedSpan, safeGap)
    })

    occupyRange(occupied, col, wantedSpan)
  }

  if (row.length > 0) {
    rows.push(sortRow(row))
  }

  return { columnCount, columns, rows }
}

// ── Track solver (grid-template-columns/rows for character cells) ──────────
//
// A track is either a fixed cell count (`number`) or a fractional share
// (`{ fr, min? }`) of whatever space the fixed tracks leave over — the same
// fixed-vs-flex split the desktop pane-shell's track model uses, solved in
// integer terminal cells. `min` defaults to 1 so a track never disappears.

export type GridTrackSize = number | { fr: number; min?: number }

const trackFr = (track: GridTrackSize) => (typeof track === 'number' ? 0 : Math.max(0.0001, track.fr))

const trackMin = (track: GridTrackSize) => (typeof track === 'number' ? 1 : Math.max(1, toInt(track.min ?? 1, 1)))

/** Floor-divide `budget` across `weights`, spreading the remainder left-to-right. */
const distributeByWeight = (budget: number, weights: number[]) => {
  const total = weights.reduce((acc, w) => acc + w, 0)

  if (total <= 0 || budget <= 0) {
    return weights.map(() => 0)
  }

  const shares = weights.map(w => Math.floor((budget * w) / total))
  let remainder = budget - shares.reduce((acc, s) => acc + s, 0)

  for (let i = 0; remainder > 0 && i < shares.length; i++, remainder--) {
    shares[i]! += 1
  }

  return shares
}

/**
 * Solve track sizes for a `total`-cell axis with `gap` cells between tracks.
 * Fixed tracks take their size; `fr` tracks share the leftover by weight,
 * re-pinning any track that falls below its `min` and re-solving the rest.
 * Every track ends up ≥ 1 cell; when the axis genuinely can't fit, the
 * overflow is shaved off the trailing tracks so the sum never exceeds the
 * drawable width unless every track is already at 1.
 */
export function resolveGridTracks(total: number, gap: number, tracks: GridTrackSize[]): number[] {
  const count = tracks.length

  if (!count) {
    return []
  }

  const safeGap = Math.max(0, toInt(gap, 0))
  const usable = Math.max(count, toInt(total, count) - safeGap * (count - 1))
  const sizes = Array.from({ length: count }, () => 0)
  let remaining = usable
  let unpinned: number[] = []

  tracks.forEach((track, idx) => {
    if (typeof track === 'number') {
      sizes[idx] = Math.max(1, toInt(track, 1))
      remaining -= sizes[idx]!
    } else {
      unpinned.push(idx)
    }
  })

  while (unpinned.length) {
    const shares = distributeByWeight(
      Math.max(0, remaining),
      unpinned.map(idx => trackFr(tracks[idx]!))
    )

    const violating = unpinned.filter((idx, i) => shares[i]! < trackMin(tracks[idx]!))

    if (!violating.length) {
      unpinned.forEach((idx, i) => {
        sizes[idx] = shares[i]!
      })

      break
    }

    for (const idx of violating) {
      sizes[idx] = trackMin(tracks[idx]!)
      remaining -= sizes[idx]!
    }

    unpinned = unpinned.filter(idx => !violating.includes(idx))
  }

  let overflow = sizes.reduce((acc, s) => acc + s, 0) - usable

  for (let idx = count - 1; idx >= 0 && overflow > 0; idx--) {
    const give = Math.min(overflow, sizes[idx]! - 1)

    sizes[idx] -= give
    overflow -= give
  }

  return sizes
}

// ── 2D area layout (the workspace mode) ────────────────────────────────────
//
// `layoutGridAreas` is the full two-axis grid: explicit column AND row tracks,
// items with `col`/`row` pins and `colSpan`/`rowSpan`, dense first-fit
// auto-placement, and each cell solved to an absolute `{ x, y, width, height }`
// rect. It is the terminal-cell equivalent of the desktop zone editor's
// `GridLayout` (rowPercents / columnPercents / cellChildMap with merged
// spans) — the renderer absolutely positions each rect, which is what makes
// `rowSpan` representable at all under Yoga flexbox.

export interface GridAreaItem {
  id: string
  /** Explicit column start (0-based). Explicitly pinned items may overlap. */
  col?: number
  colSpan?: number
  /** Explicit row start (0-based). Rows grow implicitly as needed. */
  row?: number
  rowSpan?: number
}

export interface GridAreaCell {
  col: number
  colSpan: number
  height: number
  id: string
  row: number
  rowSpan: number
  width: number
  x: number
  y: number
}

export interface GridAreasLayout {
  cells: GridAreaCell[]
  columnCount: number
  columnSizes: number[]
  height: number
  rowCount: number
  rowSizes: number[]
  width: number
}

export interface GridAreasOptions {
  /** Column tracks — a count (equal `fr` shares) or explicit track list. */
  columns: GridTrackSize[] | number
  gap?: number
  height: number
  items: GridAreaItem[]
  rowGap?: number
  /** Row tracks. Omitted or short: implicit rows are `{ fr: 1 }`. */
  rows?: GridTrackSize[] | number
  width: number
}

const normalizeTracks = (tracks: GridTrackSize[] | number | undefined, fallbackCount: number): GridTrackSize[] => {
  if (Array.isArray(tracks) && tracks.length) {
    return tracks
  }

  const count = Math.max(1, toInt(typeof tracks === 'number' ? tracks : fallbackCount, 1))

  return Array.from({ length: count }, () => ({ fr: 1 }))
}

interface PlacedArea {
  col: number
  colSpan: number
  id: string
  row: number
  rowSpan: number
}

const ensureOccupancyRows = (occupied: boolean[][], rowCount: number, columnCount: number) => {
  while (occupied.length < rowCount) {
    occupied.push(Array.from({ length: columnCount }, () => false))
  }
}

const areaIsFree = (occupied: boolean[][], row: number, col: number, rowSpan: number, colSpan: number) => {
  ensureOccupancyRows(occupied, row + rowSpan, occupied[0]?.length ?? 1)

  for (let r = row; r < row + rowSpan; r++) {
    for (let c = col; c < col + colSpan; c++) {
      if (occupied[r]![c]) {
        return false
      }
    }
  }

  return true
}

const occupyArea = (occupied: boolean[][], row: number, col: number, rowSpan: number, colSpan: number) => {
  ensureOccupancyRows(occupied, row + rowSpan, occupied[0]?.length ?? 1)

  for (let r = row; r < row + rowSpan; r++) {
    for (let c = col; c < col + colSpan; c++) {
      occupied[r]![c] = true
    }
  }
}

/** Dense first-fit auto-placement (CSS `grid-auto-flow: row dense`). */
const placeGridItems = (items: GridAreaItem[], columnCount: number): PlacedArea[] => {
  const occupied: boolean[][] = [Array.from({ length: columnCount }, () => false)]

  return items.map(item => {
    const colSpan = clamp(toInt(item.colSpan ?? 1, 1), 1, columnCount)
    const rowSpan = Math.max(1, toInt(item.rowSpan ?? 1, 1))
    const pinnedCol = item.col === undefined ? null : clamp(toInt(item.col, 0), 0, columnCount - colSpan)
    const pinnedRow = item.row === undefined ? null : Math.max(0, toInt(item.row, 0))

    let row: number
    let col: number

    if (pinnedRow !== null && pinnedCol !== null) {
      row = pinnedRow
      col = pinnedCol
    } else if (pinnedRow !== null) {
      // Pinned row: first free column run, overlapping at col 0 when full.
      col = 0

      for (let c = 0; c <= columnCount - colSpan; c++) {
        if (areaIsFree(occupied, pinnedRow, c, rowSpan, colSpan)) {
          col = c

          break
        }
      }

      row = pinnedRow
    } else {
      // Auto (or pinned col): scan row-major for the first fitting rect. New
      // rows are always empty, so the scan terminates.
      row = 0
      col = pinnedCol ?? 0

      for (let r = 0; ; r++) {
        const found =
          pinnedCol !== null
            ? areaIsFree(occupied, r, pinnedCol, rowSpan, colSpan)
              ? pinnedCol
              : null
            : (() => {
                for (let c = 0; c <= columnCount - colSpan; c++) {
                  if (areaIsFree(occupied, r, c, rowSpan, colSpan)) {
                    return c
                  }
                }

                return null
              })()

        if (found !== null) {
          row = r
          col = found

          break
        }
      }
    }

    occupyArea(occupied, row, col, rowSpan, colSpan)

    return { col, colSpan, id: item.id, row, rowSpan }
  })
}

const trackOffsets = (sizes: number[], gap: number) => {
  const offsets: number[] = []
  let cursor = 0

  for (const size of sizes) {
    offsets.push(cursor)
    cursor += size + gap
  }

  return offsets
}

const spanSize = (sizes: number[], start: number, span: number, gap: number) => {
  const end = Math.min(sizes.length, start + span)

  return sizes.slice(start, end).reduce((acc, s) => acc + s, 0) + gap * Math.max(0, end - start - 1)
}

export function layoutGridAreas({
  columns,
  gap = 1,
  height,
  items,
  rowGap = 0,
  rows,
  width
}: GridAreasOptions): GridAreasLayout {
  const safeGap = Math.max(0, toInt(gap, 1))
  const safeRowGap = Math.max(0, toInt(rowGap, 0))
  const safeWidth = Math.max(1, toInt(width, 1))
  const safeHeight = Math.max(1, toInt(height, 1))
  const columnTracks = normalizeTracks(columns, 1)
  const columnCount = columnTracks.length

  const placed = placeGridItems(items, columnCount)
  const placedRowCount = placed.reduce((acc, area) => Math.max(acc, area.row + area.rowSpan), 0)
  const explicitRowTracks = rows === undefined ? [] : normalizeTracks(rows, 1)
  const rowCount = Math.max(1, explicitRowTracks.length, placedRowCount)

  const rowTracks: GridTrackSize[] = Array.from({ length: rowCount }, (_, idx) => explicitRowTracks[idx] ?? { fr: 1 })

  const columnSizes = resolveGridTracks(safeWidth, safeGap, columnTracks)
  const rowSizes = resolveGridTracks(safeHeight, safeRowGap, rowTracks)
  const columnStarts = trackOffsets(columnSizes, safeGap)
  const rowStarts = trackOffsets(rowSizes, safeRowGap)

  const cells = placed.map(area => {
    const row = Math.min(area.row, rowCount - 1)

    return {
      col: area.col,
      colSpan: area.colSpan,
      height: spanSize(rowSizes, row, area.rowSpan, safeRowGap),
      id: area.id,
      row,
      rowSpan: area.rowSpan,
      width: spanSize(columnSizes, area.col, area.colSpan, safeGap),
      x: columnStarts[area.col] ?? 0,
      y: rowStarts[row] ?? 0
    }
  })

  return {
    cells,
    columnCount,
    columnSizes,
    height: safeHeight,
    rowCount,
    rowSizes,
    width: safeWidth
  }
}
