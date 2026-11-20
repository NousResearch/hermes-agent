import { Box } from '@hermes/ink'
import { Fragment, memo, type ReactNode, useMemo } from 'react'

import {
  type GridAreaCell,
  type GridAreaItem,
  type GridTrackSize,
  layoutGridAreas,
  layoutWidgetGrid,
  type WidgetGridCell,
  type WidgetGridItem,
  widgetGridSpanWidth
} from '../lib/widgetGrid.js'

export interface WidgetGridRenderContext {
  cell: WidgetGridCell
  width: number
}

type WidgetGridChildren = ((ctx: WidgetGridRenderContext) => ReactNode) | ReactNode

/**
 * A grid item with optional content. Use `children` for static or stateful
 * React subtrees (including a nested `WidgetGrid`) and `render` for a width-
 * aware factory; if both are provided, `render` wins.
 */
export interface WidgetGridWidget extends WidgetGridItem {
  children?: WidgetGridChildren
  render?: (width: number, cell: WidgetGridCell) => ReactNode
}

/**
 * `WidgetGrid` lays out children into rows/cols using the same primitives as
 * CSS grid: explicit `columns` count or a width-derived auto count, per-item
 * `colStart` / `colSpan`, and uniform `gap` / `rowGap`. Cells clip their
 * contents (`overflow: hidden`) so child overflow can never bleed into the
 * neighbouring cell or break the parent border.
 */
interface WidgetGridProps {
  /** Column count (equal shares) or grid-template-style track list. */
  columns?: GridTrackSize[] | number
  cols: number
  depth?: number
  gap?: number
  maxColumns?: number
  minColumnWidth?: number
  paddingX?: number
  paddingY?: number
  rowGap?: number
  widgets: WidgetGridWidget[]
}

const toInt = (value: number, fallback: number) => (Number.isFinite(value) ? Math.trunc(value) : fallback)

const columnCountHint = (columns: GridTrackSize[] | number | undefined) =>
  Array.isArray(columns) ? columns.length : (columns ?? 0)

const inferredGap = (cols: number, columns: GridTrackSize[] | number | undefined, depth: number) => {
  const count = columnCountHint(columns)

  if (cols < 36 || count >= 8) {
    return 0
  }

  if (depth > 0 || cols < 72 || count >= 4) {
    return 1
  }

  return 2
}

const inferredPaddingX = (cols: number, depth: number) => {
  if (depth <= 0 || cols < 24) {
    return 0
  }

  return cols >= 56 ? 2 : 1
}

const inferredRowGap = (depth: number) => (depth > 0 ? 0 : 1)

export const WidgetGrid = memo(function WidgetGrid({
  columns,
  cols,
  depth = 0,
  gap,
  maxColumns = 2,
  minColumnWidth = 46,
  paddingX,
  paddingY,
  rowGap,
  widgets
}: WidgetGridProps) {
  const safeCols = Math.max(1, toInt(cols, 1))
  const safePaddingX = Math.max(0, toInt(paddingX ?? inferredPaddingX(safeCols, depth), 0))
  const safePaddingY = Math.max(0, toInt(paddingY ?? 0, 0))
  const innerCols = Math.max(1, safeCols - safePaddingX * 2)
  const safeGap = Math.max(0, toInt(gap ?? inferredGap(innerCols, columns, depth), 0))
  const safeRowGap = Math.max(0, toInt(rowGap ?? inferredRowGap(depth), 0))

  const layout = useMemo(
    () =>
      layoutWidgetGrid({
        columns,
        gap: safeGap,
        items: widgets.map(({ colSpan, colStart, id, span }) => ({ colSpan, colStart, id, span })),
        maxColumns,
        minColumnWidth,
        width: innerCols
      }),
    [columns, innerCols, maxColumns, minColumnWidth, safeGap, widgets]
  )

  const widgetById = useMemo(() => new Map(widgets.map(widget => [widget.id, widget])), [widgets])

  if (!layout.rows.length) {
    return null
  }

  return (
    <Box flexDirection="column" paddingX={safePaddingX} paddingY={safePaddingY} width={safeCols}>
      {layout.rows.map((row, rowIdx) => (
        <Box flexDirection="column" key={`row-${rowIdx}`}>
          <Box flexDirection="row">
            <WidgetRow cells={row} columns={layout.columns} gap={safeGap} widgetById={widgetById} />
          </Box>

          {safeRowGap > 0 && rowIdx < layout.rows.length - 1 ? <Box height={safeRowGap} /> : null}
        </Box>
      ))}
    </Box>
  )
})

const WidgetRow = memo(function WidgetRow({
  cells,
  columns,
  gap,
  widgetById
}: {
  cells: WidgetGridCell[]
  columns: number[]
  gap: number
  widgetById: Map<string, WidgetGridWidget>
}) {
  return (
    <>
      {cells.map((cell, idx) => {
        const cursor = idx === 0 ? 0 : cells[idx - 1]!.col + cells[idx - 1]!.span

        const spacerWidth =
          cell.col === 0
            ? 0
            : cursor === 0
              ? widgetGridSpanWidth(columns, 0, cell.col, gap) + gap
              : gap + (cell.col > cursor ? widgetGridSpanWidth(columns, cursor, cell.col - cursor, gap) + gap : 0)

        return (
          <Fragment key={cell.id}>
            {spacerWidth > 0 ? <Box flexShrink={0} width={spacerWidth} /> : null}
            <WidgetCell cell={cell} widget={widgetById.get(cell.id)} />
          </Fragment>
        )
      })}
    </>
  )
})

const WidgetCell = memo(function WidgetCell({ cell, widget }: { cell: WidgetGridCell; widget?: WidgetGridWidget }) {
  const node =
    widget?.render?.(cell.width, cell) ??
    (typeof widget?.children === 'function' ? widget.children({ cell, width: cell.width }) : widget?.children) ??
    null

  return (
    <Box flexShrink={0} overflow="hidden" width={cell.width}>
      {node}
    </Box>
  )
})

// ── GridAreas: the two-axis workspace mode ──────────────────────────────────

type GridAreaChildren = ((cell: GridAreaCell) => ReactNode) | ReactNode

/**
 * An area widget: placement (`col`/`row`/`colSpan`/`rowSpan`) plus content.
 * `render` receives the solved cell (with `width`/`height` in terminal
 * cells); `children` accepts a static subtree or a factory. `render` wins
 * when both are given, mirroring `WidgetGridWidget`.
 */
export interface GridAreaWidget extends GridAreaItem {
  children?: GridAreaChildren
  render?: (cell: GridAreaCell) => ReactNode
}

interface GridAreasProps {
  /** Column tracks: a count (equal shares) or explicit fixed/`fr` tracks. */
  columns: GridTrackSize[] | number
  gap?: number
  height: number
  rowGap?: number
  /** Row tracks. Omitted: every row is an equal `fr` share of `height`. */
  rows?: GridTrackSize[] | number
  widgets: GridAreaWidget[]
  width: number
}

/**
 * `GridAreas` renders widgets into a fully two-dimensional grid: explicit
 * column AND row tracks (fixed cells or weighted `fr` shares), `colSpan` /
 * `rowSpan`, `col` / `row` pins, and dense auto-placement. Unlike the flowing
 * `WidgetGrid` (which stacks rows and cannot express `rowSpan`), every cell
 * here is solved to a rect and absolutely positioned inside a fixed-size box,
 * so a cell can span rows the same way a merged FancyZones cell does on the
 * desktop app. Requires a known `height`.
 */
export const GridAreas = memo(function GridAreas({
  columns,
  gap = 1,
  height,
  rowGap = 0,
  rows,
  widgets,
  width
}: GridAreasProps) {
  const layout = useMemo(
    () =>
      layoutGridAreas({
        columns,
        gap,
        height,
        items: widgets.map(({ col, colSpan, id, row, rowSpan }) => ({ col, colSpan, id, row, rowSpan })),
        rowGap,
        rows,
        width
      }),
    [columns, gap, height, rowGap, rows, widgets, width]
  )

  const widgetById = useMemo(() => new Map(widgets.map(widget => [widget.id, widget])), [widgets])

  if (!layout.cells.length) {
    return null
  }

  return (
    <Box flexDirection="column" height={layout.height} overflow="hidden" width={layout.width}>
      {layout.cells.map(cell => (
        <AreaCell cell={cell} key={cell.id} widget={widgetById.get(cell.id)} />
      ))}
    </Box>
  )
})

const AreaCell = memo(function AreaCell({ cell, widget }: { cell: GridAreaCell; widget?: GridAreaWidget }) {
  const node =
    widget?.render?.(cell) ??
    (typeof widget?.children === 'function' ? widget.children(cell) : widget?.children) ??
    null

  return (
    <Box height={cell.height} left={cell.x} overflow="hidden" position="absolute" top={cell.y} width={cell.width}>
      {node}
    </Box>
  )
})
