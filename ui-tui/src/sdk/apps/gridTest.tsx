import { FloatBox } from '../../components/appChrome.js'
import { GridTestOverlay } from '../../components/gridTestOverlay.js'
import { Overlay } from '../../components/overlay.js'
import { openWidget } from '../host.js'
import { defineWidgetApp } from '../registry.js'
import { isCtrl, type WidgetInput } from '../types.js'

import { dialogTestApp } from './dialogTest.js'
import { GRID_STREAM_COUNT, type GridTestState } from './gridTestState.js'

const MAX_SIZE = 12
const USAGE = 'usage: /grid-test [cols]x[rows]  ·  /grid-test [cols] [rows]  ·  /grid-test streams'

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value))

const clampSize = (value: number, fallback: number) =>
  Number.isFinite(value) ? clamp(Math.round(value), 1, MAX_SIZE) : fallback

/** null/number cycle: auto → 0 → 1 → … → max → auto. */
const cycleAutoNumber = (value: null | number, max: number) => (value === null ? 0 : value >= max ? null : value + 1)

const keepCursorInBounds = (grid: GridTestState): GridTestState => ({
  ...grid,
  activeCol: clamp(grid.activeCol, 0, grid.cols - 1),
  activeRow: clamp(grid.activeRow, 0, grid.rows - 1)
})

const initialState = (cols: number, rows: number, streams: boolean): GridTestState => ({
  activeCol: 0,
  activeRow: 0,
  areas: false,
  cols,
  gap: null,
  nested: false,
  paddingX: null,
  rows,
  streamFocus: 0,
  streamMain: 0,
  streams,
  zoomed: false
})

function parseSize(arg: string): null | { cols: number; rows: number } {
  const trimmed = arg.trim()

  if (!trimmed) {
    return { cols: 4, rows: 3 }
  }

  const grid = trimmed.match(/^(\d+)\s*x\s*(\d+)$/i)

  if (grid) {
    return { cols: clampSize(Number(grid[1]), 4), rows: clampSize(Number(grid[2]), 3) }
  }

  const [cols, rows, ...rest] = trimmed.split(/\s+/)

  if (rest.length || !cols || !rows || Number.isNaN(Number(cols)) || Number.isNaN(Number(rows))) {
    return null
  }

  return { cols: clampSize(Number(cols), 4), rows: clampSize(Number(rows), 3) }
}

const update = (grid: GridTestState, fn: (grid: GridTestState) => GridTestState) => keepCursorInBounds(fn(grid))

function reduceStreams(grid: GridTestState, { ch, key }: WidgetInput): GridTestState | null {
  if (key.escape || ch === 'q' || ch === 's') {
    return update(grid, g => ({ ...g, streams: false }))
  }

  if (key.return) {
    return update(grid, g => ({ ...g, streamMain: g.streamFocus }))
  }

  if (ch === 'r') {
    return initialState(4, 3, false)
  }

  if (key.leftArrow || key.upArrow || ch === 'h' || ch === 'k') {
    return update(grid, g => ({ ...g, streamFocus: (g.streamFocus + GRID_STREAM_COUNT - 1) % GRID_STREAM_COUNT }))
  }

  if (key.rightArrow || key.downArrow || ch === 'l' || ch === 'j') {
    return update(grid, g => ({ ...g, streamFocus: (g.streamFocus + 1) % GRID_STREAM_COUNT }))
  }

  return grid
}

export const gridTestApp = defineWidgetApp<GridTestState>({
  id: 'grid-test',
  help: 'open an interactive widget-grid demo overlay',
  usage: USAGE,

  init(arg) {
    const streams = arg.trim().toLowerCase() === 'streams'
    const size = streams ? { cols: 4, rows: 3 } : parseSize(arg)

    return size ? initialState(size.cols, size.rows, streams) : null
  },

  reduce(grid, input) {
    const { ch, key } = input

    if (isCtrl(key, ch, 'c')) {
      return null
    }

    // `d` opens the dialog app as a nested demo — apps launch each other via
    // the typed programmatic API; the host swaps the active app.
    if (ch === 'd') {
      openWidget(dialogTestApp, {
        body: 'Dialog overlaid on top of /grid-test.\n\nBackdrop dims the grid behind.',
        hint: 'Esc/q/Enter close',
        title: 'Overlay primitive',
        zone: 'center'
      })

      return grid
    }

    if (grid.streams) {
      return reduceStreams(grid, input)
    }

    if (grid.zoomed && (key.escape || ch === 'q')) {
      return update(grid, g => ({ ...g, zoomed: false }))
    }

    if (key.escape || ch === 'q') {
      return null
    }

    if (key.return) {
      return update(grid, g => ({ ...g, nested: true, zoomed: true }))
    }

    if (ch === 'n') {
      return update(grid, g => ({ ...g, nested: !g.nested }))
    }

    if (ch === 'a') {
      return update(grid, g => ({ ...g, areas: !g.areas, streams: false }))
    }

    if (ch === 's') {
      return update(grid, g => ({ ...g, areas: false, streams: true }))
    }

    if (ch === 'g') {
      return update(grid, g => ({ ...g, gap: cycleAutoNumber(g.gap, 3) }))
    }

    if (ch === 'p') {
      return update(grid, g => ({ ...g, paddingX: cycleAutoNumber(g.paddingX, 2) }))
    }

    if (ch === 'r') {
      return initialState(4, 3, false)
    }

    if (ch === '+' || ch === '=') {
      return update(grid, g => ({ ...g, cols: clamp(g.cols + 1, 1, MAX_SIZE) }))
    }

    if (ch === '-' || ch === '_') {
      return update(grid, g => ({ ...g, cols: clamp(g.cols - 1, 1, MAX_SIZE) }))
    }

    if (ch === ']') {
      return update(grid, g => ({ ...g, rows: clamp(g.rows + 1, 1, MAX_SIZE) }))
    }

    if (ch === '[') {
      return update(grid, g => ({ ...g, rows: clamp(g.rows - 1, 1, MAX_SIZE) }))
    }

    if (key.leftArrow || ch === 'h') {
      return update(grid, g => ({ ...g, activeCol: g.activeCol - 1 }))
    }

    if (key.rightArrow || ch === 'l') {
      return update(grid, g => ({ ...g, activeCol: g.activeCol + 1 }))
    }

    if (key.upArrow || ch === 'k') {
      return update(grid, g => ({ ...g, activeRow: g.activeRow - 1 }))
    }

    if (key.downArrow || ch === 'j') {
      return update(grid, g => ({ ...g, activeRow: g.activeRow + 1 }))
    }

    return grid
  },

  render({ cols, state, t }) {
    return (
      <Overlay zone="center">
        <FloatBox color={t.color.border}>
          <GridTestOverlay cols={Math.max(1, Math.min(cols - 6, 120))} state={state} t={t} />
        </FloatBox>
      </Overlay>
    )
  }
})
