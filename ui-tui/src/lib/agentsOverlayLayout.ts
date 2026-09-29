export type AgentsOverlayMode = 'compact' | 'full'

export interface AgentsOverlayGeometry {
  canCompact: boolean
  mode: AgentsOverlayMode
  rows: number
}

const MIN_COMPACT_ROWS = 28
const MIN_COMPACT_COLS = 64
const COMPACT_MIN_HEIGHT = 12
const COMPACT_MAX_HEIGHT = 22
const COMPACT_RATIO = 0.42
const MIN_CONTEXT_ROWS = 10

export const agentsOverlayGeometry = (
  terminalRows: number,
  terminalCols: number,
  expanded = false
): AgentsOverlayGeometry => {
  const rows = Math.max(1, Math.floor(terminalRows))
  const cols = Math.max(1, Math.floor(terminalCols))
  const canCompact = rows >= MIN_COMPACT_ROWS && cols >= MIN_COMPACT_COLS

  if (expanded || !canCompact) {
    return { canCompact, mode: 'full', rows }
  }

  const maxCompact = Math.max(COMPACT_MIN_HEIGHT, Math.min(COMPACT_MAX_HEIGHT, rows - MIN_CONTEXT_ROWS))
  const target = Math.round(rows * COMPACT_RATIO)
  const compactRows = Math.max(COMPACT_MIN_HEIGHT, Math.min(maxCompact, target))

  return { canCompact, mode: 'compact', rows: compactRows }
}
