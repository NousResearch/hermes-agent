export const DEV_CONTEXT_RAIL_WIDTH = 40
export const DEV_CONTEXT_MIN_TERMINAL_COLS = 112
export const DEV_CONTEXT_MIN_TRANSCRIPT_COLS = 48

export type DevContextPlacement = 'bottom' | 'hidden' | 'side'

/**
 * The context surface should exist when there is something actionable to
 * show, but its side-rail width must still yield to a readable transcript.
 */
export const devContextHasActivity = (
  todoCount: number,
  activeAgentCount: number,
  backgroundTaskCount: number,
  toolCount: number,
  queuedCount: number,
  staticInput = false,
  attention = false
): boolean =>
  todoCount > 0 ||
  activeAgentCount > 0 ||
  backgroundTaskCount > 0 ||
  toolCount > 0 ||
  queuedCount > 0 ||
  staticInput ||
  attention

/**
 * The developer rail is deliberately a layout participant, not an overlay.
 * Keep it out of cramped terminals and leave the transcript a readable body
 * width when user-provided rails are active too.
 */
export const devContextRailVisible = (
  enabled: boolean,
  columns: number,
  ambientRailColumns = 0,
  hasActivity = true
): boolean => {
  const totalColumns = Math.floor(columns)
  const otherRailColumns = Math.max(0, Math.floor(ambientRailColumns))

  return (
    enabled &&
    hasActivity &&
    totalColumns >= DEV_CONTEXT_MIN_TERMINAL_COLS &&
    totalColumns - otherRailColumns - DEV_CONTEXT_RAIL_WIDTH >= DEV_CONTEXT_MIN_TRANSCRIPT_COLS
  )
}

export const devContextPlacement = (
  enabled: boolean,
  columns: number,
  ambientRailColumns = 0,
  hasActivity = true
): DevContextPlacement => {
  if (!enabled || !hasActivity) {
    return 'hidden'
  }

  return devContextRailVisible(enabled, columns, ambientRailColumns, hasActivity) ? 'side' : 'bottom'
}

export const devContextRailWidth = (
  enabled: boolean,
  columns: number,
  ambientRailColumns = 0,
  hasActivity = true
): number => (devContextRailVisible(enabled, columns, ambientRailColumns, hasActivity) ? DEV_CONTEXT_RAIL_WIDTH : 0)
