export const DEV_CONTEXT_RAIL_WIDTH = 40
export const DEV_CONTEXT_MIN_TERMINAL_COLS = 112
export const DEV_CONTEXT_MIN_TRANSCRIPT_COLS = 48

/**
 * The developer rail is deliberately a layout participant, not an overlay.
 * Keep it out of cramped terminals and leave the transcript a readable body
 * width when user-provided rails are active too.
 */
export const devContextRailVisible = (
  enabled: boolean,
  columns: number,
  ambientRailColumns = 0
): boolean => {
  const totalColumns = Math.floor(columns)
  const otherRailColumns = Math.max(0, Math.floor(ambientRailColumns))

  return (
    enabled &&
    totalColumns >= DEV_CONTEXT_MIN_TERMINAL_COLS &&
    totalColumns - otherRailColumns - DEV_CONTEXT_RAIL_WIDTH >= DEV_CONTEXT_MIN_TRANSCRIPT_COLS
  )
}

export const devContextRailWidth = (enabled: boolean, columns: number, ambientRailColumns = 0): number =>
  devContextRailVisible(enabled, columns, ambientRailColumns) ? DEV_CONTEXT_RAIL_WIDTH : 0
