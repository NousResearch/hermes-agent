import { Box } from '@hermes/ink'
import type { ReactNode } from 'react'

import { stableComposerColumns } from '../lib/inputMetrics.js'

/**
 * The composer pane is flexShrink=0, so every row it gains is a row taken
 * straight out of the transcript window. The boundary is therefore a left
 * border only: it costs one column and zero rows, and Ink paints it inside the
 * node's own rect rather than growing it.
 */
export const COMPOSER_RAIL_WIDTH = 1

/**
 * Deduct the rail from the input's own budget rather than shrinking the total
 * handed to `stableComposerColumns`. That function reserves the transcript
 * scrollbar gutter once `total - prompt >= 24`, so shrinking the total would
 * move that threshold by a column and let the input silently regain a column
 * inside the boundary band. Deducting afterwards keeps rewrap behaviour
 * identical to base and costs exactly one column everywhere.
 */
export function composerInputColumns(totalCols: number, promptWidth: number, termuxMode = false): number {
  return Math.max(1, stableComposerColumns(totalCols, promptWidth, termuxMode) - COMPOSER_RAIL_WIDTH)
}

export function ComposerRail({
  children,
  color,
  width
}: {
  children: ReactNode
  color: string
  width: number
}) {
  return (
    <Box
      borderBottom={false}
      borderColor={color}
      borderLeft
      borderRight={false}
      borderStyle="single"
      borderTop={false}
      flexDirection="column"
      width={Math.max(1, width)}
    >
      {children}
    </Box>
  )
}
