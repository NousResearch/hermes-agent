import { Box } from '@hermes/ink'
import type { ReactNode } from 'react'

export const COMPOSER_RAIL_WIDTH = 1

export const composerRailContentColumns = (totalColumns: number): number =>
  Math.max(1, Math.floor(totalColumns) - COMPOSER_RAIL_WIDTH)

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
