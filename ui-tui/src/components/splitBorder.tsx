import { Box, Text } from '@hermes/ink'
import type { ReactNode } from 'react'

import type { Theme } from '../theme.js'

/**
 * SplitBorder — a left-only vertical border (┃) inspired by OpenCode's
 * `border={["left"]}` with `SplitBorder` custom chars.  Renders a single
 * vertical line on the left edge of a card using the given color, leaving
 * the top/bottom/right edges free (unlike the standard `borderStyle="round"`
 * which draws a full box).
 *
 * Used for:
 *  - User messages: `accent` color (blue) — visually highlights user input
 *  - Block tools / thinking: `warning` or `chatBorder` color
 *  - Assistant message content: optional, subtle `divider` color
 *
 * @example
 *   <SplitBorder color={t.color.accent} indent={2}>
 *     <Text>user message content</Text>
 *   </SplitBorder>
 */
export function SplitBorder({
  children,
  color,
  indent = 0,
  width
}: {
  children: ReactNode
  color: string
  indent?: number
  width?: number
}) {
  return (
    <Box flexDirection="row">
      {indent > 0 ? <Box width={indent} /> : null}
      <Box flexDirection="row" flexGrow={1}>
        <Box flexShrink={0} width={1}>
          <Text color={color}>┃</Text>
        </Box>
        <Box
          flexDirection="column"
          flexGrow={1}
          paddingX={1}
          width={width}
        >
          {children}
        </Box>
      </Box>
    </Box>
  )
}
