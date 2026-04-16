import { Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'

import { $uiState } from '../app/uiStore.js'
import type { ThemeColors } from '../theme.js'

export type ThemeColor = keyof ThemeColors

export interface FgProps {
  bold?: boolean
  c?: ThemeColor
  children?: ReactNode
  dim?: boolean
  italic?: boolean
  literal?: string
  strikethrough?: boolean
  underline?: boolean
  wrap?: 'end' | 'middle' | 'truncate' | 'truncate-end' | 'truncate-middle' | 'truncate-start' | 'wrap' | 'wrap-trim'
}

/**
 * Theme-aware text. `literal` wins; otherwise `c` is a palette key.
 *
 *   <Fg c="amber">hi</Fg>        // amber
 *   <Fg c="dim" dim>…</Fg>       // dim cornsilk
 *   <Fg literal="#ff00ff">x</Fg> // raw hex
 */
export function Fg({ bold, c, children, dim, italic, literal, strikethrough, underline, wrap }: FgProps) {
  const { theme } = useStore($uiState)

  return (
    <Text
      bold={bold}
      color={literal ?? (c && theme.color[c])}
      dimColor={dim}
      italic={italic}
      strikethrough={strikethrough}
      underline={underline}
      wrap={wrap}
    >
      {children}
    </Text>
  )
}
