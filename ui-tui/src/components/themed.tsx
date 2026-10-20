import { Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'

import { $uiState } from '../app/uiStore.js'
import type { ThemeColors } from '../theme.js'

export function Fg({ bold, c, children, dim, italic, literal, strikethrough, underline, wrap }: FgProps) {
  const { theme } = useStore($uiState)

  return (
    <Text color={literal ?? (c && theme.color[c])} dimColor={dim} {...{ bold, italic, strikethrough, underline, wrap }}>
      {children}
    </Text>
  )
}

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
