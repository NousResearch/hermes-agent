import { Box, NoSelect, Text } from '@hermes/ink'
import type { ReactNode } from 'react'
import type { Theme } from '../theme.js'

export type TreeBranch = 'mid' | 'last'
export type TreeRails = readonly boolean[]

export function nextTreeRails(rails: TreeRails, branch: TreeBranch): TreeRails {
  return [...rails, branch === 'mid']
}

export function treeLead(rails: TreeRails, branch: TreeBranch): string {
  return `${rails.map(on => (on ? '│ ' : '  ')).join('')}${branch === 'mid' ? '├─ ' : '└─ '}`
}

export function TreeRow({
  branch,
  children,
  rails = [],
  stemColor,
  stemDim = true,
  t
}: {
  branch: TreeBranch
  children: ReactNode
  rails?: TreeRails
  stemColor?: string
  stemDim?: boolean
  t: Theme
}) {
  const lead = treeLead(rails, branch)

  return (
    <Box>
      <NoSelect flexShrink={0} fromLeftEdge width={lead.length}>
        <Text color={stemColor ?? t.color.dim} dim={stemDim}>
          {lead}
        </Text>
      </NoSelect>
      <Box flexDirection="column" flexGrow={1}>
        {children}
      </Box>
    </Box>
  )
}

export function TreeTextRow({
  branch,
  color,
  content,
  dimColor,
  rails = [],
  t,
  wrap = 'wrap-trim'
}: {
  branch: TreeBranch
  color: string
  content: ReactNode
  dimColor?: boolean
  rails?: TreeRails
  t: Theme
  wrap?: 'truncate-end' | 'wrap' | 'wrap-trim'
}) {
  const text = dimColor ? (
    <Text color={color} dim wrap={wrap}>
      {content}
    </Text>
  ) : (
    <Text color={color} wrap={wrap}>
      {content}
    </Text>
  )

  return (
    <TreeRow branch={branch} rails={rails} t={t}>
      {text}
    </TreeRow>
  )
}
