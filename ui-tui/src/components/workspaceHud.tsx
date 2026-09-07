import { Box, Text } from '@hermes/ink'
import { Fragment } from 'react'

import type { WorkspaceHudPart } from '../domain/workspaceHud.js'
import type { Theme } from '../theme.js'

interface WorkspaceHudProps {
  cols: number
  parts: readonly WorkspaceHudPart[]
  t: Theme
}

const toneColor = (part: WorkspaceHudPart, t: Theme) => {
  if (part.tone === 'accent') {
    return t.color.accent
  }

  if (part.tone === 'error') {
    return t.color.error
  }

  if (part.tone === 'good') {
    return t.color.statusGood
  }

  if (part.tone === 'warn') {
    return t.color.statusWarn
  }

  return t.color.muted
}

/** A display-only, already-fitted workspace row for the status footer. */
export function WorkspaceHud({ cols, parts, t }: WorkspaceHudProps) {
  if (!parts.length) {
    return null
  }

  return (
    <Box flexDirection="row" flexShrink={0} height={1} overflow="hidden" width={Math.max(1, cols)}>
      <Text color={t.color.border} wrap="truncate-end">
        {cols >= 2 ? '└ ' : ''}
        {parts.map((part, index) => (
          <Fragment key={part.field}>
            {index > 0 ? <Text color={t.color.border}>{' · '}</Text> : null}
            <Text color={toneColor(part, t)}>{part.text}</Text>
          </Fragment>
        ))}
      </Text>
    </Box>
  )
}
