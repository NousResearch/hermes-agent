import { Box, Text } from '@hermes/ink'
import { type ReactNode, useState } from 'react'

import type { Theme } from '../theme.js'

/**
 * THE expand/collapse primitive — the session panel's tool/skill sections
 * and widget-app accordions are the same component. Click the header to
 * toggle (mouse works even in ambient widgets, which receive no keys);
 * modal apps may instead drive `open` from reducer state (controlled).
 * Uncontrolled by default: pass `defaultOpen` and forget it.
 */
export function Accordion({
  children,
  count,
  defaultOpen = false,
  onToggle,
  open,
  suffix,
  t,
  title
}: {
  children: ReactNode
  count?: number
  defaultOpen?: boolean
  /** Controlled open state; omit for internal (click-toggled) state. */
  open?: boolean
  onToggle?: () => void
  suffix?: string
  t: Theme
  title: string
}) {
  const [uncontrolled, setUncontrolled] = useState(defaultOpen)
  const isOpen = open ?? uncontrolled

  const toggle = () => {
    onToggle?.()

    if (open === undefined) {
      setUncontrolled(v => !v)
    }
  }

  return (
    <Box flexDirection="column">
      <Box onClick={toggle}>
        <Text color={t.color.accent}>{isOpen ? '▾ ' : '▸ '}</Text>
        <Text bold color={t.color.accent}>
          {title}
        </Text>
        {typeof count === 'number' ? <Text color={t.color.muted}> ({count})</Text> : null}
        {suffix ? <Text color={t.color.muted}> {suffix}</Text> : null}
      </Box>

      {isOpen ? children : null}
    </Box>
  )
}
