import { Box, Text } from '@hermes/ink'
import { type ReactNode, useState } from 'react'

import type { Theme } from '../theme.js'

export type ChevronTone = 'dim' | 'error' | 'label' | 'muted' | 'warn'

const TONE_COLOR: Record<ChevronTone, (t: Theme) => string> = {
  dim: t => t.color.muted,
  error: t => t.color.error,
  label: t => t.color.label,
  muted: t => t.color.muted,
  warn: t => t.color.warn
}

/**
 * The one ▸/▾ section header. `onClick` receives `deep` (shift/ctrl held) so
 * callers like the tool trail can expand every section at once; `bold`
 * renders the title as an accent heading (banner sections).
 */
export function Chevron({
  bold,
  count,
  onClick,
  open,
  suffix,
  t,
  title,
  tone = 'dim'
}: {
  bold?: boolean
  count?: number
  onClick: (deep?: boolean) => void
  open: boolean
  suffix?: string
  t: Theme
  title: string
  tone?: ChevronTone
}) {
  return (
    <Box onClick={(e: any) => onClick(!!e?.shiftKey || !!e?.ctrlKey)}>
      <Text color={TONE_COLOR[tone](t)} dim={tone === 'dim'}>
        <Text color={t.color.accent}>{open ? '▾ ' : '▸ '}</Text>
        {bold ? (
          <Text bold color={t.color.accent}>
            {title}
          </Text>
        ) : (
          title
        )}
        {typeof count === 'number' ? ` (${count})` : ''}
        {suffix ? (
          <Text color={t.color.statusFg} dim>
            {'  '}
            {suffix}
          </Text>
        ) : null}
      </Text>
    </Box>
  )
}

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
      <Chevron bold count={count} onClick={toggle} open={isOpen} suffix={suffix} t={t} title={title} tone="muted" />

      {isOpen ? children : null}
    </Box>
  )
}
