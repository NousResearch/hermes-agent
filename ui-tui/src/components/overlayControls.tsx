import { Text, useInput } from '@hermes/ink'

import type { Theme } from '../theme.js'

export function useOverlayKeys({ disabled = false, onBack, onClose }: OverlayKeysOptions) {
  useInput((ch, key) => {
    if (disabled) {
      return
    }

    if (ch === 'q') {
      return onClose()
    }

    if (key.escape) {
      return onBack ? onBack() : onClose()
    }
  })
}

export function OverlayHint({ children, t }: OverlayHintProps) {
  return (
    <Text color={t.color.muted} wrap="truncate-end">
      {children}
    </Text>
  )
}

export const windowOffset = (count: number, selected: number, visible: number) =>
  Math.max(0, Math.min(selected - Math.floor(visible / 2), count - visible))

export function windowItems<T>(items: T[], selected: number, visible: number) {
  const offset = windowOffset(items.length, selected, visible)

  return {
    items: items.slice(offset, offset + visible),
    offset
  }
}

interface OverlayHintProps {
  children: string
  t: Theme
}

interface OverlayKeysOptions {
  disabled?: boolean
  onBack?: () => void
  onClose: () => void
}
