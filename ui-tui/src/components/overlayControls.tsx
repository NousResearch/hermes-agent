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

/**
 * Clamp a selection index into `[0, length - 1]`, returning 0 for an empty
 * list. Shared by the type-to-filter overlays to keep a cursor in bounds when
 * the filtered list shrinks beneath it.
 */
export const clampIndex = (index: number, length: number): number =>
  length <= 0 ? 0 : Math.max(0, Math.min(index, length - 1))

interface OverlayHintProps {
  children: string
  t: Theme
}

interface OverlayKeysOptions {
  disabled?: boolean
  onBack?: () => void
  onClose: () => void
}
