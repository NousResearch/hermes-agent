import { Text, useInput } from '@hermes/ink'

import type { Theme } from '../theme.js'

type TextWrap = 'end' | 'middle' | 'truncate' | 'truncate-end' | 'truncate-middle' | 'wrap' | 'wrap-char' | 'wrap-trim'

export function useOverlayKeys({ disabled = false, onBack, onClose }: OverlayKeysOptions) {
  useInput((ch, key) => {
    if (disabled) {
      return
    }

    if (ch.toLowerCase() === 'q') {
      return onClose()
    }

    if (key.escape) {
      return onBack ? onBack() : onClose()
    }
  })
}

export function OverlayControls({ children, t, wrap = 'truncate-end' }: OverlayControlsProps) {
  return (
    <Text color={t.color.dim} wrap={wrap}>
      {children}
    </Text>
  )
}

interface OverlayControlsProps {
  children: string
  t: Theme
  wrap?: TextWrap
}

interface OverlayKeysOptions {
  disabled?: boolean
  onBack?: () => void
  onClose: () => void
}
