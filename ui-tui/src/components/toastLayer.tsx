import { Box, Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { memo } from 'react'

import { $toastState, dismissToast, type ToastTone } from '../app/toastStore.js'
import { $uiState } from '../app/uiStore.js'
import type { Theme } from '../theme.js'

const NARROW_COLS_THRESHOLD = 30

const toneIcon = (tone: ToastTone): string => {
  switch (tone) {
    case 'success':
      return '✓ '
    case 'error':
      return '✗ '
    case 'warn':
      return '⚠ '
    case 'info':
    default:
      return 'ℹ '
  }
}

const toneColor = (tone: ToastTone, t: Theme): string => {
  switch (tone) {
    case 'success':
      return t.color.ok
    case 'error':
      return t.color.error
    case 'warn':
      return t.color.warn
    case 'info':
    default:
      return t.color.primary
  }
}

const ToastItemView = memo(function ToastItemView({
  message,
  tone,
  t
}: {
  message: string
  tone: ToastTone
  t: Theme
}) {
  return (
    <Box
      alignSelf="flex-end"
      borderColor={toneColor(tone, t)}
      borderStyle="round"
      flexDirection="row"
      marginBottom={1}
      opaque
      paddingX={1}
    >
      <Text bold color={toneColor(tone, t)}>
        {toneIcon(tone)}
      </Text>
      <Text color={t.color.text} wrap="truncate-end">
        {message}
      </Text>
    </Box>
  )
})

export const ToastLayer = memo(function ToastLayer({ cols }: { cols: number }) {
  const { toasts } = useStore($toastState)
  const ui = useStore($uiState)

  if (cols < NARROW_COLS_THRESHOLD || !toasts.length) {
    return null
  }

  return (
    <Box flexDirection="column" position="absolute" right={1} top={1} zIndex={10}>
      {toasts.map(item => (
        <Box key={item.id} onClick={() => dismissToast(item.id)}>
          <ToastItemView message={item.message} tone={item.tone} t={ui.theme} />
        </Box>
      ))}
    </Box>
  )
})
