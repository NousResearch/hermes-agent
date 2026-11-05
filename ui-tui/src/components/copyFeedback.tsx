import { Box, Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { $copyFeedback, clearCopyFeedback } from '../app/copyFeedbackStore.js'
import { $uiSessionId } from '../app/uiStore.js'
import { useT } from '../i18n/useT.js'

/** Shares the existing spacer above the composer, so copying never moves the selected transcript. */
export function CopyFeedback() {
  const delivery = useStore($copyFeedback)
  const sid = useStore($uiSessionId)
  const T = useT()

  useEffect(() => {
    clearCopyFeedback()

    return clearCopyFeedback
  }, [sid])

  if (!delivery) {
    return null
  }

  return (
    <Box flexShrink={1} justifyContent="flex-end" minWidth={0}>
      <Text color="ansi:blueBright" wrap="truncate-end">
        {delivery === 'terminal' ? T.composer.copySentToTerminal : T.slashCmd.core.copy.copied}
      </Text>
    </Box>
  )
}
