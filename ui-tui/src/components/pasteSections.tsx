import { Box, Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { memo, useEffect, useState } from 'react'

import { $pasteSectionsOpen } from '../app/pasteStore.js'
import { useT } from '../i18n/useT.js'
import type { Theme } from '../theme.js'
import type { PasteSection } from '../types.js'

import { Accordion } from './accordion.js'

/**
 * The collapsed pastes of a sent user message, one accordion each: the chip
 * stays the message's text, the full paste opens on a click (or Ctrl+P for
 * every section at once) and shows exactly what the model received.
 */
export const PasteSections = memo(function PasteSections({
  marginLeft,
  pastes,
  t,
  width
}: {
  marginLeft: number
  pastes?: PasteSection[]
  t: Theme
  width: number
}) {
  const T = useT().chatBits.messageLine
  const allOpen = useStore($pasteSectionsOpen)

  if (!pastes?.length) {
    return null
  }

  return (
    <Box flexDirection="column" marginLeft={marginLeft} width={width}>
      {pastes.map((paste, i) => (
        <PasteSectionRow allOpen={allOpen} key={i} paste={paste} t={t} title={T.pasteSection(i + 1)} />
      ))}
    </Box>
  )
})

function PasteSectionRow({
  allOpen,
  paste,
  t,
  title
}: {
  allOpen: boolean
  paste: PasteSection
  t: Theme
  title: string
}) {
  const T = useT().chatBits.messageLine
  const [open, setOpen] = useState(allOpen)

  // Ctrl+P re-syncs every section; a click afterwards flips just this one.
  useEffect(() => setOpen(allOpen), [allOpen])

  const lines = paste.text.split('\n').length

  return (
    <Accordion
      onToggle={() => setOpen(v => !v)}
      open={open}
      suffix={open ? T.pasteSectionHideHint(lines) : T.pasteSectionShowHint(lines)}
      t={t}
      title={title}
    >
      <Box
        borderBottom={false}
        borderColor={t.color.border}
        borderLeft
        borderRight={false}
        borderStyle="single"
        borderTop={false}
        paddingLeft={1}
      >
        <Text color={t.color.text}>{paste.text}</Text>
      </Box>
    </Accordion>
  )
}

/** The composer's Ctrl+P hint: shown while a paste chip is in the draft, or expanded in it. */
export function PasteHint({ hint, indent, t }: { hint: null | 'collapse' | 'expand'; indent: number; t: Theme }) {
  const T = useT().composer

  if (!hint) {
    return null
  }

  return (
    <Box marginLeft={indent}>
      <Text color={t.color.muted}>{hint === 'collapse' ? T.pasteCollapseHint : T.pasteExpandHint}</Text>
    </Box>
  )
}
