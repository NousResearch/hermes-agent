import { Box, Text } from '@hermes/ink'

import type { StashPanelRow } from '../hooks/usePromptStash.js'
import { messages } from '../i18n/runtime.js'
import { compactPreview } from '../lib/text.js'
import type { Theme } from '../theme.js'

/** Rows the panel shows before it windows. The classic panel draws every row; the TUI's
 *  composer zone is fixed-height, so it scrolls the window with the highlight instead. */
export const STASH_WINDOW = 6

export function getStashWindow(len: number, cursor: number): { end: number; showLead: boolean; showTail: boolean; start: number } {
  const start = Math.max(0, Math.min(cursor - 1, Math.max(0, len - STASH_WINDOW)))
  const end = Math.min(len, start + STASH_WINDOW)

  return { end, showLead: start > 0, showTail: end < len, start }
}

/**
 * The stash chrome above the prompt: the `📌 N` badge while drafts are parked, and the browse
 * panel while it is open (the panel's own header replaces the badge then, as the classic CLI's
 * status bar does). One component so the composer pane — whose branch budget is capped — stays
 * a plain render.
 */
export function StashChrome({
  cols,
  cursor,
  indicator,
  open,
  rows,
  t
}: {
  cols: number
  cursor: number
  indicator: string
  open: boolean
  rows: StashPanelRow[]
  t: Theme
}) {
  if (!open && !indicator) {
    return null
  }

  return (
    <>
      {open ? null : (
        <Text color={t.color.muted} dimColor>
          {indicator}
        </Text>
      )}

      <StashPanel cols={cols} cursor={cursor} rows={open ? rows : []} t={t} />
    </>
  )
}

/**
 * Ctrl+S browse panel: the drafts parked this session, newest first.
 *
 * Same information and wording as the classic CLI's box (`_render_stash_panel`): header with
 * the count, one row per draft with its age and preview, and the key legend. Only the
 * drawing differs — the TUI's composer zone draws an unboxed list (as the queue peek does)
 * rather than the classic CLI's box frame.
 */
export function StashPanel({
  cols,
  cursor,
  rows,
  t
}: {
  cols: number
  cursor: number
  rows: StashPanelRow[]
  t: Theme
}) {
  if (!rows.length) {
    return null
  }

  const s = messages().composer.stash
  const win = getStashWindow(rows.length, cursor)
  const previewWidth = Math.max(16, cols - 24)

  return (
    <Box flexDirection="column" marginTop={1}>
      <Text color={t.color.muted} dimColor>
        {s.header(rows.length)}
      </Text>

      {win.showLead && (
        <Text color={t.color.muted} dimColor>
          {' '}
          …
        </Text>
      )}

      {rows.slice(win.start, win.end).map((row, i) => {
        const idx = win.start + i
        const active = cursor === idx

        return (
          <Text color={active ? t.color.accent : t.color.muted} dimColor key={`${idx}-${row.preview.slice(0, 16)}`}>
            {active ? '►' : ' '} [{idx + 1}] {row.age}{' '.repeat(Math.max(1, 11 - row.age.length))}
            {compactPreview(row.preview, previewWidth)}
          </Text>
        )
      })}

      {win.showTail && (
        <Text color={t.color.muted} dimColor>
          {'  '}
          {s.more(rows.length - win.end)}
        </Text>
      )}

      <Text color={t.color.muted} dimColor>
        {s.footer}
      </Text>
    </Box>
  )
}
