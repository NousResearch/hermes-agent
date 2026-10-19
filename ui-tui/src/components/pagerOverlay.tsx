import { Box, Text, useStdout } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { memo } from 'react'

import { $overlayState } from '../app/overlayStore.js'
import { $uiTheme } from '../app/uiStore.js'
import { pagerPageSize, pagerPanelWidth } from '../lib/pager.js'

import { OverlayHint } from './overlayControls.js'

/**
 * `/history` and `/logs` pager. Rendered as an absolute, full-height panel
 * instead of a floating box anchored above the composer: the box used to size
 * itself from its content, so a long page pushed it past the terminal edge and
 * the rows that did not fit were simply never shown — while the pager still
 * advanced the offset by the full page, skipping them.
 *
 * `rows` comes from the terminal, so the panel height and the page slice are
 * derived from the same number (`lib/pager.ts`).
 */
export const PagerOverlay = memo(function PagerOverlay({ cols }: { cols: number }) {
  const pager = useStore($overlayState).pager
  const theme = useStore($uiTheme)
  const { stdout } = useStdout()

  require('fs').appendFileSync(
    '/tmp/pager-debug.log',
    `render ${pager ? pager.lines.length : 'null'} rows=${stdout?.rows}\n`
  )

  if (!pager) {
    return null
  }

  const rows = Math.max(6, stdout?.rows ?? 24)
  const pageSize = pagerPageSize(rows)
  const panelWidth = pagerPanelWidth(cols)
  const visible = pager.lines.slice(pager.offset, pager.offset + pageSize)

  return (
    <Box flexDirection="column" flexGrow={1} width={cols}>
      <Box
        borderColor={theme.color.border}
        borderStyle="double"
        flexDirection="column"
        paddingX={1}
        paddingY={1}
        width={panelWidth}
      >
        {pager.title ? (
          <Box justifyContent="center" marginBottom={1}>
            <Text bold color={theme.color.primary}>
              {pager.title}
            </Text>
          </Box>
        ) : null}

        {/* One Text for the whole page: rows arrive pre-wrapped by `page()`, and a
            single node keeps the panel height exactly `pageSize` rows (per-line
            nodes let Yoga give the block less height than its content). */}
        <Text wrap="truncate-end">{visible.join('\n')}</Text>

        <Box marginTop={1}>
          <OverlayHint t={theme}>
            {pager.offset + pageSize < pager.lines.length
              ? `↑↓/jk line · Enter/Space/PgDn page · b/PgUp back · g/G top/bottom · Esc/q close (${Math.min(pager.offset + pageSize, pager.lines.length)}/${pager.lines.length})`
              : `end · ↑↓/jk · b/PgUp back · g top · Esc/q close (${pager.lines.length} lines)`}
          </OverlayHint>
        </Box>
      </Box>
    </Box>
  )
})
