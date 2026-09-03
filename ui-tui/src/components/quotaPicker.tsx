import { Box, Text, useStdout } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { useState } from 'react'

import type { AccountUsageInfo, QuotaDisplay } from '../app/interfaces.js'
import { $uiState, patchUiState } from '../app/uiStore.js'
import { selectQuotaWindows } from '../app/useAccountUsagePoll.js'
import type { GatewayClient } from '../gatewayClient.js'
import { rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { OverlayHint } from './overlayControls.js'
import { clampOverlayWidth, listRowStyle, useMenu } from './overlayPrimitives.js'

const MIN_WIDTH = 46
const MAX_WIDTH = 96

// The modes, in the order they read: the immediate cap first, the off switch
// last. Descriptions are the same one-liners /quota prints on the text path.
const MODES: readonly { help: string; mode: QuotaDisplay }[] = [
  { help: 'the short rolling window', mode: 'session' },
  { help: 'session, with the weekly cap last', mode: 'both' },
  { help: 'the weekly cap only', mode: 'weekly' },
  { help: 'whichever window is closest to spent', mode: 'tightest' },
  { help: 'hidden; the provider is not polled', mode: 'off' }
]

/**
 * The segment a mode would render, from the snapshot already on screen.
 *
 * Exported for the test: the whole point of the picker is that each row shows
 * what you are choosing, so the preview must match the status bar exactly —
 * one glyph, the trailing window appended bare.
 */
export const quotaModePreview = (usage: AccountUsageInfo | null, mode: QuotaDisplay): string => {
  if (mode === 'off') {
    return '(nothing)'
  }

  const windows = usage ? selectQuotaWindows(usage.windows, mode) : []

  if (!windows.length) {
    return '—'
  }

  const parts = windows.map(w => `${w.remainingPercent}%${w.resetIn ? ` ${w.resetIn}` : ''}`)

  return `◔ ${parts.join(' · ')}`
}

/**
 * Interactive `display.quota` picker: ↑/↓ + Enter over the modes, each row
 * previewing the status-bar segment it produces with the account's live
 * numbers. Applying goes through `config.set` — the same persistence path the
 * text `/quota <mode>` takes — and the read-out follows on the next config
 * sync, so there is nothing to restart.
 */
export function QuotaPicker({ gw, maxWidth, onClose, t }: QuotaPickerProps) {
  const { accountUsage, quotaDisplay } = useStore($uiState)
  const [err, setErr] = useState('')
  const { stdout } = useStdout()

  const width = clampOverlayWidth(Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, (stdout?.columns ?? 80) - 6)), maxWidth)

  const apply = (mode: QuotaDisplay) => {
    // Optimistic: the segment reacts immediately, and the next config sync
    // confirms it from disk. A failed write surfaces in the overlay instead of
    // leaving the user guessing why nothing changed.
    patchUiState({ quotaDisplay: mode })
    gw.request('config.set', { key: 'quota', value: mode })
      .then(() => onClose())
      .catch((e: unknown) => {
        patchUiState({ quotaDisplay })
        setErr(rpcErrorMessage(e))
      })
  }

  const sel = useMenu(
    MODES.map(({ mode }) => ({ label: mode, run: () => apply(mode) })),
    onClose
  )

  const modeWidth = Math.max(...MODES.map(m => m.mode.length))
  const previews = MODES.map(({ mode }) => quotaModePreview(accountUsage, mode))
  const previewWidth = Math.max(...previews.map(p => p.length))

  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={t.color.primary}>
        Quota read-out
      </Text>
      <Text color={t.color.muted} wrap="truncate-end">
        {accountUsage
          ? `${accountUsage.provider}${accountUsage.plan ? ` · ${accountUsage.plan}` : ''} — previews use your current limits`
          : 'no limits reported yet — the provider may not expose a quota'}
      </Text>

      <Box flexDirection="column" marginTop={1}>
        {MODES.map(({ help, mode }, i) => (
          <Text key={mode} wrap="truncate-end" {...listRowStyle(t, i === sel)}>
            {mode === quotaDisplay ? '▸ ' : '  '}
            {mode.padEnd(modeWidth)}
            {'  '}
            {previews[i]!.padEnd(previewWidth)}
            {'  '}
            <Text color={i === sel ? undefined : t.color.muted}>{help}</Text>
          </Text>
        ))}
      </Box>

      {err ? (
        <Box marginTop={1}>
          <Text color={t.color.error} wrap="truncate-end">
            {err}
          </Text>
        </Box>
      ) : null}

      <Box marginTop={1}>
        <OverlayHint t={t}>↑/↓ move · enter apply · esc cancel · ▸ current</OverlayHint>
      </Box>
    </Box>
  )
}

interface QuotaPickerProps {
  gw: GatewayClient
  maxWidth?: number
  onClose: () => void
  t: Theme
}
