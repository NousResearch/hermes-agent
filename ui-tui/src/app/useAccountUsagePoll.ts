import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import type { AccountUsageResponse } from '../gatewayTypes.js'
import { asRpcResult } from '../lib/rpc.js'

import type { AccountUsageInfo, AccountUsageWindow, QuotaDisplay } from './interfaces.js'
import { $uiState, patchUiState } from './uiStore.js'

const USAGE_POLL_MS = 60_000

// A session created moments ago has no resident agent, and its row may not be
// on disk yet, so the first resolve can legitimately come up empty. Retry on a
// short cadence until the first snapshot lands — otherwise the read-out stays
// blank for a full minute after launch, which is exactly when the user is
// looking at the branding panel. Bounded, so a provider that simply has no
// quota API settles onto the steady cadence instead of polling forever.
const USAGE_WARMUP_MS = 5_000
const USAGE_WARMUP_TRIES = 6

/**
 * The windows the status bar shows, in render order, under `display.quota`.
 *
 * The short rolling window leads: it is the cap that bites next, so its
 * percentage and countdown are the pair worth reading mid-turn. `both` appends
 * the weekly window after it — last, and separately width-budgeted, so it is
 * the first thing to go on a narrow terminal.
 *
 * 'session' / 'weekly' match on the label a provider reports (Codex calls them
 * "Session" and "Weekly"); a provider that names its windows otherwise falls
 * back to the tightest one, so a setting never blanks the segment.
 */
export const selectQuotaWindows = (
  windows: readonly AccountUsageWindow[],
  mode: QuotaDisplay
): readonly AccountUsageWindow[] => {
  if (mode === 'off' || !windows.length) {
    return []
  }

  const tightest = windows.reduce((best, w) => (w.remainingPercent < best.remainingPercent ? w : best))

  if (mode === 'tightest') {
    return [tightest]
  }

  const byPrefix = (prefix: string) => windows.find(w => w.label.toLowerCase().startsWith(prefix))
  const session = byPrefix('session')
  const weekly = byPrefix('week')

  if (mode === 'session') {
    return [session ?? tightest]
  }

  if (mode === 'weekly') {
    return [weekly ?? tightest]
  }

  const ordered = [session, weekly].filter((w): w is AccountUsageWindow => !!w)

  return ordered.length ? ordered : windows.slice(0, 2)
}

/** Delay before the next poll: steady once a snapshot lands, brief while warming up. */
export const nextPollDelay = (landed: boolean, warmupLeft: number): number =>
  landed || warmupLeft <= 0 ? USAGE_POLL_MS : USAGE_WARMUP_MS

/**
 * Coarse countdown to a quota reset: `12m`, `2h 45m`, `5d 3h`.
 *
 * Deliberately low-resolution — the value is recomputed once per poll, so a
 * live-ticking seconds field would only ever be a minute stale. An instant
 * already past (clock skew, a window that just rolled) reads `now`.
 */
export const formatResetIn = (resetAt: null | string | undefined, now = Date.now()): string => {
  if (!resetAt) {
    return ''
  }

  const target = Date.parse(resetAt)

  if (!Number.isFinite(target)) {
    return ''
  }

  const minutes = Math.max(0, Math.round((target - now) / 60_000))

  if (minutes < 1) {
    return 'now'
  }

  const days = Math.floor(minutes / 1440)
  const hours = Math.floor((minutes % 1440) / 60)

  if (days > 0) {
    return `${days}d ${hours}h`
  }

  return hours > 0 ? `${hours}h ${minutes % 60}m` : `${minutes}m`
}

/** Coerce an `account.usage` RPC payload into the UI's AccountUsageInfo shape. */
export const toAccountUsageInfo = (r: AccountUsageResponse | null): AccountUsageInfo | null => {
  if (!r?.available) {
    return null
  }

  const windows: AccountUsageWindow[] = []

  for (const w of r.windows ?? []) {
    if (typeof w.used_percent !== 'number' || !Number.isFinite(w.used_percent)) {
      continue
    }

    const usedPercent = Math.max(0, Math.min(100, Math.round(w.used_percent)))

    windows.push({
      label: w.label || 'Quota',
      remainingPercent: 100 - usedPercent,
      resetAt: typeof w.reset_at === 'string' ? w.reset_at : null,
      resetIn: formatResetIn(w.reset_at),
      usedPercent
    })
  }

  // A snapshot with no usable window is indistinguishable from "no quota API"
  // as far as the read-outs are concerned — both hide the segment.
  return windows.length ? { plan: r.plan ?? null, provider: r.provider ?? '', windows } : null
}

/**
 * Poll the provider's account quota for the branding panel + status bar.
 *
 * Session-gated: the gateway resolves provider/base_url/api_key from the live
 * agent, so there is nothing to ask for before a session exists (and the
 * snapshot is cleared when one ends). One minute is plenty for a limit that
 * moves per turn at most, and the RPC is on the gateway's long-handler pool,
 * so a slow provider never stalls the stdin reader. A failed poll keeps the
 * last-good snapshot rather than blanking the read-out mid-conversation.
 */
export function useAccountUsagePoll(gw: GatewayClient) {
  const { quotaDisplay, sid } = useStore($uiState)

  useEffect(() => {
    // `display.quota: off` is a real off switch, not just a hidden read-out:
    // no poll, so the provider is never asked for a quota nobody is showing.
    if (!sid || quotaDisplay === 'off') {
      patchUiState({ accountUsage: null })

      return
    }

    let cancelled = false
    let warmupLeft = USAGE_WARMUP_TRIES
    let timer: ReturnType<typeof setTimeout> | undefined

    const poll = async () => {
      let landed = false

      try {
        const r = asRpcResult<AccountUsageResponse>(
          await gw.request<AccountUsageResponse>('account.usage', { session_id: sid })
        )

        const info = toAccountUsageInfo(r)

        if (cancelled) {
          return
        }

        landed = !!info
        patchUiState({ accountUsage: info })
      } catch {
        // Keep the last-good snapshot on a transient RPC failure (a session
        // the gateway has not registered yet answers with an error, which is
        // exactly the case the warm-up cadence below covers).
      }

      if (!cancelled) {
        const delay = nextPollDelay(landed, warmupLeft)

        warmupLeft = landed ? 0 : Math.max(0, warmupLeft - 1)
        timer = setTimeout(() => void poll(), delay)
      }
    }

    void poll()

    return () => {
      cancelled = true

      if (timer) {
        clearTimeout(timer)
      }
    }
  }, [gw, quotaDisplay, sid])
}
