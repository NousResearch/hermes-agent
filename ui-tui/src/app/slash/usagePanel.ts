import { formatResetsIn } from '@hermes/shared/plan-usage'

import type { BillingStateResponse } from '../../gatewayTypes.js'
import { t } from '../../i18n/runtime.js'
import type { PanelSection } from '../../types.js'

/** Main's `/usage` session-totals panel (model, input / output / total tokens, API calls, the
 * context line, compressions), from main's `Usage` shape: the sidecar's `session.usage` RPC, or a
 * canonical owner's `session.stats` mapped through `usageFromTick`. */
export interface UsageTotals {
  calls?: null | number
  compressions?: null | number
  context_estimated?: boolean | null
  context_max?: null | number
  context_percent?: null | number
  context_used?: null | number
  input?: null | number
  model?: null | string
  output?: null | number
  total?: null | number
}

export function usageTotalsSections(r: UsageTotals): PanelSection[] {
  const f = (v: null | number | undefined) => (v ?? 0).toLocaleString()

  const rows: [string, string][] = [
    [t('slashCmd.session.usage.rowModel'), r.model ?? ''],
    [t('slashCmd.session.usage.rowInputTokens'), f(r.input)],
    [t('slashCmd.session.usage.rowOutputTokens'), f(r.output)],
    [t('slashCmd.session.usage.rowTotalTokens'), f(r.total)],
    [t('slashCmd.session.usage.rowApiCalls'), f(r.calls)]
  ]

  const sections: PanelSection[] = [{ rows }]

  if (r.context_max) {
    const mark = r.context_estimated ? '~' : ''
    sections.push({
      text: t(
        'slashCmd.session.usage.context',
        `${mark}${f(r.context_used)}`,
        f(r.context_max),
        `${mark}${r.context_percent}`
      )
    })
  }

  if (r.compressions) {
    sections.push({ text: t('slashCmd.session.usage.compressions', String(r.compressions)) })
  }

  return sections
}

/** Usage-window plans: the CLI's `/usage` allowance rows (`agent/account_usage.py`
 *  `_render_usage_window_lines`): plan name, then `Today  [|||       ]  31% used  ·  resets in 5h`.
 *  Plain English like the CLI's lines; `[]` for every other account. */
export function planUsageLines(plan: BillingStateResponse['plan_usage'], nowMs: number = Date.now()): string[] {
  if (plan?.kind !== 'usage_window') {
    return []
  }

  const rows = (plan.gauges ?? []).flatMap(g => {
    const left = Number(g.left_fraction)

    if (!Number.isFinite(left)) {
      return []
    }

    const used = Math.max(0, Math.min(1, 1 - left))
    // Same cells as `render_bar`: any use shows one, only a full allowance fills all ten.
    const filled = Math.min(used < 1 ? 9 : 10, Math.max(used > 0 ? 1 : 0, Math.round(used * 10)))
    // Same percent as the CLI / Desktop (`format_percent_used` / `formatPercentUsed`): floor, so a nearly
    // full allowance reads 99% beside its 9 cells; 100% only once it is spent. Micro-rounded first:
    // `1 - left` can land a hair under an exact basis point.
    const bp = Math.floor(Math.round(used * 10_000 * 1e6) / 1e6)
    const pct = bp <= 0 ? '0%' : bp < 100 ? '<1%' : `${Math.floor(bp / 100)}%`
    const when = formatResetsIn(g.resets_at_ms, nowMs)
    const reset = when === 'soon' ? 'resets soon' : when ? `resets in ${when}` : ''

    return [
      {
        label: g.title || g.label,
        rest: `[${'|'.repeat(filled)}${' '.repeat(10 - filled)}]  ${pct} used${reset ? `  ·  ${reset}` : ''}`
      }
    ]
  })

  const width = Math.max(0, ...rows.map(r => r.label.length))

  return rows.length ? [...(plan.plan ? [plan.plan] : []), ...rows.map(r => `${r.label.padEnd(width)}  ${r.rest}`)] : []
}
