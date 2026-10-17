import type { SessionUsageResponse } from '../../gatewayTypes.js'
import { t } from '../../i18n/runtime.js'
import type { PanelSection } from '../../types.js'

export function usageProviderBlocks(r: null | SessionUsageResponse | undefined): string[] {
  return [r?.account_lines, r?.rate_limit_lines]
    .filter((lines): lines is string[] => Boolean(lines?.length))
    .map(lines => lines.join('\n'))
}

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
