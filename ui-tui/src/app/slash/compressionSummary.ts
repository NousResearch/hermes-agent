import type { SessionCompressResponse } from '../../gatewayTypes.js'
import { t } from '../../i18n/runtime.js'

function compressionTokenLine(summary: NonNullable<SessionCompressResponse['summary']>): null | string {
  if (summary.before_tokens == null || summary.after_tokens == null) {
    return null
  }

  return summary.refused_would_grow || (summary.noop && summary.before_tokens === summary.after_tokens)
    ? t('compression.tokensUnchanged', String(summary.before_tokens))
    : t('compression.tokensChanged', String(summary.before_tokens), String(summary.after_tokens))
}

/** Render structured compression feedback in the active TUI locale.
 * Legacy backends without structured fields return ``null`` so their
 * pre-rendered summary remains a compatibility fallback. */
export const formatCompressionSummary = (response: SessionCompressResponse): null | string[] => {
  const summary = response.summary

  if (!summary || summary.before_count == null || summary.after_count == null) {
    return null
  }

  const values = { before: summary.before_count, after: summary.after_count }

  const headline = summary.refused_would_grow
    ? t('compression.refused', values.before)
    : summary.aborted
      ? t('compression.aborted', values.before)
      : summary.fallback_used
        ? t('compression.fallback', values.before, values.after)
        : summary.noop
          ? t('compression.noop', values.before)
          : t('compression.done', values.before, values.after)

  const lines = [headline]

  const tokenLine = compressionTokenLine(summary)

  if (tokenLine) {
    lines.push(tokenLine)
  }

  if (summary.refused_would_grow) {
    lines.push(t('compression.refusedNote'))
  } else if (summary.aborted) {
    lines.push(t('compression.abortedNote'))
  } else if (summary.fallback_used) {
    lines.push(t('compression.fallbackNote', summary.dropped_count ?? 0))
  } else if (
    !summary.noop &&
    summary.after_count < summary.before_count &&
    (summary.after_tokens ?? 0) > (summary.before_tokens ?? 0)
  ) {
    lines.push(t('compression.denseNote'))
  }

  if (summary.failure_reason) {
    lines.push(t('compression.reason', summary.failure_reason))
  }

  return lines
}

/** Render the same structured report for sidecar and shared-gateway commands. */
export function printCompressionSummary(
  summary: SessionCompressResponse['summary'],
  writeLine: (line: string) => void
): boolean {
  const lines = formatCompressionSummary({ summary })

  if (!lines) {
    return false
  }

  lines.forEach((line, index) => {
    const prefix =
      index === 0 ? (!summary?.refused_would_grow && !summary?.aborted && !summary?.noop ? '✓ ' : '') : '  '

    writeLine(`${prefix}${line}`)
  })

  return true
}
