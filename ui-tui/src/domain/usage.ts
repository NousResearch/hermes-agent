import type { Usage } from '@hermes/shared/gateway-events'

export const ZERO: Usage = { calls: 0, input: 0, output: 0, total: 0 }

const num = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) ? value : undefined

const round1 = (value: number) => Math.round(value * 10) / 10
const pct = (part: number, whole: number) => Math.max(0, Math.min(100, Math.round((part / whole) * 100)))

// Main `Usage` key ← owner stats key(s), first non-zero wins (`gateway/session_usage.py::tui_usage`).
const COUNTERS: [keyof Usage, string[]][] = [
  ['calls', ['api_calls']],
  ['input', ['input_tokens', 'prompt_tokens']],
  ['output', ['output_tokens', 'completion_tokens']],
  ['reasoning', ['reasoning_tokens']],
  ['prompt', ['prompt_tokens']],
  ['completion', ['completion_tokens']],
  ['total', ['total_tokens']]
]

function contextFields(p: Record<string, unknown>): Partial<Usage> {
  const used = num(p.last_prompt_tokens) ?? 0
  const max = num(p.context_length) ?? 0
  const fields: Partial<Usage> = p.context_length == null ? {} : { compressions: num(p.compression_count) ?? 0 }

  if (!used || !max) {
    return fields
  }

  const source = typeof p.context_source === 'string' && p.context_source ? p.context_source : 'provider_usage'
  const clamped = Math.min(used, max)

  return {
    ...fields,
    context_estimated: source !== 'provider_usage',
    context_max: max,
    context_percent: pct(clamped, max),
    context_source: source,
    context_used: clamped
  }
}

function rateFields(p: Record<string, unknown>): Partial<Usage> {
  const fields: Partial<Usage> = {}
  const promptTotal = num(p.prompt_tokens) ?? 0
  const cacheRead = num(p.cache_read_tokens) ?? 0
  const latency = num(p.avg_latency_s)
  const tps = num(p.tokens_per_s)

  if (promptTotal > 0 && cacheRead > 0) {
    fields.cache_hit_pct = pct(cacheRead, promptTotal)
  }

  if (latency !== undefined) {
    fields.avg_latency_s = round1(latency)
  }

  if (tps !== undefined) {
    fields.avg_tps = round1(tps)
  }

  if (num(p.active_subagents) !== undefined) {
    fields.active_subagents = num(p.active_subagents)
  }

  return fields
}

/** A canonical local viewer's `session.usage` tick carries the owner's session-cumulative counters at
 * the top level (`gateway/session_usage.py::usage_stats`, the `session.stats` dict); main's sidecar
 * ticker sent its `_get_usage` shape under `usage`. Both land as the same status-bar `Usage`
 * (`session_usage.tui_usage` is the Python twin used for `message.complete.usage`). */
export function usageFromTick(payload: unknown): Partial<Usage> | undefined {
  if (!payload || typeof payload !== 'object') {
    return undefined
  }

  const p = payload as Record<string, unknown>

  if (p.usage && typeof p.usage === 'object') {
    return p.usage as Partial<Usage>
  }

  if (num(p.api_calls) === undefined && num(p.total_tokens) === undefined) {
    return undefined
  }

  const usage: Partial<Usage> = Object.fromEntries(
    COUNTERS.map(([key, sources]) => [key, sources.map(source => num(p[source])).find(Boolean) ?? 0])
  )

  if (typeof p.model === 'string' && p.model) {
    usage.model = p.model
  }

  return { ...usage, ...contextFields(p), ...rateFields(p) }
}
