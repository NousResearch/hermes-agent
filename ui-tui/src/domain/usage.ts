import type { TokenUsagePayload, Usage } from '../types.js'

export const ZERO: Usage = { calls: 0, input: 0, output: 0, total: 0 }

function finiteNumber(value: unknown): number | undefined {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : undefined
  }

  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value)

    return Number.isFinite(parsed) ? parsed : undefined
  }

  return undefined
}

function nonNegativeNumber(value: unknown): number | undefined {
  const parsed = finiteNumber(value)

  return parsed === undefined ? undefined : Math.max(0, parsed)
}

function positiveNumber(value: unknown): number | undefined {
  const parsed = finiteNumber(value)

  return parsed === undefined || parsed <= 0 ? undefined : parsed
}

function percentFrom(payload: TokenUsagePayload): number | undefined {
  const explicit = nonNegativeNumber(payload.context_pct)

  if (explicit !== undefined) {
    return Math.min(100, explicit)
  }

  const used = nonNegativeNumber(payload.context_tokens)
  const max = positiveNumber(payload.context_length)

  return used !== undefined && max !== undefined ? Math.min(100, (used / max) * 100) : undefined
}

export function usageFromTokenUsagePayload(payload: null | TokenUsagePayload | undefined): Partial<Usage> | null {
  if (!payload) {
    return null
  }

  const usage: Partial<Usage> = {}
  const input = nonNegativeNumber(payload.input_tokens)
  const output = nonNegativeNumber(payload.output_tokens)
  const total = nonNegativeNumber(payload.total_tokens)
  const contextUsed = nonNegativeNumber(payload.context_tokens)
  const contextMax = positiveNumber(payload.context_length)
  const contextPercent = percentFrom(payload)

  if (input !== undefined) {
    usage.input = input
  }

  if (output !== undefined) {
    usage.output = output
  }

  if (total !== undefined) {
    usage.total = total
  }

  if (contextUsed !== undefined) {
    usage.context_used = contextUsed
  }

  if (contextMax !== undefined) {
    usage.context_max = contextMax
  }

  if (contextPercent !== undefined) {
    usage.context_percent = contextPercent
  }

  return Object.keys(usage).length ? usage : null
}

export function mergeTokenUsagePayload(current: Usage, payload: null | TokenUsagePayload | undefined): Usage {
  const usage = usageFromTokenUsagePayload(payload)

  return usage ? { ...current, ...usage } : current
}
