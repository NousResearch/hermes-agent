import type { Usage } from '@hermes/shared/gateway-events'

export const ZERO: Usage = {
  active_subagents: null,
  avg_latency_s: null,
  avg_tps: null,
  cache_hit_pct: null,
  cache_read: null,
  cache_write: null,
  calls: 0,
  completion: 0,
  compressions: null,
  context_estimated: null,
  context_max: null,
  context_percent: null,
  context_source: null,
  context_used: null,
  cost_status: null,
  cost_usd: null,
  dev_credits_spent_micros: null,
  input: 0,
  model: '',
  output: 0,
  prompt: 0,
  reasoning: 0,
  total: 0
}
