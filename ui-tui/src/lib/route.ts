import type { RouteInfo, SessionInfo, SubagentProgress } from '../types.js'

const DEFAULT_EFFORT_LABEL = 'default'

export const normalizedEffortLabel = (effort?: string | null) => {
  const value = String(effort ?? '')
    .trim()
    .toLowerCase()

  return value || DEFAULT_EFFORT_LABEL
}

export const shortModelLabel = (model?: string | null) =>
  String(model ?? '')
    .split('/')
    .pop()!
    .replace(/^claude[-_]/, '')
    .replace(/^anthropic[-_]/, '')
    .replace(/[-_]/g, ' ')
    .replace(/\b(\d+)\s+(\d+)\b/g, '$1.$2')
    .trim()

export const providerModelLabel = (provider?: string | null, model?: string | null) => {
  const providerLabel = String(provider ?? '').trim()
  const modelLabel = shortModelLabel(model) || String(model ?? '').trim() || 'inherit'

  return providerLabel && providerLabel !== 'unknown' ? `${providerLabel}/${modelLabel}` : modelLabel
}

export const routeModeLabel = (mode?: string | null) => String(mode ?? '').trim() || 'inline'

export const routeTargetLabel = (profile?: string | null) => String(profile ?? '').trim() || 'default'

export const sessionRouteLabel = (info?: Partial<SessionInfo> | null) => {
  const route: Partial<RouteInfo> = info?.route ?? {}
  const mode = routeModeLabel(route.execution_mode)
  const target = routeTargetLabel(route.target_profile ?? info?.profile_name)
  const provider = route.provider ?? info?.provider
  const model = route.model ?? info?.model
  const effort = normalizedEffortLabel(route.reasoning_effort ?? info?.reasoning_effort)
  const fast = info?.fast || info?.service_tier === 'priority' || route.service_tier === 'priority'

  return [
    `route ${mode}→${target}`,
    providerModelLabel(provider, model),
    `effort ${effort}`,
    route.reason ? `reason ${route.reason}` : '',
    fast ? 'fast' : ''
  ]
    .filter(Boolean)
    .join(' · ')
}

export const subagentRouteLabel = (
  item: Pick<SubagentProgress, 'executionMode' | 'model' | 'provider' | 'reasoningEffort' | 'role'>
) =>
  [routeModeLabel(item.executionMode), providerModelLabel(item.provider, item.model), `effort ${normalizedEffortLabel(item.reasoningEffort)}`, item.role]
    .filter(Boolean)
    .join(' · ')
