import type { GatewayEvent } from '../gatewayTypes.js'
import type { Usage } from '../types.js'

export type StatusCapsuleObservation = 'observed' | 'unknown'
export type StatusCapsulePrecision = 'observed' | 'unknown'
export type StatusCapsuleRoute = 'Auto Session' | 'Hermes' | 'MoA'
export type StatusCapsuleState = 'delegating' | 'error' | 'idle' | 'interrupted' | 'running' | 'session_starting' | 'tooling'
export type StatusCapsuleTokenSource = 'context_window' | 'provider_usage' | 'unknown'
export type StatusCapsuleElapsedSource = 'local_wall_clock' | 'unknown'

export interface StatusCapsuleObserved {
  deep: StatusCapsuleObservation
  delegate: StatusCapsuleObservation
  moa: StatusCapsuleObservation
  opencode: StatusCapsuleObservation
}

export interface StatusCapsulePrecisionBlock {
  elapsed: StatusCapsulePrecision
  tokens: StatusCapsulePrecision
}

export interface StatusCapsule {
  elapsedSource: StatusCapsuleElapsedSource
  observed: StatusCapsuleObserved
  precision: StatusCapsulePrecisionBlock
  route: StatusCapsuleRoute
  schemaVersion: 1
  state: StatusCapsuleState
  tokenSource: StatusCapsuleTokenSource
  tools: string[]
}

export interface BuildStatusCapsuleInput {
  autoSession?: boolean
  busy: boolean
  lastTurnEndedAt?: null | number
  prior?: null | StatusCapsule
  sessionStartedAt?: null | number
  status: string
  turnStartedAt?: null | number
  usage: Usage
}

const UNKNOWN_OBSERVED: StatusCapsuleObserved = {
  deep: 'unknown',
  delegate: 'unknown',
  moa: 'unknown',
  opencode: 'unknown'
}

const uniqAppend = (items: string[], item: string): string[] => {
  const value = item.trim()

  if (!value) {
    return items
  }

  return items.includes(value) ? items : [...items, value].slice(-4)
}

const hasProviderUsage = (usage?: Partial<Usage> | null): boolean =>
  Boolean(usage && ((usage.total ?? 0) > 0 || (usage.calls ?? 0) > 0 || (usage.input ?? 0) > 0 || (usage.output ?? 0) > 0))

const hasContextWindowUsage = (usage?: Partial<Usage> | null): boolean =>
  Boolean(usage && ((usage.context_used ?? 0) > 0 || (usage.context_max ?? 0) > 0))

const tokenPrecisionFromUsage = (usage?: Partial<Usage> | null): StatusCapsulePrecision => {
  return hasProviderUsage(usage) || hasContextWindowUsage(usage) ? 'observed' : 'unknown'
}

const tokenSourceFromUsage = (usage?: Partial<Usage> | null): StatusCapsuleTokenSource => {
  if (hasProviderUsage(usage)) {
    return 'provider_usage'
  }

  return hasContextWindowUsage(usage) ? 'context_window' : 'unknown'
}

const isAutoSessionStatus = (status: string): boolean => /forging|resuming|recovering|starting agent/i.test(status)

const routeFrom = (input: BuildStatusCapsuleInput): StatusCapsuleRoute => {
  if (input.autoSession || isAutoSessionStatus(input.status)) {
    return 'Auto Session'
  }

  if (input.prior?.route === 'MoA' || input.prior?.observed.moa === 'observed') {
    return 'MoA'
  }

  return 'Hermes'
}

const stateFrom = (input: BuildStatusCapsuleInput, prior: StatusCapsule): StatusCapsuleState => {
  if (input.autoSession || isAutoSessionStatus(input.status)) {
    return 'session_starting'
  }

  if (prior.state === 'error' || prior.state === 'interrupted') {
    return prior.state
  }

  if ((input.usage.active_subagents ?? 0) > 0) {
    return 'delegating'
  }

  if (input.busy && (prior.state === 'tooling' || prior.state === 'delegating')) {
    return prior.state
  }

  return input.busy ? 'running' : 'idle'
}

export const initialStatusCapsule = (): StatusCapsule => ({
  elapsedSource: 'unknown',
  observed: { ...UNKNOWN_OBSERVED },
  precision: { elapsed: 'unknown', tokens: 'unknown' },
  route: 'Hermes',
  schemaVersion: 1,
  state: 'idle',
  tokenSource: 'unknown',
  tools: []
})

export function buildStatusCapsule(input: BuildStatusCapsuleInput): StatusCapsule {
  const prior = input.prior ?? initialStatusCapsule()
  const tokens = tokenPrecisionFromUsage(input.usage)

  const elapsed: StatusCapsulePrecision =
    input.turnStartedAt || input.lastTurnEndedAt || input.sessionStartedAt ? 'observed' : 'unknown'

  const delegate = (input.usage.active_subagents ?? 0) > 0 ? 'observed' : prior.observed.delegate
  const route = routeFrom(input)

  return {
    elapsedSource: elapsed === 'observed' ? 'local_wall_clock' : 'unknown',
    observed: {
      ...prior.observed,
      delegate,
      moa: route === 'MoA' ? 'observed' : prior.observed.moa
    },
    precision: { elapsed, tokens },
    route,
    schemaVersion: 1,
    state: stateFrom(input, prior),
    tokenSource: tokenSourceFromUsage(input.usage),
    tools: prior.tools
  }
}

export function foldStatusCapsuleEvent(previous: StatusCapsule | null | undefined, event: Pick<GatewayEvent, 'payload' | 'type'>): StatusCapsule {
  const next: StatusCapsule = previous ? { ...previous, observed: { ...previous.observed }, precision: { ...previous.precision }, tools: [...previous.tools] } : initialStatusCapsule()

  switch (event.type) {
    case 'message.start':
      return {
        ...next,
        elapsedSource: 'local_wall_clock',
        observed: { ...UNKNOWN_OBSERVED },
        precision: { elapsed: 'observed', tokens: 'unknown' },
        route: 'Hermes',
        state: 'running',
        tokenSource: 'unknown',
        tools: []
      }
    case 'tool.start': {
      const payload = (event.payload ?? {}) as { name?: unknown }
      const name = typeof payload.name === 'string' ? payload.name : 'tool'
      const lower = name.toLowerCase()

      return {
        ...next,
        observed: {
          ...next.observed,
          delegate: lower === 'delegate_task' ? 'observed' : next.observed.delegate,
          opencode: lower === 'opencode' ? 'observed' : next.observed.opencode
        },
        state: lower === 'delegate_task' ? 'delegating' : 'tooling',
        tools: uniqAppend(next.tools, name)
      }
    }

    case 'subagent.spawn_requested':

    case 'subagent.start':

    case 'subagent.thinking':

    case 'subagent.tool':

    case 'subagent.progress':
      return {
        ...next,
        observed: { ...next.observed, delegate: 'observed' },
        state: 'delegating'
      }

    case 'subagent.complete':
      return {
        ...next,
        observed: { ...next.observed, delegate: 'observed' }
      }

    case 'moa.aggregating':

    case 'moa.reference':
      return {
        ...next,
        observed: { ...next.observed, moa: 'observed' },
        route: 'MoA',
        state: next.state === 'idle' ? 'running' : next.state
      }
    case 'message.complete': {
      const usage = event.payload && 'usage' in event.payload ? event.payload.usage : undefined
      const status = event.payload && 'status' in event.payload ? String(event.payload.status ?? '') : ''
      const tokens = tokenPrecisionFromUsage(usage)
      const state: StatusCapsuleState = status === 'interrupted' ? 'interrupted' : status === 'error' ? 'error' : 'idle'

      return {
        ...next,
        elapsedSource: 'local_wall_clock',
        precision: {
          elapsed: 'observed',
          tokens: tokens === 'observed' ? 'observed' : next.precision.tokens
        },
        state,
        tokenSource: tokens === 'observed' ? tokenSourceFromUsage(usage) : next.tokenSource
      }
    }

    case 'error':
      return {
        ...next,
        elapsedSource: 'local_wall_clock',
        precision: { ...next.precision, elapsed: 'observed' },
        state: 'error'
      }

    default:
      return next
  }
}

const stateLabel: Record<StatusCapsuleState, string> = {
  delegating: 'delegating',
  error: 'error',
  idle: 'idle',
  interrupted: 'interrupted',
  running: 'running',
  session_starting: 'starting',
  tooling: 'tool'
}

export function formatStatusCapsule(capsule: StatusCapsule): string {
  const parts = [`⚙ ${capsule.route}`]

  if (capsule.state === 'session_starting' || capsule.state === 'tooling' || capsule.state === 'error' || capsule.state === 'interrupted') {
    parts.push(stateLabel[capsule.state])
  }

  if (capsule.observed.delegate === 'observed') {
    parts.push('delegate')
  } else if (capsule.state === 'delegating') {
    parts.push('delegating')
  }

  if (capsule.observed.moa === 'observed' && capsule.route !== 'MoA') {
    parts.push('moa')
  }

  if (capsule.route === 'MoA' || capsule.observed.delegate === 'observed') {
    parts.push(capsule.observed.opencode === 'observed' ? 'opencode' : 'opencode?')
  }

  if (capsule.observed.deep === 'observed') {
    parts.push('deep')
  }

  parts.push(capsule.precision.tokens === 'observed' ? 'tok obs' : 'tok?')
  parts.push(capsule.precision.elapsed === 'observed' ? 'time obs' : 'time?')

  return parts.join(' · ')
}
