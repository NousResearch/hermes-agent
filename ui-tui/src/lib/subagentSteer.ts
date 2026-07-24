import type { AsyncDelegationRecord } from '../gatewayTypes.js'
import type { SubagentProgress } from '../types.js'

// Parse + resolve the composer's `@<id> steer text` shorthand. Kept pure and
// ink-free so it's unit testable and the submit hot-path only calls into it.

export interface SteerCommand {
  body: string
  token: string
}

// `@<token> <text>` where token is a run of non-space chars and text is the
// rest (dotall so multi-line steers survive). Returns null for anything that
// isn't shaped like a steer so ordinary prompts pass straight through.
const STEER_RE = /^@(\S+)\s+([\s\S]+)$/

export const parseSteerCommand = (text: string): SteerCommand | null => {
  const m = STEER_RE.exec(text.trimStart())

  if (!m) {
    return null
  }

  const body = m[2]!.trim()

  return body ? { body, token: m[1]! } : null
}

/** Resolve a steer token to a live subagent id. Only live in-turn subagents
 * (which are actually addressable in the backend registry) are candidates.
 * Matches an exact id first, then a unique id-prefix; returns null when the
 * token is ambiguous or matches nothing so the caller falls back to a normal
 * turn. Ambiguity resolving to nothing is deliberate — never steer a guess. */
export const resolveSteerTargetId = (token: string, subagents: SubagentProgress[]): null | string => {
  const running = subagents.filter(s => s.status === 'running' || s.status === 'queued')

  const exact = running.find(s => s.id === token)

  if (exact) {
    return exact.id
  }

  const prefixed = running.filter(s => s.id.startsWith(token))

  return prefixed.length === 1 ? prefixed[0]!.id : null
}

export const resolveAsyncSteerTargetId = (
  token: string,
  delegations: readonly AsyncDelegationRecord[]
): null | string => {
  const running = delegations.filter(d => d.status === 'running')
  const exact = running.find(d => d.delegation_id === token)

  if (exact) {
    return exact.delegation_id
  }

  const prefixed = running.filter(d => d.delegation_id.startsWith(token))

  return prefixed.length === 1 ? prefixed[0]!.delegation_id : null
}
