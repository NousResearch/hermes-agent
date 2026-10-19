// OSC 7501 Program Status Protocol (https://www.superlogical.com/rex/docs/build/program-status):
// the TUI's twin of hermes_cli/terminal_notify.py::program_status_sequence. Terminals that track it
// show which Hermes tab is working, waiting on the user, finished or failed.
import type { ConnectionOperationTarget } from '@hermes/shared/gateway-events'
import { atom } from 'nanostores'

import type { OverlayState } from '../app/interfaces.js'

export type ProgramState = 'blocked' | 'done' | 'error' | 'idle' | 'working'
export type BlockedKind = 'auth' | 'permission' | 'question'
export type TurnOutcome = 'done' | 'error' | 'idle' | null

const MSG_LIMIT_BYTES = 2048
// The spec discards a whole report whose decoded msg carries C0, DEL or C1 controls.
// eslint-disable-next-line no-control-regex
const MSG_CONTROLS = /[\u0000-\u001f\u007f-\u009f]+/g

// null until the first successful config read, so an opted-out user never sees a startup report.
export const $programStatusEnabled = atom<boolean | null>(null)
// How the last turn ended (message.complete / error set it, message.start clears it); shown once busy drops.
export const $turnOutcome = atom<TurnOutcome>(null)

// A cancelled turn reports idle (the spec's rule for user interrupts); a reply-less failure is error.
export function completionOutcome(interrupted: boolean, payload: { partial?: unknown; status?: unknown }): TurnOutcome {
  // The backend's own verdict too: a voice barge-in interrupts without the TUI's local flag.
  if (interrupted || payload.status === 'interrupted') {
    return 'idle'
  }

  return payload.status === 'error' && !payload.partial ? 'error' : 'done'
}

export function encodeProgramStatus(state: ProgramState, kind?: BlockedKind, msg = ''): string {
  const pairs = [`state=${state}`, 'app=hermes']

  if (state === 'blocked' && kind) {
    pairs.push(`kind=${kind}`)
  }

  const line = msg.replace(MSG_CONTROLS, ' ').split(/\s+/).filter(Boolean).join(' ')

  if (line) {
    const bytes = Buffer.from(line, 'utf8').subarray(0, MSG_LIMIT_BYTES)
    // Drop a code point the byte cap cut in half.
    const whole = new TextDecoder('utf-8', { fatal: false }).decode(bytes).replace(/\uFFFD$/, '')

    pairs.push(`msg=${Buffer.from(whole, 'utf8').toString('base64')}`)
  }

  return `\x1b]7501;${pairs.join(':')}\x1b\\`
}

const BLOCKING: readonly [keyof OverlayState, BlockedKind][] = [
  ['approval', 'permission'],
  ['clarify', 'question'],
  ['sudo', 'auth'],
  ['secret', 'auth'],
  ['vaultUnlock', 'auth'],
  ['vaultCode', 'auth'],
  ['vaultSaveLogin', 'auth']
]

const RESOLVED = new Set(['connected', 'not_connected', 'skipped'])

// A connection card waits on the user only for a row still unresolved and not mid-authorize
// (initiated without a link yet = the backend minting it); answered or working rows are not blocking.
export const connectionNeedsUser = (targets: readonly ConnectionOperationTarget[] | undefined): boolean =>
  (targets ?? []).some(
    target => !RESOLVED.has(target.state) && !(target.state === 'initiated' && !target.connect_url)
  )

export function deriveProgramStatus(
  busy: boolean,
  overlay: OverlayState,
  outcome: TurnOutcome,
  connectionTargets?: readonly ConnectionOperationTarget[]
): { kind?: BlockedKind; state: ProgramState } {
  const blocked =
    BLOCKING.find(([key]) => Boolean(overlay[key])) ??
    (overlay.connection && connectionNeedsUser(connectionTargets) ? (['connection', 'auth'] as const) : undefined)

  if (blocked) {
    return { kind: blocked[1], state: 'blocked' }
  }

  if (busy) {
    return { state: 'working' }
  }

  return { state: outcome ?? 'idle' }
}
