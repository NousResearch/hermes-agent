// OSC 7501 Program Status Protocol (https://www.superlogical.com/rex/docs/build/program-status):
// the TUI's twin of hermes_cli/terminal_notify.py::program_status_sequence. Terminals that track it
// show which Hermes tab is working, waiting on the user, finished or failed.
import { atom } from 'nanostores'

import type { OverlayState } from '../app/interfaces.js'

export type ProgramState = 'blocked' | 'done' | 'error' | 'idle' | 'working'
export type BlockedKind = 'auth' | 'permission' | 'question'
export type TurnOutcome = 'done' | 'error' | 'idle' | null

const MSG_LIMIT_BYTES = 2048
// The spec discards a whole report whose decoded msg carries C0, DEL or C1 controls.
// eslint-disable-next-line no-control-regex
const MSG_CONTROLS = /[\u0000-\u001f\u007f-\u009f]+/g

export const $programStatusEnabled = atom(true)
// How the last turn ended (message.complete / error set it, message.start clears it); shown once busy drops.
export const $turnOutcome = atom<TurnOutcome>(null)

// A cancelled turn reports idle (the spec's rule for user interrupts); a reply-less failure is error.
export function completionOutcome(interrupted: boolean, payload: { partial?: unknown; status?: unknown }): TurnOutcome {
  if (interrupted) {
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
  ['vaultSaveLogin', 'auth'],
  ['connection', 'auth']
]

export function deriveProgramStatus(
  busy: boolean,
  overlay: OverlayState,
  outcome: TurnOutcome
): { kind?: BlockedKind; state: ProgramState } {
  const blocked = BLOCKING.find(([key]) => Boolean(overlay[key]))

  if (blocked) {
    return { kind: blocked[1], state: 'blocked' }
  }

  if (busy) {
    return { state: 'working' }
  }

  return { state: outcome ?? 'idle' }
}
