import { afterEach, describe, expect, it } from 'vitest'

import { getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { completionOutcome, deriveProgramStatus, encodeProgramStatus } from '../lib/programStatus.js'
import type { ApprovalReq } from '../types.js'

const decode = (seq: string) => {
  const msg = /:msg=([A-Za-z0-9+/=]+)/.exec(seq)?.[1]

  return msg ? Buffer.from(msg, 'base64').toString('utf8') : ''
}

describe('OSC 7501 program status', () => {
  afterEach(() => resetOverlayState())

  it('matches the classic CLI encoder and strips every control the spec refuses', () => {
    const seq = encodeProgramStatus('blocked', 'permission', 'Allow\ncommand?\u0085 x')

    expect(seq.startsWith('\x1b]7501;state=blocked:app=hermes:kind=permission:msg=')).toBe(true)
    expect(seq.endsWith('\x1b\\')).toBe(true)
    expect(decode(seq)).toBe('Allow command? x')
    expect(encodeProgramStatus('working', 'permission')).toBe('\x1b]7501;state=working:app=hermes\x1b\\')
  })

  it('a prompt outranks busy, and an answered prompt falls back to working, then the turn outcome', () => {
    patchOverlayState({ approval: { command: 'rm -rf build' } as ApprovalReq })
    expect(deriveProgramStatus(true, getOverlayState(), null)).toEqual({ kind: 'permission', state: 'blocked' })

    patchOverlayState({ approval: null })
    expect(deriveProgramStatus(true, getOverlayState(), null)).toEqual({ state: 'working' })
    expect(deriveProgramStatus(false, getOverlayState(), 'error')).toEqual({ state: 'error' })
    expect(deriveProgramStatus(false, getOverlayState(), null)).toEqual({ state: 'idle' })
  })

  it('a connection card blocks only while a row waits on the user; backend interrupts read idle', () => {
    patchOverlayState({ connection: { opId: 'op' } })
    const target = (state: string, connect_url?: string) => ({ connect_url, name: 'x', state }) as never

    expect(deriveProgramStatus(true, getOverlayState(), null, [target('initiated')])).toEqual({ state: 'working' })
    expect(deriveProgramStatus(true, getOverlayState(), null, [target('connected')])).toEqual({ state: 'working' })
    expect(deriveProgramStatus(true, getOverlayState(), null, [target('initiated', 'https://x')])).toEqual({
      kind: 'auth',
      state: 'blocked'
    })
    expect(completionOutcome(false, { status: 'interrupted' })).toBe('idle')
  })
})
