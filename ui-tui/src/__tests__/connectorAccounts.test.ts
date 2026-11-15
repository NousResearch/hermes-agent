import type { ConnectionUpdatePayload, ConnectorAccountRow } from '@hermes/shared/gateway-events'
import { afterEach, describe, expect, it } from 'vitest'

import {
  applyConnectionRequest,
  applyConnectionUpdate,
  resetConnectionOperationsForTests
} from '../app/connectionOperationStore.js'
import { getOverlayState, hasPromptOpen, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import {
  accountRowText,
  accountView,
  nameProblem,
  needsRepair,
  offersReconnect,
  panelShortcut,
  reconnectRefusal
} from '../domain/connectorAccounts.js'

const row = (over: Partial<ConnectorAccountRow> & { disabled?: boolean }): ConnectorAccountRow =>
  ({
    active: true,
    alias: null,
    connection_id: 'ca_1',
    connector: 'gmail',
    created_at: '2026-10-08T00:00:00Z',
    label: 'gmail-x1',
    status: 'active',
    updated_at: '2026-10-08T00:00:00Z',
    ...over
  }) as ConnectorAccountRow

describe('connector account rows', () => {
  const rows = [
    row({ alias: 'work', connection_id: 'ca_w' }),
    row({ alias: 'home', connection_id: 'ca_h' }),
    row({ alias: null, connection_id: 'ca_old', disabled: true, status: 'inactive' }),
    row({ connection_id: 'ca_n', label: 'sid@example.com' })
  ]

  it('folds retired accounts and names unnamed ones by their label', () => {
    const folded = accountView(rows, false)

    expect(folded.retired).toBe(1)
    expect(folded.rows.map(r => r.connection_id)).toEqual(['ca_h', 'ca_n', 'ca_w'])
    expect(accountView(rows, true).rows.at(-1)?.connection_id).toBe('ca_old')
    expect(accountRowText(folded.rows[1]!, 'active', '(unnamed)')).toBe('gmail · sid@example.com (unnamed) · active')
  })

  it('refuses a malformed name and a name another live account of the same app holds', () => {
    expect(nameProblem('Work', 'gmail', rows)).toBe('invalid')
    expect(nameProblem('-x', 'gmail', rows)).toBe('invalid')
    expect(nameProblem('home', 'gmail', rows)).toBe('taken')
    expect(nameProblem('home', 'gmail', rows, 'ca_h')).toBeNull()
    expect(nameProblem('home', 'slack', rows)).toBeNull()
  })
})

describe('reconnect is offered only as a repair', () => {
  it('skips a healthy account and a retired one, and offers every other status', () => {
    expect(needsRepair(row({ status: 'active' }))).toBe(false)
    expect(needsRepair(row({ disabled: true, status: 'inactive' }))).toBe(false)

    for (const status of ['failed', 'expired', 'revoked', 'inactive'] as const) {
      expect(needsRepair(row({ active: false, status }))).toBe(true)
      expect(reconnectRefusal(row({ active: false, status }))).toBeNull()
    }

    expect(reconnectRefusal(row({ status: 'active' }))).toBe('healthy')
    expect(reconnectRefusal(row({ disabled: true, status: 'inactive' }))).toBe('retired')
    expect(offersReconnect(row({ status: 'active' }))).toBe(false)
    expect(offersReconnect(undefined)).toBe(true)
  })
})

describe('panel letter shortcuts', () => {
  it('maps a bare letter to its action and never a Ctrl or Meta chord, so Ctrl+C cancels without reconnecting', () => {
    expect(panelShortcut('c', { ctrl: false, meta: false })).toBe('reconnect')
    expect(panelShortcut('c', { ctrl: true, meta: false })).toBeNull()
    expect(panelShortcut('x', { ctrl: false, meta: true })).toBeNull()
    expect(panelShortcut('z', {})).toBeNull()
  })
})

describe('settled connection lines', () => {
  afterEach(resetConnectionOperationsForTests)

  it('tell the user the name the agent chose once an aliased account connects', () => {
    const target = { action: 'connect', alias: 'work', kind: 'connector', name: 'gmail', state: 'initiated' }

    applyConnectionRequest({ deadline_at: 0, op_id: 'op', seq: 1, targets: [target] } as never)

    const lines = applyConnectionUpdate({
      deadline_at: 0,
      op_id: 'op',
      seq: 2,
      settled: true,
      targets: [{ ...target, state: 'connected' }]
    } as unknown as ConnectionUpdatePayload)

    expect(lines).toEqual(['gmail (work): connected', 'Hermes named this gmail account work. Rename it anytime.'])
  })
})

describe('panels yield the keyboard to prompts', () => {
  afterEach(resetOverlayState)

  it('counts an agent question or a confirm as open, and the accounts panel alone as not', () => {
    patchOverlayState({ connectors: true })
    expect(hasPromptOpen(getOverlayState())).toBe(false)

    patchOverlayState({ confirm: { onConfirm: () => undefined, title: 'Remove?' } })
    expect(hasPromptOpen(getOverlayState())).toBe(true)

    patchOverlayState({ clarify: { choices: null, question: 'Which?', requestId: 'r1' } as never, confirm: null })
    expect(hasPromptOpen(getOverlayState())).toBe(true)
  })
})
