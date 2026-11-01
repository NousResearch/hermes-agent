import { expect, it, vi } from 'vitest'

import { openCanonicalApproval, settleSharedPrompt } from '../app/gatewayCanonicalEvents.js'
import { $overlayState, patchOverlayState } from '../app/overlayStore.js'
import { approvalOwnership, canonicalRequest, sharedControlParams } from '../canonicalGateway.js'
import { GatewayClient } from '../gatewayClient.js'
import type { AnyGatewayEvent } from '../gatewayTypes.js'

it.each([{ dispatch_owner: 'lease' }, { execution_generation: 3 }])('serializes an actual approval card with exactly its owner', owner => {
  patchOverlayState({ approval: null })
  openCanonicalApproval({ type: 'approval.request', session_id: 's', payload: {
    prompt_id: 'p', ...owner
  } }, vi.fn(), vi.fn())
  const params = { choice: 'once', ...sharedControlParams($overlayState.get().approval) }
  expect(canonicalRequest('approval.respond', params).params).toEqual({ session_id: 's', prompt_id: 'p', choice: 'once', ...owner })
})

it.each([{}, { dispatch_owner: '' }, { dispatch_owner: null }, { dispatch_owner: 'lease', execution_generation: 3 }, { execution_generation: '3' }])('refuses malformed owner before displaying consent', owner => {
  patchOverlayState({ approval: null })
  expect(approvalOwnership(owner)).toBeNull()
  openCanonicalApproval({ type: 'approval.request', session_id: 's', payload: { prompt_id: 'p', ...owner } } as never, vi.fn(), vi.fn())
  expect($overlayState.get().approval).toBeNull()
})

it('preserves dispatch ownership when the actual client hydrates a resume snapshot', () => {
  const client = new GatewayClient()
  const events: unknown[] = []
  // Exercise the canonical snapshot producer without network or credentials.
  Object.defineProperty(client, 'isCanonical', { value: true })
  vi.spyOn(client, 'publishLocalEvent').mockImplementation(event => { events.push(event) })
  client.hydrateSharedPrompts({ session_id: 's', authority_epoch: 1, prompts: [
    { kind: 'approval', prompt_id: 'p', dispatch_owner: 'lease' }
  ] })
  expect(events).toEqual([{ type: 'approval.request', session_id: 's', payload: {
    kind: 'approval', prompt_id: 'p', dispatch_owner: 'lease', execution_epoch: '1'
  } }])
})

it.each([{ dispatch_owner: 'lease' }, { execution_generation: 3 }])('retains ownership across real wire normalization, card, response and settlement', async owner => {
  patchOverlayState({ approval: null })
  const client = new GatewayClient()
  Object.defineProperty(client, 'isCanonical', { value: true })
  const events: AnyGatewayEvent[] = []
  client.on('event', event => {
    events.push(event)
    if (event.type === 'approval.request') {
      openCanonicalApproval(event, vi.fn(), vi.fn())
    } else {
      settleSharedPrompt(event)
    }
  })
  client.drain()
  await Promise.resolve()
  // Feed actual JSON-RPC transport frames without opening a socket or sidecar.
  const wire = client as unknown as { handleWebSocketFrame: (frame: string) => void }
  wire.handleWebSocketFrame(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: {
    type: 'approval.request', session_id: 's', authority_epoch: 1, execution_generation: 'dispatch_owner' in owner ? 99 : 3,
    payload: { prompt_id: 'p', ...owner }
  } }))
  expect(events[0]?.payload).toEqual({ prompt_id: 'p', execution_epoch: '1', ...owner })
  expect(canonicalRequest('approval.respond', {
    choice: 'once', ...sharedControlParams($overlayState.get().approval)
  }).params).toEqual({ session_id: 's', prompt_id: 'p', choice: 'once', ...owner })
  for (const prompt_id of ['stale', 'p']) {
    wire.handleWebSocketFrame(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: {
      type: 'approval.settled', session_id: 's', authority_epoch: 1, execution_generation: 'dispatch_owner' in owner ? 99 : 3,
      payload: { prompt_id, ...owner }
    } }))
    expect($overlayState.get().approval === null).toBe(prompt_id === 'p')
  }
})
