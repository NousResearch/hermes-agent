import type { MessageUserPayload } from '@hermes/shared/gateway-events'
import { expect, it } from 'vitest'

import { projectUserTurn, toTranscriptMessages } from '../domain/messages.js'
import type { Msg } from '../types.js'

const payload: MessageUserPayload = {
  message: { role: 'user', text: 'canonical words', row_id: 41, message_uid: 'occurrence', timestamp: 10 },
  client_message_ids: ['optimistic']
}

it('shows external turns once, preserves optimistic display, and keeps identical text as distinct occurrences', () => {
  const external = projectUserTurn([], payload)
  expect(external).toHaveLength(1)
  expect(projectUserTurn(external, payload)).toBe(external)
  const own: Msg[] = [{ role: 'user', text: '[[ collapsed paste ]]', clientMessageId: 'optimistic' }]
  const confirmed = projectUserTurn(own, payload)
  expect(confirmed).toHaveLength(1)
  expect(confirmed[0]).toMatchObject({ text: '[[ collapsed paste ]]', rowId: 41, messageUid: 'occurrence' })
  expect(
    projectUserTurn(confirmed, { message: { ...payload.message, row_id: 42, message_uid: 'another' } })
  ).toHaveLength(2)
})

it('adopts a hydrated queued occurrence at its dispatch position and merges optimistic sends by identity', () => {
  const early = toTranscriptMessages([{ ...payload.message, row_id: 5 }])
  const projected = projectUserTurn([...early, { role: 'assistant', text: 'previous reply' }], payload)
  expect(projected.map(m => m.role)).toEqual(['assistant', 'user'])
  expect(projected[1].rowId).toBe(41)
  expect(projectUserTurn(projected, { message: { ...payload.message, row_id: 5 } })).toBe(projected)
  const merged = projectUserTurn(
    [
      { role: 'user', text: 'one', clientMessageId: 'c1' },
      { role: 'user', text: 'two', clientMessageId: 'c2' }
    ],
    { ...payload, client_message_ids: ['c1', 'c2'] }
  )
  expect(merged).toHaveLength(1)
  expect(merged[0].text).toBe('canonical words')
})
