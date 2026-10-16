import { expect, it } from 'vitest'

import { lastTurnRewind } from '../app/slash/canonicalSessionCommands.js'
import { canonicalRequest, canonicalResult } from '../canonicalGateway.js'
import { toTranscriptMessages } from '../domain/messages.js'

// TUI parity (PR #106742 rows 12b/31/32): a resumed canonical session renders main's display
// projection (`display_messages`, the sidecar's `_history_to_messages`), not the raw stored rows.

it('asks the owner for the display projection on every rendering resume', () => {
  expect(canonicalRequest('session.resume', { session_id: 's', cols: 80 }).params).toEqual({ session_id: 's', display_projection: true })
  expect(canonicalRequest('session.activate', { session_id: 's' })).toEqual({ method: 'session.resume', params: { session_id: 's', display_projection: true } })
  // A metadata-only activation never asked for rows.
  expect(canonicalRequest('session.activate', { session_id: 's', omit_messages: true }).params).toEqual({ session_id: 's' })
})

it('renders display_messages and keeps the stored rows for rewind targeting', () => {
  const marker = '[OUT-OF-BAND USER MESSAGE — a direct message from the user]\nfocus on the numbers\n[/OUT-OF-BAND USER MESSAGE]'

  const stored = [
    { role: 'user', content: '[CONTEXT COMPACTION — REFERENCE ONLY] summary END OF CONTEXT SUMMARY', row_id: 1 },
    { role: 'user', content: 'run the tool', row_id: 2 },
    { role: 'assistant', content: '', tool_calls: [{ id: 'c1', function: { name: 'read_file', arguments: '{"path":"notes.txt"}' } }], row_id: 3 },
    { role: 'tool', content: 'body', tool_name: 'read_file', tool_call_id: 'c1', row_id: 4 },
    { role: 'user', content: marker, display_kind: 'steer', row_id: 5 },
    { role: 'assistant', content: 'done', row_id: 6 }
  ]

  // The tool trail rides on the next assistant row, as main renders it.
  const display = [
    { role: 'user', text: 'run the tool', row_id: 2 },
    { role: 'tool', name: 'read_file', context: 'notes.txt', tool_call_id: 'c1', args: { path: 'notes.txt' } },
    { role: 'assistant', text: 'read it', row_id: 7 },
    { role: 'user', text: 'focus on the numbers', display_kind: 'steer', row_id: 5 },
    { role: 'assistant', text: 'done', row_id: 6 }
  ]

  const snapshot = canonicalResult('session.resume', { session_id: 's', stored_session_id: 's', authority_epoch: 1,
    execution_generation: 0, messages: stored, display_messages: display, info: {} })

  const rendered = toTranscriptMessages(snapshot.messages)
  const text = JSON.stringify(rendered)
  expect(text).not.toContain('CONTEXT COMPACTION')
  expect(text).not.toContain('OUT-OF-BAND')
  expect(rendered.filter(row => row.role === 'user').map(row => row.text)).toEqual(['run the tool', 'focus on the numbers'])
  expect(text).toContain('notes.txt')
  // Controls still see the stored rows: /undo targets the steer row by its durable id.
  expect(lastTurnRewind(snapshot)).toEqual({ target_message_id: 5 })
})
