import { expect, it } from 'vitest'

import { canonicalResult } from '../canonicalGateway.js'
import { toTranscriptMessages } from '../domain/messages.js'

// N94: a resumed canonical snapshot whose user row is multipart (text + image parts) still renders
// that user turn; `content` stays on the row for /undo targeting.
it('a multipart user row survives the canonical resume projection', () => {
  const content = [
    { type: 'text', text: 'what is in this picture?' },
    { type: 'image_url', image_url: { url: 'data:image/png;base64,AA' } }
  ]

  const snapshot = canonicalResult('session.resume', {
    session_id: 's',
    stored_session_id: 's',
    authority_epoch: 1,
    execution_generation: 0,
    messages: [
      { role: 'user', content },
      { role: 'assistant', content: 'a cat' }
    ],
    info: {}
  })

  expect(snapshot.messages[0].content).toEqual(content)
  const rows = toTranscriptMessages(snapshot.messages)
  expect(rows.map(row => [row.role, row.text])).toEqual([
    ['user', 'what is in this picture?\n[image]'],
    ['assistant', 'a cat']
  ])
})
