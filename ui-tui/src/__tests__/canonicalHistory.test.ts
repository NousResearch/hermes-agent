import { expect, it } from 'vitest'

import { canonicalResult } from '../canonicalGateway.js'
import { toTranscriptMessages } from '../domain/messages.js'

it('hydrates canonical persisted replies through the same transcript path as resumed sessions', () => {
  const messages = [
    { role: 'user', content: 'Pick a color', api_content: 'private model-only context' },
    { role: 'assistant', content: '', tool_calls: [{ function: { name: 'clarify' } }] },
    { role: 'tool', content: 'private tool result', tool_name: 'clarify' },
    { role: 'assistant', content: 'CLARIFY_FINISHED', timestamp: 123 },
    { role: 'session_meta', content: null },
  ]

  for (const method of ['session.create', 'session.resume', 'session.activate']) {
    const snapshot = canonicalResult(method, { messages, authority_epoch: 1, execution_generation: 2 })
    const rendered = toTranscriptMessages(snapshot.messages)
    expect(rendered.filter(row => row.role === 'user').map(row => row.text)).toEqual(['Pick a color'])
    expect(rendered.filter(row => row.role === 'assistant')).toEqual([
      { role: 'assistant', text: 'CLARIFY_FINISHED', createdAt: 123, tools: expect.any(Array) },
    ])
    expect(JSON.stringify(rendered)).not.toContain('private')
    expect(messages[0]).not.toHaveProperty('text')
  }
})
