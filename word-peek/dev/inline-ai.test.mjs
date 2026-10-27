import assert from 'node:assert/strict'
import { test } from 'node:test'
import { generateAnswer } from './inline-ai.js'

const atom = value => ({ get: () => value })
const answer = { title: 'A useful concept', body: 'A plain explanation.', sources: [] }
const passage = { selectedText: 'A concept', context: 'The surrounding paragraph.' }

test('uses one stateless RPC and returns plain data without chat mutations', async () => {
  const calls = []
  const host = {
    state: { focusedSessionId: atom('focused-session'), activeSessionId: atom('other-session') },
    request: async (...args) => { calls.push(args); return { text: JSON.stringify(answer) } }
  }
  const output = await generateAnswer(host, { passage, question: 'What does this mean?', history: [{ question: 'Earlier?', answer: { body: 'Earlier answer.' } }] })
  assert.equal(output.body, answer.body)
  assert.equal(calls.length, 1)
  assert.equal(calls[0][0], 'llm.oneshot')
  assert.equal(calls[0][1].session_id, 'focused-session')
  const input = JSON.parse(calls[0][1].input)
  assert.equal(input.document.surroundingText, passage.context)
  assert.equal(input.previousQuestions[0].answer, 'Earlier answer.')
})

test('pins the original owner even if the foreground profile changes while routes load', async () => {
  let finishRoutes
  let selected = 'original'
  let observed
  const host = {
    state: {
      focusedSessionId: { get: () => selected + '-session' },
      focusedSessionOwner: { get: () => ({ connectionId: 'local', profile: selected }) }
    },
    profileRoutes: () => new Promise(resolve => { finishRoutes = resolve }),
    requestProfile: async (...args) => { observed = args; return { text: JSON.stringify(answer) } }
  }
  const pending = generateAnswer(host, { passage, question: 'Explain.' })
  selected = 'other'
  finishRoutes([{ connectionId: 'local', profile: 'original', targetProfile: 'original', mode: 'local' }])
  await pending
  assert.equal(observed[0].profile, 'original')
  assert.equal(observed[2].session_id, 'original-session')
  assert.equal(observed[3], 120000)
})

test('accepts only source URLs captured from the passage', async () => {
  const supplied = { title: 'Document reference', url: 'https://example.com/reference' }
  const host = { request: async () => ({ text: JSON.stringify({ ...answer, sources: [supplied.url] }) }) }
  const result = await generateAnswer(host, { passage: { ...passage, sources: [supplied] }, question: 'Explain.' })
  assert.deepEqual(result.sources, [supplied])
  host.request = async () => ({ text: JSON.stringify({ ...answer, sources: ['https://invented.invalid/'] }) })
  await assert.rejects(generateAnswer(host, { passage, question: 'Explain.' }), /outside this document/)
})

test('rejects model HTML and malformed JSON without returning insertable content', async () => {
  for (const text of [JSON.stringify({ ...answer, body: '<img src=x onerror=alert(1)>' }), 'not json', JSON.stringify({ ...answer, body: 'See https://invented.invalid/' })]) {
    await assert.rejects(generateAnswer({ request: async () => ({ text }) }, { passage, question: 'Explain.' }))
  }
})

test('dismissal ignores late results; an already-cancelled request never calls Hermes', async () => {
  const controller = new AbortController()
  let finishRequest
  const pending = generateAnswer({ request: () => new Promise(resolve => { finishRequest = resolve }) }, { passage, question: 'Explain.', signal: controller.signal })
  controller.abort()
  await assert.rejects(pending, { name: 'AbortError' })
  finishRequest({ text: JSON.stringify(answer) })
  let called = false
  await assert.rejects(generateAnswer({ request: () => { called = true } }, { passage, question: 'Explain.', signal: controller.signal }), { name: 'AbortError' })
  assert.equal(called, false)
})

