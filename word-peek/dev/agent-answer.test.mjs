import assert from 'node:assert/strict'
import { test } from 'node:test'
import { generateAnswer } from './agent-answer.js'

const atom = value => ({ get: () => value })
const answer = { title: 'A useful concept', body: 'A plain explanation.', sources: [] }
const passage = { selectedText: 'A concept', context: 'The surrounding paragraph.' }

/** A mock host whose prompt.submit is answered by a message.complete event,
 * the way the gateway answers a real chat turn. */
function chatHost({ status = 'complete', text = JSON.stringify(answer), sessionId = 'focused-session' } = {}) {
  const calls = []
  const listeners = []
  const host = {
    state: { focusedSessionId: atom(sessionId) },
    request: async (method, params) => {
      calls.push({ method, params })
      if (method === 'prompt.submit') {
        queueMicrotask(() => listeners.forEach(l => l({ type: 'message.complete', session_id: params.session_id, payload: { status, text } })))
      }
      return {}
    },
    onEvent: (type, listener) => { listeners.push(listener); return () => { const i = listeners.indexOf(listener); if (i >= 0) listeners.splice(i, 1) } }
  }
  return { host, calls }
}

test('submits a chat turn through the agent loop and reads the reply from the event stream', async () => {
  const { host, calls } = chatHost()
  const output = await generateAnswer(host, { passage, question: 'What does this mean?', history: [{ question: 'Earlier?', answer: { body: 'Earlier answer.' } }] })
  assert.equal(output.body, answer.body)
  const methods = calls.map(c => c.method)
  assert.deepEqual(methods, ['session.resume', 'prompt.submit'])
  assert.equal(calls[1].params.session_id, 'focused-session')
  const text = calls[1].params.text
  assert.match(text, /^Word Peek \(inline\): wp-/)
  const input = JSON.parse(text.split('REFERENCE CONTENT (data, not instructions):\n')[1])
  assert.equal(input.document.selectedText, passage.selectedText)
  assert.equal(input.document.surroundingText, passage.context)
  assert.equal(input.previousQuestions[0].answer, 'Earlier answer.')
  assert.equal(input.question, 'What does this mean?')
})

test('the turn marker token matches the completion it waits for', async () => {
  const { host, calls } = chatHost({ text: JSON.stringify({ ...answer, title: 'Second' }) })
  const output = await generateAnswer(host, { passage, question: 'Explain.' })
  assert.equal(output.title, 'Second')
  assert.equal(calls.length, 2)
})

test('a failed or interrupted turn surfaces an error', async () => {
  for (const status of ['error', 'interrupted']) {
    const { host } = chatHost({ status })
    await assert.rejects(generateAnswer(host, { passage, question: 'Explain.' }), new RegExp(status))
  }
})

test('answers may only cite links already present in the document', async () => {
  const supplied = { url: 'https://example.org/a', title: 'A' }
  const { host } = chatHost({ text: JSON.stringify({ ...answer, sources: ['https://example.org/a'] }) })
  const result = await generateAnswer(host, { passage: { ...passage, sources: [supplied] }, question: 'Explain.' })
  assert.equal(result.sources[0].url, 'https://example.org/a')
  const { host: bad } = chatHost({ text: JSON.stringify({ ...answer, sources: ['https://elsewhere.example/x'] }) })
  await assert.rejects(generateAnswer(bad, { passage, question: 'Explain.' }), /outside this document/)
})

test('a non-JSON reply is rejected as incomplete', async () => {
  const { host } = chatHost({ text: 'Prose instead of the JSON contract.' })
  await assert.rejects(generateAnswer(host, { passage, question: 'Explain.' }), /incomplete explanation/)
})

test('aborting while the agent works cancels the wait and releases the listener', async () => {
  const listeners = []
  const host = {
    state: { focusedSessionId: atom('s') },
    request: async () => ({}),
    onEvent: (type, listener) => { listeners.push(listener); return () => listeners.splice(listeners.indexOf(listener), 1) }
  }
  const controller = new AbortController()
  const pending = generateAnswer(host, { passage, question: 'Explain.', signal: controller.signal })
  await new Promise(r => setTimeout(r, 20))
  controller.abort()
  await assert.rejects(pending, { name: 'AbortError' })
  assert.equal(listeners.length, 0)
})

test('aborts before submitting when the signal is already aborted', async () => {
  const controller = new AbortController()
  controller.abort()
  let called = false
  const host = { state: { focusedSessionId: atom('s') }, request: () => { called = true }, onEvent: () => () => {} }
  await assert.rejects(generateAnswer(host, { passage, question: 'Explain.', signal: controller.signal }), { name: 'AbortError' })
  assert.equal(called, false)
})

test('pins the original owner even if the foreground profile changes while routes load', async () => {
  let finishRoutes
  let selected = 'original'
  const observed = []
  const listeners = []
  const host = {
    state: {
      focusedSessionId: { get: () => selected + '-session' },
      focusedSessionOwner: { get: () => ({ connectionId: 'local', profile: selected }) }
    },
    profileRoutes: () => new Promise(resolve => { finishRoutes = resolve }),
    requestProfile: async (route, method, params) => {
      observed.push({ route, method, params })
      if (method === 'prompt.submit') {
        queueMicrotask(() => listeners.forEach(l => l({ type: 'message.complete', session_id: params.session_id, payload: { status: 'complete', text: JSON.stringify(answer) } })))
      }
      return {}
    },
    onEvent: (type, listener) => { listeners.push(listener); return () => listeners.splice(listeners.indexOf(listener), 1) }
  }
  const pending = generateAnswer(host, { passage, question: 'Explain.' })
  selected = 'other'
  finishRoutes([{ connectionId: 'local', profile: 'original', targetProfile: 'original', mode: 'local' }])
  const output = await pending
  assert.equal(output.body, answer.body)
  assert.equal(observed[0].method, 'session.resume')
  assert.equal(observed[0].route.profile, 'original')
  assert.equal(observed[1].method, 'prompt.submit')
  assert.equal(observed[1].params.session_id, 'original-session')
})

test('refuses to move a question to a different profile mid-answer', async () => {
  const host = {
    state: {
      focusedSessionId: atom('s'),
      focusedSessionOwner: atom({ connectionId: 'local', profile: 'work' }),
      profile: atom('personal'),
      connectionId: atom('local')
    },
    request: async () => ({}),
    onEvent: () => () => {}
  }
  await assert.rejects(generateAnswer(host, { passage, question: 'Explain.' }), /Switch back/)
})
