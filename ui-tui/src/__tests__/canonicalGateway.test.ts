import { expect, it } from 'vitest'

import { canonicalRequest, canonicalResult, localCreationOptions, sharedControlParams } from '../canonicalGateway.js'

it('retains prepared identity and rejects unsupported TUI launch policy instead of impersonating CLI', () => {
  const contract = { sources: ['tui'], parameters: ['request_id', 'source', 'model', 'cwd', 'toolsets'] }
  expect(canonicalRequest('session.create', { request_id: 'fresh', model: 'local-model', cwd: '/tmp/project' }, contract)).toEqual({ method: 'session.create', params: { request_id: 'fresh', source: 'tui', model: 'local-model', cwd: '/tmp/project' } })
  expect(() => canonicalRequest('session.create', {}, { sources: ['cli'], parameters: [] })).toThrow('tui')
  expect(() => canonicalRequest('session.create', { skills: ['test'] }, contract)).toThrow('skills')
  expect(canonicalRequest('prompt.submit', { session_id: 'sid', submission_id: 'prepared-id', text: 'hello', queued: true }, contract).params).toEqual({ session_id: 'sid', input_id: 'prepared-id', text: 'hello', queued: true })
  expect(sharedControlParams({ sharedControl: { session_id: 'sid', execution_generation: 9, prompt_id: 'approval-9' } })).toEqual({ session_id: 'sid', execution_generation: 9, prompt_id: 'approval-9' })
  const receipt = canonicalResult('prompt.submit', { admission_id: 'server-admission', ref: { profile_id: '/tmp/profile', session_id: 'sid' }, status: 'queued' }, { input_id: 'prepared-id' })
  expect(receipt).toMatchObject({ admission_id: 'server-admission', input_id: 'prepared-id', target_profile_home: '/tmp/profile', target_session_id: 'sid' })
})

it('rebuilds --max-turns from the launcher environment as the integer the session policy requires', () => {
  const options = localCreationOptions({ HERMES_TUI_MAX_TURNS: '5', HERMES_MODEL: 'local-model' } as NodeJS.ProcessEnv)
  expect(options.max_turns).toBe(5)
  expect(localCreationOptions({} as NodeJS.ProcessEnv)).not.toHaveProperty('max_turns')
})
