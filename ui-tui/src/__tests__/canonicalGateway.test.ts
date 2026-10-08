import { expect, it } from 'vitest'

import { canonicalRequest, canonicalResult, localCreationOptions, sharedControlParams } from '../canonicalGateway.js'

it.each(['HERMES_SAFE_MODE', 'HERMES_IGNORE_USER_CONFIG'])('carries %s on the launching session policy', flag => {
  const field = flag === 'HERMES_SAFE_MODE' ? 'safe_mode' : 'ignore_user_config'
  const options = localCreationOptions({ [flag]: '1' } as NodeJS.ProcessEnv)
  expect(options[field]).toBe(true)
  expect(localCreationOptions({ [flag]: '0' } as NodeJS.ProcessEnv)).not.toHaveProperty(field)
  const contract = { sources: ['tui'], parameters: ['source', 'request_id', field] }
  expect(canonicalRequest('session.create', { ...options, request_id: 'owned' }, contract).params[field]).toBe(true)
})

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
  // The pre-gateway "unlimited" spellings reach session.create as values the policy reads, never NaN/null.
  const wire = (value: string) => JSON.parse(JSON.stringify(localCreationOptions({ HERMES_TUI_MAX_TURNS: value } as NodeJS.ProcessEnv))).max_turns
  expect(['0', '-1', 'none', 'unlimited'].map(wire)).toEqual([0, -1, 'none', 'unlimited'])
})

it('carries `hermes --tui --yolo` (HERMES_YOLO_MODE) onto session.create as the frozen launch flag', () => {
  expect(localCreationOptions({ HERMES_YOLO_MODE: '1' } as NodeJS.ProcessEnv).yolo).toBe(true)
  expect(localCreationOptions({ HERMES_YOLO_MODE: '0' } as NodeJS.ProcessEnv)).not.toHaveProperty('yolo')
  expect(localCreationOptions({} as NodeJS.ProcessEnv)).not.toHaveProperty('yolo')
})

it('carries `hermes --tui --ignore-rules` (HERMES_IGNORE_RULES) onto session.create as the frozen launch flag', () => {
  expect(localCreationOptions({ HERMES_IGNORE_RULES: '1' } as NodeJS.ProcessEnv).ignore_rules).toBe(true)
  expect(localCreationOptions({ HERMES_IGNORE_RULES: '0' } as NodeJS.ProcessEnv)).not.toHaveProperty('ignore_rules')
  expect(localCreationOptions({} as NodeJS.ProcessEnv)).not.toHaveProperty('ignore_rules')
})
