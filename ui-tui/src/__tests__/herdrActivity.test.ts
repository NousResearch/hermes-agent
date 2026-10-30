import { afterEach, expect, it, vi } from 'vitest'

import { $sessionTabs, openSessionTab, resetSessionTabs } from '../app/sessionTabsStore.js'
import { sessionTabActivity } from '../app/useHerdrActivity.js'
import { createHerdrReporter } from '../lib/herdrAgentState.js'

afterEach(() => resetSessionTabs())
const env = { HERDR_ENV: '1', HERDR_PANE_ID: 'test:pane', HERDR_BIN_PATH: '/herdr', HERDR_SOCKET_PATH: '/test/socket' }

it('reports work in non-focused tabs and not in another directory; current idle state cannot hide it', () => {
  openSessionTab({ id: 'a', cwd: '/own', running: false })
  openSessionTab({ id: 'b', cwd: '/own', running: true })
  openSessionTab({ id: 'foreign', cwd: '/other', running: true })
  expect(sessionTabActivity($sessionTabs.get(), '/own', 'a', false, false)).toBe('working')
  expect(sessionTabActivity($sessionTabs.get(), '/own', 'b', false, false)).toBe('idle')
  expect(sessionTabActivity($sessionTabs.get(), '/own', 'a', false, true)).toBe('blocked')
  expect(sessionTabActivity($sessionTabs.get(), '/own', 'a', false, false, false)).toBe('unknown')
})

it('does nothing outside a fully identified Herdr pane', async () => {
  const run = vi.fn(async () => {})

  for (const context of [{}, { HERDR_ENV: '1' }, { ...env, HERDR_ENV: '0' }]) {
    const reporter = createHerdrReporter(context, run)
    reporter.update({ state: 'working' })
    await reporter.close()
  }

  expect(run).not.toHaveBeenCalled()
})

it('serializes and coalesces state reports, increases sequence values, and releases only its own source', async () => {
  const calls: string[][] = []
  let finish!: () => void

  const run = vi.fn((_file: string, args: string[]) => {
    calls.push(args)

    return calls.length === 1
      ? new Promise<void>(resolve => {
          finish = resolve
        })
      : Promise.resolve()
  })

  const reporter = createHerdrReporter(env, run)
  reporter.update({ state: 'idle', sessionId: 'a' })
  reporter.update({ state: 'working', sessionId: 'a' })
  reporter.update({ state: 'blocked', sessionId: 'a' })
  expect(calls).toHaveLength(1)
  finish()
  await vi.waitFor(() => expect(calls).toHaveLength(2))
  expect(calls[1]).toContain('blocked')
  expect(calls.flat()).not.toContain('working')
  await reporter.close()
  expect(calls[2]!.slice(0, 3)).toEqual(['pane', 'release-agent', 'test:pane'])
  expect(calls[2]).toContain('hermes-tui')
  const sequences = calls.map(args => Number(args[args.indexOf('--seq') + 1]))
  expect(sequences[1]).toBeGreaterThan(sequences[0]!)
  expect(sequences[2]).toBeGreaterThan(sequences[1]!)
})

it('ignores unavailable Herdr failures rather than breaking the chat', async () => {
  const reporter = createHerdrReporter(env, async () => {
    throw new Error('offline')
  })

  reporter.update({ state: 'working' })
  await expect(reporter.close()).resolves.toBeUndefined()
})
