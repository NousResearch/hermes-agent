import { PassThrough } from 'node:stream'

import { Box, renderSync } from '@hermes/ink'
import React from 'react'
import stripAnsi from 'strip-ansi'
import { beforeEach, expect, it, vi } from 'vitest'

import {
  activatePluginCard,
  getPluginCardState,
  publishPluginCard,
  resetPluginCards
} from '../app/pluginCardStore.js'
import { PluginCardSurface } from '../components/pluginCard.js'
import { DEFAULT_THEME } from '../theme.js'

const card = (id: string, title = id) => ({
  actions: [{ args: id, command: 'inspect', id: 'inspect', label: 'Inspect' }],
  body: Array.from({ length: 20 }, (_, i) => `line ${i}`).join('\n'),
  id,
  plugin_id: 'build-tools',
  plugin_name: 'build-tools',
  title
})

beforeEach(() => resetPluginCards())

it('queues unrelated ambient cards without taking focus', () => {
  publishPluginCard('sid-a', card('first'))
  publishPluginCard('sid-a', card('second'))

  expect(getPluginCardState()).toMatchObject({ activeKey: null, cards: [{ card: { id: 'first' } }, { card: { id: 'second' } }] })
})

it('opens explicitly and drops cards from another session', () => {
  publishPluginCard('sid-a', card('first'))
  activatePluginCard('sid-a', card('menu'))
  publishPluginCard('sid-b', card('other'))

  expect(getPluginCardState()).toMatchObject({ activeKey: null, sessionId: 'sid-b' })
  expect(getPluginCardState().cards).toHaveLength(1)
})

it('renders text and a next card returned by plugin actions', async () => {
  activatePluginCard('sid-a', card('menu', 'Build menu'))
  const next = { ...card('next', 'Next menu'), body: 'Next step.' }

  const dispatch = vi
    .fn()
    .mockResolvedValueOnce({ kind: 'text', text: 'Inspection opened.' })
    .mockResolvedValueOnce({ card: next, kind: 'card' })

  const stdout = Object.assign(new PassThrough(), { columns: 80, rows: 18, isTTY: false })
  const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: () => {}, ref: () => {}, unref: () => {} })
  let output = ''

  stdout.on('data', chunk => (output += stripAnsi(chunk.toString())))

  const view = renderSync(
    <Box height={18}>
      <PluginCardSurface dispatch={dispatch} sessionId="sid-a" theme={DEFAULT_THEME} />
    </Box>,
    {
      stdout: stdout as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stderr: new PassThrough() as unknown as NodeJS.WriteStream,
      patchConsole: false
    }
  )

  try {
    stdin.write('\r')
    await vi.waitFor(() => expect(output).toContain('Inspection opened.'))
    stdin.write('\r')
    await vi.waitFor(() => expect(output).toContain('Next menu'))
  } finally {
    view.unmount()
    view.cleanup()
  }
})

it('navigates actions, suppresses duplicate submit, reports failure, scrolls, and closes on escape', async () => {
  activatePluginCard('sid-a', card('menu', 'Build menu'))
  let reject!: (reason: Error) => void
  const dispatch = vi.fn(() => new Promise<never>((_resolve, rej) => (reject = rej)))
  const close = vi.fn()
  const stdout = Object.assign(new PassThrough(), { columns: 80, rows: 18, isTTY: false })
  const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: () => {}, ref: () => {}, unref: () => {} })
  let output = ''
  stdout.on('data', chunk => (output += stripAnsi(chunk.toString())))

  const view = renderSync(
    <Box height={18}>
      <PluginCardSurface dispatch={dispatch} onClose={close} sessionId="sid-a" theme={DEFAULT_THEME} />
    </Box>,
    {
      stdout: stdout as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stderr: new PassThrough() as unknown as NodeJS.WriteStream,
      patchConsole: false
    }
  )

  try {
    await vi.waitFor(() => expect(output).toContain('Build menu'))
    stdin.write('j')
    await vi.waitFor(() => expect(output).toContain('line 19'))
    stdin.write('\r')
    await vi.waitFor(() => expect(dispatch).toHaveBeenCalledTimes(1))
    stdin.write('\r')
    await new Promise(resolve => setTimeout(resolve, 20))
    expect(dispatch).toHaveBeenCalledTimes(1)
    reject(new Error('plugin went away'))
    await vi.waitFor(() => expect(output).toContain('plugin went away'))
    stdin.write('\x1b')
    await vi.waitFor(() => expect(close).toHaveBeenCalledTimes(1))
  } finally {
    view.unmount()
    view.cleanup()
  }
})
