import { PassThrough } from 'stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { afterEach, describe, expect, it } from 'vitest'

import { patchUiState } from '../app/uiStore.js'
import { widgetSdk } from '../sdk/userWidgets.js'

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

const mount = (element: React.ReactElement) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 60, isTTY: false, rows: 6 })
  Object.assign(stdin, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(element, {
    patchConsole: false,
    stderr: stdout as unknown as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream
  })

  return {
    output: () => output,
    stop: () => {
      instance.unmount()
      instance.cleanup()
    }
  }
}

afterEach(() => patchUiState({ info: null, sid: null, storedSid: null }))

describe('widget SDK session access', () => {
  it('getSession reports the durable id and the session folder', () => {
    patchUiState({ sid: 'runtime-1', storedSid: null })
    expect(widgetSdk.getSession().id).toBeNull()

    patchUiState({ info: { cwd: '/work/repo' } as never, sid: 'runtime-1', storedSid: '20261021_120000_abcd' })
    expect(widgetSdk.getSession()).toEqual({ cwd: '/work/repo', id: '20261021_120000_abcd' })
  })

  it('useSession re-renders a widget when the session changes', async () => {
    patchUiState({ storedSid: 'first-session' })

    const Probe = () => React.createElement(Text, null, `sid=${widgetSdk.useSession().id}`)
    const view = mount(React.createElement(Probe))

    try {
      await delay(20)
      expect(view.output()).toContain('sid=first-session')

      patchUiState({ storedSid: 'second-session' })
      await delay(60)
      expect(view.output()).toContain('second-session')
    } finally {
      view.stop()
    }
  })

  it('Link wraps its label in an OSC 8 hyperlink', async () => {
    const url = 'https://github.com/NousResearch/hermes-agent/pull/1'
    const view = mount(React.createElement(widgetSdk.Link, { url }, 'PR#1'))

    try {
      await delay(20)
      expect(view.output()).toMatch(new RegExp(`\\x1b\\]8;[^;]*;${url.replace(/[.\\/]/g, '\\$&')}\\x07`))
      expect(view.output()).toContain('PR#1')
    } finally {
      view.stop()
    }
  })
})
