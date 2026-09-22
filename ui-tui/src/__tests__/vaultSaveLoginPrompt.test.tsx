import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import stripAnsi from 'strip-ansi'
import { expect, it, vi } from 'vitest'

import { VaultSaveLoginPrompt } from '../components/vaultSaveLoginPrompt.js'
import { DEFAULT_THEME } from '../theme.js'

it('collects an identifier then a masked password and returns one vault-only payload', async () => {
  const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 20 })
  const stdin = Object.assign(new PassThrough(), { isTTY: true, ref: () => {}, setRawMode: () => {}, unref: () => {} })
  const stderr = Object.assign(new PassThrough(), { isTTY: false })
  const onSubmit = vi.fn()
  let output = ''

  stdout.on('data', chunk => {
    output += stripAnsi(chunk.toString())
  })

  const view = renderSync(
    <VaultSaveLoginPrompt
      onSubmit={onSubmit}
      origin="https://example.test"
      site="Example"
      t={DEFAULT_THEME}
    />,
    {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    }
  )

  try {
    await vi.waitFor(() => expect(output).toContain('Username >'))
    stdin.write('alice')
    await vi.waitFor(() => expect(output).toContain('alice'))
    stdin.write('\r')
    await vi.waitFor(() => expect(output).toContain('Password >'))

    output = ''
    stdin.write('test-secret')
    await new Promise(resolve => setTimeout(resolve, 25))
    stdin.write('\r')

    await vi.waitFor(() =>
      expect(onSubmit).toHaveBeenCalledWith(JSON.stringify({ identifier: 'alice', password: 'test-secret' }))
    )
    stdin.write('\r')
    await new Promise(resolve => setTimeout(resolve, 25))
    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(output).not.toContain('test-secret')
  } finally {
    view.unmount()
    view.cleanup()
  }
})
