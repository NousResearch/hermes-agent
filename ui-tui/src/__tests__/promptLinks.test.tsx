import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { expect, it } from 'vitest'

import { SecretPrompt } from '../components/maskedPrompt.js'
import { ApprovalPrompt } from '../components/prompts.js'
import { DEFAULT_THEME } from '../theme.js'

// Minimal raw-mode TTY stdin: the prompt's useInput needs setRawMode to mount.
class FakeInput extends EventEmitter {
  isRaw = false
  isTTY = true
  readableLength = 0
  read = () => null
  ref = () => {}
  setEncoding = () => {}
  setRawMode = (enabled: boolean) => {
    this.isRaw = enabled
  }
  unref = () => {}
}

const paint = (node: React.ReactElement) => {
  const stdout = Object.assign(new PassThrough(), { columns: 120, isTTY: false, rows: 30 })
  const frames: string[] = []
  stdout.on('data', chunk => frames.push(chunk.toString()))

  const app = renderSync(node, {
    exitOnCtrlC: false,
    patchConsole: false,
    stdin: new FakeInput() as never,
    stdout: stdout as never
  })

  app.unmount()

  return frames.join('')
}

it('an approval description URL is an OSC 8 link without the trailing period', () => {
  const out = paint(
    <ApprovalPrompt
      onChoice={() => {}}
      req={{ command: 'curl -fsSL x | sh', description: 'see https://docs.example.dev/keys.' } as never}
      t={DEFAULT_THEME}
    />
  )

  // OSC 8 is ESC ] 8 ; <params> ; <url> BEL; the trailing period stays plain text.
  // eslint-disable-next-line no-control-regex -- the assertion is about raw escape bytes
  expect(out).toMatch(/\u001b\]8;[^;]*;https:\/\/docs\.example\.dev\/keys\u0007/)
  expect(out).not.toMatch(/\]8;[^;]*;https:\/\/docs\.example\.dev\/keys\./)
})

it('a secret prompt shows its help URL as a link', () => {
  const out = paint(
    <SecretPrompt
      envVar="PROBE_KEY"
      help="Create one at https://platform.example.dev/api-keys."
      onSubmit={() => {}}
      prompt="Paste your key"
      t={DEFAULT_THEME}
    />
  )

  // eslint-disable-next-line no-control-regex -- the assertion is about raw escape bytes
  expect(out).toMatch(/\u001b\]8;[^;]*;https:\/\/platform\.example\.dev\/api-keys\u0007/)
})
