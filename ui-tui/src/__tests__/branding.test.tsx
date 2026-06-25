import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { Banner } from '../components/branding.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME, fromSkin } from '../theme.js'

// Render a node against a mocked 80-col stdout and return the plain text.
const renderPlain = (node: React.ReactNode): string => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 80, isTTY: false, rows: 24 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(node, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  instance.unmount()
  instance.cleanup()

  return stripAnsi(output)
}

describe('Banner branding', () => {
  it('renders the bundled vendor + tagline by default', () => {
    const out = renderPlain(<Banner maxWidth={80} t={DEFAULT_THEME} />)

    expect(out).toContain('Nous Research')
    expect(out).toContain('Messenger of the Digital Gods')
  })

  it('omits the vendor and tagline when a skin sets both empty', () => {
    const t = fromSkin({}, { tagline: '', vendor_label: '' })
    const out = renderPlain(<Banner maxWidth={80} t={t} />)

    expect(out).not.toContain('Nous Research')
    expect(out).not.toContain('Messenger')
  })

  it('keeps the tagline when only the vendor is empty', () => {
    const t = fromSkin({}, { vendor_label: '' })
    const out = renderPlain(<Banner maxWidth={80} t={t} />)

    expect(out).not.toContain('Nous Research')
    expect(out).toContain('Messenger of the Digital Gods')
  })

  it('renders a custom vendor credit', () => {
    const t = fromSkin({}, { tagline: 'Local Agent', vendor_label: 'Acme Labs' })
    const out = renderPlain(<Banner maxWidth={80} t={t} />)

    expect(out).toContain('Acme Labs')
    expect(out).toContain('Local Agent')
    expect(out).not.toContain('Nous Research')
  })
})
