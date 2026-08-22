import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { Banner } from '../components/branding.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

const renderBanner = (compact: boolean, maxWidth = 140) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 140, isTTY: false, rows: 20 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const theme = { ...DEFAULT_THEME, bannerLogo: 'CUSTOM LOGO' }
  const instance = renderSync(<Banner compact={compact} maxWidth={maxWidth} t={theme} />, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  try {
    return stripAnsi(output)
  } finally {
    instance.unmount()
    instance.cleanup()
  }
}

describe('Banner compact preference', () => {
  it('forces the compact tier on a terminal wide enough for the full logo', () => {
    const frame = renderBanner(true)

    expect(frame).toContain('Messenger of the Digital Gods')
    expect(frame).not.toContain('CUSTOM LOGO')
  })

  it('preserves responsive full-logo selection when compact is disabled', () => {
    expect(renderBanner(false)).toContain('CUSTOM LOGO')
  })

  it('keeps the explicit compact banner visible below the responsive hide threshold', () => {
    expect(renderBanner(true, 30)).toContain('Hermes Agent')
    expect(renderBanner(false, 30)).toBe('')
  })
})
