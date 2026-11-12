import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { Md } from '../components/markdown.js'
import { DEFAULT_THEME } from '../theme.js'

const renderMarkdown = (text: string) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 80, rows: 24, isTTY: true })
  Object.assign(stdin, { isTTY: true, setRawMode: () => {} })
  Object.assign(stderr, { isTTY: true })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const view = renderSync(<Md cols={80} t={DEFAULT_THEME} text={text} />, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  view.unmount()
  view.cleanup()

  return output
}

describe('markdown hyperlink terminal output', () => {
  it('does not let a model-authored URL inject a title OSC after BEL', () => {
    const output = renderMarkdown('[click](https://example.com/\x07\x1b]0;OWNED\x07)')

    expect(output).toContain('https://example.com/]0;OWNED')
    expect(output).not.toContain('\x1b]0;OWNED')
  })

  it('does not let a model-authored URL inject a clipboard OSC after ST', () => {
    const output = renderMarkdown('[click](https://example.com/\x1b\\\x1b]52;c;SGVsbG8=\x07)')

    expect(output).not.toContain('\x1b]52;c;SGVsbG8=')
  })
})
