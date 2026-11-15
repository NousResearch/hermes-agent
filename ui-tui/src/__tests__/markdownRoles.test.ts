import { PassThrough } from 'stream'

import { Box, renderSync } from '@hermes/ink'
import chalk from 'chalk'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { Md } from '../components/markdown.js'
import { fromSkin } from '../theme.js'

// The markdown element roles are split so a TITLE colour (headings) does not
// bleed into prose (inline code / links / math). This renders real markdown
// through the real renderer and reads back the foregrounds it emitted, so the
// assertion is about what reaches the terminal, not about the theme object.

const foregrounds = (text: string, t: ReturnType<typeof fromSkin>): string[] => {
  // chalk is a singleton that defaults to level 0 under vitest (no TTY),
  // which would emit no SGR at all and make every assertion vacuous.
  const savedLevel = chalk.level
  chalk.level = 3

  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 80, isTTY: true, rows: 24 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(React.createElement(Box, { width: 70 }, React.createElement(Md, { cols: 68, t, text })), {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  instance.unmount()
  instance.cleanup()
  chalk.level = savedLevel

  const ESC = String.fromCharCode(27)

  return [...output.matchAll(new RegExp(`${ESC}\\[38;2;(\\d+);(\\d+);(\\d+)m`, 'g'))].map(
    m => `#${[m[1], m[2], m[3]].map(n => Number(n).toString(16).padStart(2, '0')).join('')}`
  )
}

describe('markdown element roles', () => {
  const split = fromSkin(
    {
      background: '#16121f',
      ui_accent: '#a277ff',
      ui_text: '#e6e1f5',
      markdown_heading: '#a277ff',
      markdown_code: '#61ffca',
      markdown_link: '#9dff65',
      markdown_math: '#ffca85'
    },
    {}
  )

  it('paints a heading with the heading colour', () => {
    expect(foregrounds('## A heading', split)).toContain('#a277ff')
  })

  it('paints inline code with the code colour, not the heading colour', () => {
    const runs = foregrounds('a `code` span', split)

    expect(runs).toContain('#61ffca')
    expect(runs).not.toContain('#a277ff')
  })

  it('keeps prose in the body ink', () => {
    expect(foregrounds('plain prose', split)).toContain('#e6e1f5')
  })

  it('falls back to the accent when a skin does not split the roles', () => {
    const unsplit = fromSkin({ background: '#16121f', ui_accent: '#a277ff', ui_text: '#e6e1f5' }, {})

    // Headings AND inline code both take the accent: the pre-split look.
    expect(foregrounds('## A heading', unsplit)).toContain('#a277ff')
    expect(foregrounds('a `code` span', unsplit)).toContain('#a277ff')
  })
})
