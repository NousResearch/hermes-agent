/**
 * Regression for #20665: the ▸/▾ section header is one shared `Chevron`, so
 * the banner accordion and the tool-trail sections cannot drift apart again.
 */
import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import { stripAnsi } from '@hermes/shared/ansi'
import React, { type ReactElement } from 'react'
import { describe, expect, it } from 'vitest'

import { Accordion, Chevron } from '../components/accordion.js'
import { DEFAULT_THEME } from '../theme.js'

const renderRaw = (node: ReactElement) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 80, isTTY: false, rows: 10 })
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

  return output.trim()
}

const renderText = (node: ReactElement) => stripAnsi(renderRaw(node))

describe('Chevron', () => {
  it('reflects open state and carries count + suffix', () => {
    const props = { count: 3, onClick: () => {}, suffix: 'in 2 categories', t: DEFAULT_THEME, title: 'Skills' }

    const closed = renderText(<Chevron {...props} open={false} />)
    const open = renderText(<Chevron {...props} open />)

    expect(closed).toContain('▸ Skills (3)  in 2 categories')
    expect(closed).not.toContain('▾')
    expect(open).toContain('▾ Skills (3)  in 2 categories')
    expect(open).not.toContain('▸')
  })

  it('is the header Accordion renders', () => {
    const props = { count: 5, suffix: '12 chars', t: DEFAULT_THEME, title: 'System Prompt' }

    for (const open of [false, true]) {
      const header = renderRaw(<Chevron {...props} bold onClick={() => {}} open={open} tone="muted" />)

      const accordion = renderRaw(
        <Accordion {...props} open={open}>
          {null}
        </Accordion>
      )

      expect(accordion).toBe(header)
    }
  })
})
