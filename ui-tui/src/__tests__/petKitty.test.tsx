import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { PetKitty, PetSprite } from '../components/petSprite.js'

const renderOutput = (element: React.ReactElement) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 40, isTTY: false, rows: 12 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(element, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  instance.unmount()
  instance.cleanup()

  return output
}

describe('PetKitty', () => {
  it('preserves the exact image-id foreground color in rendered placeholders', () => {
    const row = '\u{10eeee}\u{0305}\u{10eeee}'
    const output = renderOutput(<PetKitty color="#00104b" placeholder={[row]} />)

    expect(output).toContain(`\x1b[38;2;0;16;75m${row}\x1b[39m`)
    expect(output).not.toContain('38;5;17m')
  })

  it('leaves ordinary half-block pet rendering on its native Ink path', () => {
    const output = renderOutput(<PetSprite grid={[[[255, 0, 0, 255, 0, 0, 255, 255]]]} />)
    expect(output).toContain('▀')
  })
})
