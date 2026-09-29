import { PassThrough } from 'stream'

import { renderSync, Text } from '@hermes/ink'
import { stripAnsi } from '@hermes/shared/ansi'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { ComposerRail, composerRailContentColumns } from '../components/composerRail.js'

describe('composerRailContentColumns', () => {
  it.each([40, 80, 120])('spends exactly one column at %i columns', cols => {
    expect(composerRailContentColumns(cols)).toBe(cols - 1)
  })

  it('never returns less than one content column', () => {
    expect(composerRailContentColumns(1)).toBe(1)
    expect(composerRailContentColumns(0)).toBe(1)
  })
})

describe.each([40, 80, 120])('ComposerRail at %i columns', columns => {
  it('adds no vertical rows for a one-line draft', () => {
    const stdout = new PassThrough()
    const stdin = new PassThrough()
    const stderr = new PassThrough()
    let output = ''

    Object.assign(stdout, { columns, isTTY: false, rows: 20 })
    Object.assign(stdin, { isTTY: false })
    Object.assign(stderr, { isTTY: false })
    stdout.on('data', chunk => {
      output += chunk.toString()
    })

    const instance = renderSync(
      <ComposerRail color="#888888" width={columns}>
        <Text>draft</Text>
      </ComposerRail>,
      {
        patchConsole: false,
        stderr: stderr as NodeJS.WriteStream,
        stdin: stdin as NodeJS.ReadStream,
        stdout: stdout as NodeJS.WriteStream
      }
    )

    const lines = stripAnsi(output)
      .replace(/\r/g, '')
      .split('\n')
      .filter(Boolean)

    expect(lines).toHaveLength(1)
    expect(lines[0]?.startsWith('│')).toBe(true)
    expect(lines[0]).toContain('draft')

    instance.unmount()
    instance.cleanup()
  })

  it('keeps every multiline draft row inside one continuous rail', () => {
    const stdout = new PassThrough()
    const stdin = new PassThrough()
    const stderr = new PassThrough()
    let output = ''

    Object.assign(stdout, { columns, isTTY: false, rows: 20 })
    Object.assign(stdin, { isTTY: false })
    Object.assign(stderr, { isTTY: false })
    stdout.on('data', chunk => {
      output += chunk.toString()
    })

    const instance = renderSync(
      <ComposerRail color="#888888" width={columns}>
        <Text>first</Text>
        <Text>second</Text>
      </ComposerRail>,
      {
        patchConsole: false,
        stderr: stderr as NodeJS.WriteStream,
        stdin: stdin as NodeJS.ReadStream,
        stdout: stdout as NodeJS.WriteStream
      }
    )

    const lines = stripAnsi(output)
      .replace(/\r/g, '')
      .split('\n')
      .filter(Boolean)

    expect(lines).toHaveLength(2)
    expect(lines.every(line => line.startsWith('│'))).toBe(true)

    instance.unmount()
    instance.cleanup()
  })
})
