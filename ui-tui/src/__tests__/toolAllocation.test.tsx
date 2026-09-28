import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import { stripAnsi } from '@hermes/shared/ansi'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { MessageLine } from '../components/messageLine.js'
import {
  AllocatedToolTrail,
  allocateSettledToolTrailLine,
  flattenToolHeader,
  isSettledToolTrailCandidate
} from '../components/toolAllocation.js'
import { DEFAULT_THEME } from '../theme.js'

const settledRead =
  'read_file(src/components/example.ts) (0.2s) :: lines 1-120\\nsecond detail that should fold ✓'

describe('settled tool row allocation', () => {
  it('flattens embedded newlines in tool headers', () => {
    expect(flattenToolHeader('read_file(foo)\\nbar')).toBe('read_file(foo) bar')
  })

  it('recognizes only finalized result lines', () => {
    expect(isSettledToolTrailCandidate([settledRead])).toBe(true)
    expect(isSettledToolTrailCandidate(['drafting read_file(foo)…'])).toBe(false)
  })

  it('lets a successful settled tool disappear at zero rows', () => {
    expect(allocateSettledToolTrailLine(settledRead, 0)?.rows).toEqual([])
  })

  it('keeps an error visible even when the requested budget is zero', () => {
    const block = allocateSettledToolTrailLine('terminal(make test) :: exit 1 ✗', 0)

    expect(block?.rows).toHaveLength(1)
    expect(block?.rows[0]).toContain('✗ terminal(make test)')
  })

  it('renders a settled read in exactly one semantic row', () => {
    const block = allocateSettledToolTrailLine(settledRead, 1)

    expect(block?.rows).toHaveLength(1)
    expect(block?.rows[0]).toContain('✓ read_file(src/components/example.ts)')
    expect(block?.rows[0]).not.toContain('\\n')
  })

  it('renders the two-row folded-card shape', () => {
    const block = allocateSettledToolTrailLine(settledRead, 2)

    expect(block?.rows).toHaveLength(2)
    expect(block?.rows[0]).toMatch(/^╭─ ✓ /)
    expect(block?.rows[1]).toMatch(/^╰─ /)
    expect(block?.rows[1]).toContain('+1 line')
  })

  it('bounds 3+ rows and reports omitted detail exactly', () => {
    const line = 'terminal(test) :: one\\ntwo\\nthree\\nfour ✓'
    const block = allocateSettledToolTrailLine(line, 3)

    expect(block?.rows).toHaveLength(3)
    expect(block?.rows.at(-1)).toContain('… 3 more lines')
  })
})

describe.each([40, 80, 120])('AllocatedToolTrail at %i columns', columns => {
  it('keeps a one-row settled read on one physical line', () => {
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
      <AllocatedToolTrail lines={[settledRead]} rowsPerTool={1} t={DEFAULT_THEME} />,
      {
        patchConsole: false,
        stderr: stderr as NodeJS.WriteStream,
        stdin: stdin as NodeJS.ReadStream,
        stdout: stdout as NodeJS.WriteStream
      }
    )

    const printable = stripAnsi(output).replace(/\r/g, '')
    const visibleLines = printable.split('\n').filter(Boolean)

    expect(visibleLines).toHaveLength(1)
    expect(visibleLines[0]).toContain('✓ read_file')

    instance.unmount()
    instance.cleanup()
  })
})

describe('MessageLine settled-tool integration', () => {
  it('routes one finalized collapsed tool result through the one-row renderer', () => {
    const stdout = new PassThrough()
    const stdin = new PassThrough()
    const stderr = new PassThrough()
    let output = ''

    Object.assign(stdout, { columns: 80, isTTY: false, rows: 20 })
    Object.assign(stdin, { isTTY: false })
    Object.assign(stderr, { isTTY: false })
    stdout.on('data', chunk => {
      output += chunk.toString()
    })

    const instance = renderSync(
      <MessageLine
        cols={80}
        msg={{ kind: 'trail', role: 'system', text: '', tools: [settledRead] }}
        t={DEFAULT_THEME}
      />,
      {
        patchConsole: false,
        stderr: stderr as NodeJS.WriteStream,
        stdin: stdin as NodeJS.ReadStream,
        stdout: stdout as NodeJS.WriteStream
      }
    )

    const printable = stripAnsi(output).replace(/\r/g, '')
    const visibleLines = printable.split('\n').filter(Boolean)

    expect(visibleLines).toHaveLength(1)
    expect(visibleLines[0]).toContain('✓ read_file')
    expect(printable).not.toContain('Tool calls')

    instance.unmount()
    instance.cleanup()
  })
})
