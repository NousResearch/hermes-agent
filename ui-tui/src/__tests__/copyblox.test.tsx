/**
 * Tests for `src/components/copyblox.tsx` — CopyBlox React component.
 */

import { PassThrough } from 'stream'

import { Box, renderSync, Text } from '@hermes/ink'
import React from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const { copyText, noSelectClickHandlers } = vi.hoisted(() => ({
  copyText: vi.fn(),
  noSelectClickHandlers: [] as Array<unknown>
}))

vi.mock('../lib/copyText.js', () => ({ copyText }))

vi.mock('@hermes/ink', async importOriginal => {
  const actual = await importOriginal<typeof import('@hermes/ink')>()

  return {
    ...actual,
    NoSelect: (props: React.ComponentProps<typeof actual.NoSelect>) => {
      noSelectClickHandlers.push(props.onClick)

      return React.createElement(actual.NoSelect, props)
    }
  }
})

import { CopyBlox } from '../components/copyblox.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

const BEL = String.fromCharCode(7)
const ESC = String.fromCharCode(27)
const CSI_RE = new RegExp(`${ESC}\\[[0-?]*[ -/]*[@-~]`, 'g')
const OSC_RE = new RegExp(`${ESC}\\][\\s\\S]*?(?:${BEL}|${ESC}\\\\)`, 'g')

const renderPlain = (node: React.ReactNode) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 80, isTTY: false, rows: 24 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => { output += chunk.toString() })

  const instance = renderSync(node, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  instance.unmount()
  instance.cleanup()

  return output
    .replace(OSC_RE, '')
    .split('\n')
    .map(line => stripAnsi(line).replace(CSI_RE, '').trimEnd())
}

describe('CopyBlox', () => {
  beforeEach(() => {
    copyText.mockReset()
    noSelectClickHandlers.length = 0
  })

  it('renders language label and idle COPY button for empty block', () => {
    const lines = renderPlain(
      React.createElement(CopyBlox, { closed: true, language: 'python', rawContent: '', theme: DEFAULT_THEME, cols: 80 })
    )

    const output = lines.join('\n')

    expect(output).toContain('python')
  })

  it('renders children inside the code body', () => {
    const lines = renderPlain(
      React.createElement(
        CopyBlox,
        { closed: true, language: 'ts', rawContent: 'x = 1', theme: DEFAULT_THEME, cols: 80 },
        React.createElement(Box, null,
          React.createElement(Text, null, 'x = 1')
        )
      )
    )

    // Rendered output should contain the code text
    expect(lines.length).toBeGreaterThan(1)
  })

  it('defaults language to "text" when empty', () => {
    const lines = renderPlain(
      React.createElement(CopyBlox, { closed: true, language: '', rawContent: 'content', theme: DEFAULT_THEME, cols: 80 })
    )

    const output = lines.join('\n')

    expect(output).toContain('text')
  })

  it('renders borders with correct characters', () => {
    const output = renderPlain(
      React.createElement(CopyBlox, { closed: true, language: 'py', rawContent: '', theme: DEFAULT_THEME, cols: 80 })
    ).join('\n')

    // Top border should contain ┌ and bottom border should contain ┘
    expect(output).toMatch(/┌/)
    expect(output).toMatch(/┘/)
  })

  it('shows idle 3×2 copy icon by default', () => {
    const lines = renderPlain(
      React.createElement(CopyBlox, { closed: true, language: 'py', rawContent: '', theme: DEFAULT_THEME, cols: 80 })
    )

    const output = lines.join('\n')

    expect(output).toContain('⧉⧉⧉')
  })

  it('does not register a clickable copy control for an unclosed streaming fence', () => {
    const output = renderPlain(
      React.createElement(CopyBlox, { closed: false, language: 'py', rawContent: 'partial code', theme: DEFAULT_THEME, cols: 80 })
    ).join('\n')

    expect(output).toContain('⟳')
    expect(output).not.toContain('⧉⧉⧉')
    expect(noSelectClickHandlers).toEqual([undefined])
    expect(copyText).not.toHaveBeenCalled()
  })

  it('renders multi-line content correctly', () => {
    const codeLines = ['def hello():', '    print("hello")', '']
    const rawContent = codeLines.join('\n')

    const lines = renderPlain(
      React.createElement(
        CopyBlox,
        { closed: true, language: 'python', rawContent: rawContent, theme: DEFAULT_THEME, cols: 80 },
        React.createElement(Box, { flexDirection: 'column' },
          ...codeLines.map(line =>
            React.createElement(Text, { key: line }, line)
          )
        )
      )
    )

    // Should have more lines than just the border
    expect(lines.length).toBeGreaterThan(3)
  })

  it('does not throw with special characters in rawContent', () => {
    const specialContent = '\t\ttabbed\n  spaces  \nunicode: ñ → [ñ]\n'

    expect(() => {
      renderPlain(
        React.createElement(
          CopyBlox,
          { closed: true, language: 'text', rawContent: specialContent, theme: DEFAULT_THEME, cols: 80 },
          React.createElement(Box, { flexDirection: 'column' },
            React.createElement(Text, null, specialContent)
          )
        )
      )
    }).not.toThrow()
  })
})
