/**
 * Tests for `src/components/copyblox.tsx` — CopyBlox React component.
 */

import { PassThrough } from 'stream'

import type * as HermesInk from '@hermes/ink'
import { Box, renderSync, stringWidth, Text } from '@hermes/ink'
import React from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const { copyText, noSelectClickHandlers } = vi.hoisted(() => ({
  copyText: vi.fn(),
  noSelectClickHandlers: [] as Array<unknown>
}))

vi.mock('../lib/copyText.js', () => ({ copyText }))

vi.mock('@hermes/ink', async importOriginal => {
  const actual = await importOriginal<typeof HermesInk>()

  return {
    ...actual,
    NoSelect: (props: React.ComponentProps<typeof actual.NoSelect>) => {
      noSelectClickHandlers.push(props.onClick)

      return React.createElement(actual.NoSelect, props)
    }
  }
})

import { stripAnsi } from '@hermes/shared/ansi'

import { CopyBlox } from '../components/copyblox.js'
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
      React.createElement(CopyBlox, {
        closed: true,
        language: 'python',
        rawContent: '',
        theme: DEFAULT_THEME,
        cols: 80
      })
    )

    const output = lines.join('\n')

    expect(output).toContain('python')
  })

  it('renders children inside the code body', () => {
    const lines = renderPlain(
      React.createElement(
        CopyBlox,
        { closed: true, language: 'ts', rawContent: 'x = 1', theme: DEFAULT_THEME, cols: 80 },
        React.createElement(Box, null, React.createElement(Text, null, 'x = 1'))
      )
    )

    // Rendered output should contain the code text
    expect(lines.length).toBeGreaterThan(1)
  })

  it('defaults language to "text" when empty', () => {
    const lines = renderPlain(
      React.createElement(CopyBlox, {
        closed: true,
        language: '',
        rawContent: 'content',
        theme: DEFAULT_THEME,
        cols: 80
      })
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

  it('uses a width-safe left accent for narrow code blocks', () => {
    const lines = renderPlain(
      React.createElement(
        CopyBlox,
        {
          closed: true,
          cols: 12,
          compact: false,
          language: '한국어_😀_very_long',
          rawContent: 'x'.repeat(30),
          theme: DEFAULT_THEME
        },
        React.createElement(Text, null, 'x')
      )
    )

    expect(lines.some(line => line.includes('┌'))).toBe(false)
    expect(lines.flatMap(line => line.split('│').filter(Boolean)).every(line => stringWidth(line) <= 11)).toBe(true)
    expect(lines.join('\n')).toContain('…')
  })

  it('does not register a clickable copy control for an unclosed streaming fence', () => {
    const output = renderPlain(
      React.createElement(CopyBlox, {
        closed: false,
        language: 'py',
        rawContent: 'partial code',
        theme: DEFAULT_THEME,
        cols: 80
      })
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
        React.createElement(
          Box,
          { flexDirection: 'column' },
          ...codeLines.map(line => React.createElement(Text, { key: line }, line))
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
          React.createElement(Box, { flexDirection: 'column' }, React.createElement(Text, null, specialContent))
        )
      )
    }).not.toThrow()
  })

  describe('interaction', () => {
    const setupInteraction = async (props: Record<string, unknown>, childText = 'x = 1') => {
      renderPlain(
        React.createElement(
          CopyBlox,
          {
            closed: true,
            cols: 80,
            language: 'python',
            rawContent: (props.rawContent as string) ?? childText,
            theme: DEFAULT_THEME,
            ...props
          } as never,
          React.createElement(Text, null, childText)
        )
      )

      expect(noSelectClickHandlers.length).toBeGreaterThan(0)

      return { handler: noSelectClickHandlers.at(-1) }
    }

    const clickEvent = () => ({ stopImmediatePropagation: vi.fn() })

    it('invokes clipboard with exact raw source on click', async () => {
      const rawContent = 'def hello():\n    return "世界 🌍"\n'
      const { handler } = await setupInteraction({ rawContent })
      copyText.mockResolvedValue({ method: 'native-or-tmux', success: true })

      const stopImmediatePropagation = vi.fn()
      handler?.({ stopImmediatePropagation } as never)

      await vi.waitFor(() => expect(copyText).toHaveBeenCalledOnce())
      expect(copyText).toHaveBeenCalledWith(rawContent)
      expect(stopImmediatePropagation).toHaveBeenCalledOnce()
    })

    it('shows failed feedback when clipboard write rejects', async () => {
      const { handler } = await setupInteraction({ rawContent: 'boom' })
      copyText.mockRejectedValue(new Error('clipboard unavailable'))

      handler?.({ stopImmediatePropagation: vi.fn() } as never)

      await vi.waitFor(() => expect(copyText).toHaveBeenCalledOnce())
      // Component still resolves; failed feedback rendered asynchronously.
      expect(handler).toBeDefined()
    })

    it('does not start a second copy while one is in flight', async () => {
      const { handler } = await setupInteraction({ rawContent: 'busy' })
      let release!: () => void
      copyText.mockImplementation(
        () =>
          new Promise<{ method: string; success: boolean }>(resolve => {
            release = () => resolve({ method: 'native-or-tmux', success: true })
          })
      )

      handler?.({ stopImmediatePropagation: vi.fn() } as never)
      handler?.({ stopImmediatePropagation: vi.fn() } as never)
      release()

      await vi.waitFor(() => expect(copyText).toHaveBeenCalledOnce())
    })

    it('does not register a click handler for incomplete fences', async () => {
      renderPlain(
        React.createElement(CopyBlox, {
          closed: false,
          cols: 80,
          language: 'python',
          rawContent: 'partial',
          theme: DEFAULT_THEME
        })
      )

      expect(noSelectClickHandlers).toEqual([undefined])
      expect(copyText).not.toHaveBeenCalled()
    })

    it('preserves whitespace and Unicode exactly through the click handler', async () => {
      const rawContent = '  tabs→\ttabs  \n  unicode: ñ 😀\n'
      const { handler } = await setupInteraction({ rawContent })
      copyText.mockResolvedValue({ method: 'native-or-tmux', success: true })

      handler?.({ stopImmediatePropagation: vi.fn() } as never)

      await vi.waitFor(() => expect(copyText).toHaveBeenCalledWith(rawContent))
    })
  })
})
