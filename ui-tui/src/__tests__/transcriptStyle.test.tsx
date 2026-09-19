import '../lib/forceTruecolor.js'

import { PassThrough } from 'stream'

import { Box, renderSync, ScrollBox, type ScrollBoxHandle } from '@hermes/ink'
import { stripAnsi } from '@hermes/shared/ansi'
import chalk from 'chalk'
import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { MessageLine } from '../components/messageLine.js'
import { userMessageLayout } from '../lib/inputMetrics.js'
import { estimatedMsgHeight } from '../lib/virtualHeights.js'
import { contrastRatio, fromSkin, themeToneHex } from '../theme.js'
import type { Msg } from '../types.js'

const colors = {
  background: '#121212',
  ui_accent: '#e68e0d',
  ui_user: '#e68e0d',
  ui_heading: '#4dd0e1',
  user_message_bg: '#282828'
}

const theme = fromSkin(colors, {})

function streams(columns: number) {
  const stdout = Object.assign(new PassThrough(), { columns, isTTY: false, rows: 100 })
  const stdin = Object.assign(new PassThrough(), { isTTY: false })
  const stderr = Object.assign(new PassThrough(), { isTTY: false })
  let output = ''
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  return {
    options: {
      patchConsole: false,
      stdout: stdout as NodeJS.WriteStream,
      stdin: stdin as NodeJS.ReadStream,
      stderr: stderr as NodeJS.WriteStream
    },
    take: () => {
      const value = output
      output = ''

      return value
    }
  }
}

// Inspect emitted cells, not JSX/source shape: spaces must carry the fill too.
function filledRows(output: string) {
  return output.split('\n').filter(row => row.includes('\x1b[48;2;40;40;40m'))
}

describe('transcript semantic styles', () => {
  const originalLevel = chalk.level
  beforeEach(() => {
    chalk.level = 3
  })
  afterEach(() => {
    chalk.level = originalLevel
    vi.unstubAllEnvs()
  })
  it('separates user and both heading forms without changing legacy colors or code', () => {
    const legacy = fromSkin({ ui_accent: colors.ui_accent }, {})
    const io = streams(64)

    const view = renderSync(
      <Box flexDirection="column" width={60}>
        <MessageLine cols={64} msg={{ role: 'user', text: 'User text' }} t={theme} />
        <MessageLine
          cols={64}
          msg={{ role: 'assistant', text: '# Heading\n\nOther heading\n---\n\n```text\ncode sample\n```' }}
          t={theme}
        />
      </Box>,
      io.options
    )

    view.unmount()
    view.cleanup()
    const output = io.take()
    expect(output).toContain('\x1b[38;2;230;142;13m')

    for (const heading of ['Heading', 'Other heading']) {
      const row = output.split('\n').find(line => stripAnsi(line).includes(heading))
      expect(row).toContain('\x1b[38;2;77;208;225m')
    }

    expect(legacy.color.user).toBe(legacy.color.label)
    expect(legacy.color.heading).toBe(legacy.color.accent)
    expect(legacy.userMessageBg).toBeUndefined()
    expect(theme.color.user).toBe(colors.ui_user)
    expect(theme.color.heading).toBe(colors.ui_heading)
    expect(theme.color.syntaxString).toBe(legacy.color.syntaxString)
    expect(stripAnsi(output)).toContain('code sample')
    expect(filledRows(output).length).toBeGreaterThanOrEqual(3)
  })

  it('preserves legacy ANSI-light label normalization and contrasts against an authored user surface', () => {
    vi.stubEnv('TERM_PROGRAM', 'Apple_Terminal')
    vi.stubEnv('COLORTERM', '')
    vi.stubEnv('COLORFGBG', '0;15')
    const legacy = fromSkin({ background: '#ffffff', ui_label: '#ffffff' }, {})
    expect(legacy.color.user).toBe(legacy.color.label)
    expect(contrastRatio(themeToneHex(legacy.color.user), '#ffffff')).toBeGreaterThan(3)

    const darkSurface = fromSkin({ background: '#ffffff', ui_user: '#ffffff', user_message_bg: '#121212' }, {})
    expect(contrastRatio(themeToneHex(darkSurface.color.user), darkSurface.userMessageBg!)).toBeGreaterThan(15)
    const invisible = fromSkin({ background: '#ffffff', ui_user: '#121212', user_message_bg: '#121212' }, {})
    expect(contrastRatio(themeToneHex(invisible.color.user), invisible.userMessageBg!)).toBeGreaterThanOrEqual(1.45)
    const lightSurface = fromSkin({ background: '#ffffff', ui_user: '#ffffff', user_message_bg: '#ffffff' }, {})
    expect(contrastRatio(themeToneHex(lightSurface.color.user), lightSurface.userMessageBg!)).toBeGreaterThanOrEqual(
      1.18
    )
  })

  it('fills blank and wrapped rows, measures padded heights and clips them on scroll/resize', () => {
    const msg: Msg = { role: 'user', text: `${'abcdefghij'.repeat(18)}\n\nlast line` }

    for (const cols of [64, 28, 12, 7]) {
      const io = streams(cols)
      let height = () => 0

      const view = renderSync(
        <Box
          flexDirection="column"
          ref={node => {
            if (node) {
              height = () => node.yogaNode?.getComputedHeight() ?? 0
            }
          }}
          width={cols - 4}
        >
          <MessageLine cols={cols} msg={msg} t={theme} />
        </Box>,
        io.options
      )

      const measured = height()
      const rows = filledRows(io.take())
      view.unmount()
      view.cleanup()

      const expected = estimatedMsgHeight(msg, cols, {
        compact: true,
        details: false,
        userPrompt: theme.brand.prompt,
        userMessageFilled: true
      })

      expect(rows.length).toBe(expected - 2)

      for (const row of rows) {
        expect(stripAnsi(row).length).toBe(cols - 4)
      }

      const layout = userMessageLayout(cols, theme.brand.prompt, true)
      expect(layout.bodyWidth + layout.gutter + 2 * layout.paddingX).toBe(cols - 4)
      expect(measured).toBe(expected)
    }

    const io = streams(40)
    const scroll = React.createRef<ScrollBoxHandle>()

    const tree = (cols: number) => (
      <ScrollBox flexDirection="column" height={5} ref={scroll} width={cols - 4}>
        <MessageLine cols={cols} msg={msg} t={theme} />
      </ScrollBox>
    )

    const view = renderSync(tree(40), io.options)
    io.take()
    scroll.current!.scrollTo(3)
    view.rerender(tree(40))
    const clipped = filledRows(io.take())
    expect(clipped).toHaveLength(5)

    for (const row of clipped) {
      expect(stripAnsi(row).length).toBe(36)
    }

    view.rerender(tree(20))
    io.take()
    scroll.current!.scrollToBottom()
    view.rerender(tree(20))
    const resized = io.take()
    expect(stripAnsi(resized)).toContain('last line')

    for (const row of filledRows(resized)) {
      expect(stripAnsi(row).length).toBe(16)
    }

    view.unmount()
    view.cleanup()
  })
})
