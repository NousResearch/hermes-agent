import { Box } from '@hermes/ink'
import chalk from 'chalk'
import React from 'react'
import { expect, it } from 'vitest'

import { renderToScreen } from '../../packages/hermes-ink/src/ink/render-to-screen.js'
import { cellAtIndex, CellWidth } from '../../packages/hermes-ink/src/ink/screen.js'
import { Md } from '../components/markdown.js'
import { DEFAULT_THEME, LIGHT_THEME } from '../theme.js'

// Use the real Yoga/Ink paint path. The style ID's low bit records styling
// visible on spaces; these fixtures have backgrounds but no inverse/underline.
const renderRows = (text: string, width: number, theme = DEFAULT_THEME) => {
  const level = chalk.level
  chalk.level = 3
  try {
    const { screen, height } = renderToScreen(
      React.createElement(
        Box,
        { width, flexDirection: 'column' },
        React.createElement(Md, { cols: width, t: theme, text })
      ),
      width
    )
    return Array.from({ length: height }, (_, row) =>
      Array.from({ length: width }, (_, col) => cellAtIndex(screen, row * width + col))
    )
  } finally {
    chalk.level = level
  }
}

it('fills added/removed visual rows, including wrapped Unicode, without painting the gutter', () => {
  for (const theme of [DEFAULT_THEME, LIGHT_THEME]) {
    for (const width of [12, 32]) {
      for (const line of ['+x', '-old', '+中文🙂 more words that wrap across rows', '-']) {
        const rows = renderRows('```diff\n' + line + '\n```', width, theme)
        expect(rows.length).toBeGreaterThan(0)
        for (const row of rows) {
          expect(row.slice(0, 2).map(cell => cell.styleId & 1)).toEqual([0, 0])
          // Wide glyph continuation cells inherit the terminal glyph's style;
          // Ink stores no separate SGR on those spacer cells.
          for (const cell of row.slice(2)) {
            if (cell.width !== CellWidth.SpacerTail) expect(cell.styleId & 1).toBe(1)
          }
        }
        expect(
          rows
            .flat()
            .map(cell => cell.char)
            .join('')
            .replaceAll(' ', '')
        ).toBe(line.replaceAll(' ', ''))
      }
    }
  }
})

it('leaves neutral diff rows, ordinary code and empty fences unpainted', () => {
  for (const text of ['```diff\n context\n@@ hunk @@\n```', '```text\n+not a diff\n```', '```diff\n```']) {
    for (const row of renderRows(text, 32)) {
      expect(row.every(cell => (cell.styleId & 1) === 0)).toBe(true)
    }
  }
})
