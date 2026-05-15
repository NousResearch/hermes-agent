/**
 * Adversarial / fuzzer-style table fixtures.
 *
 * Every table here is designed to tickle a specific edge case in
 * renderTable() / Ink's layout engine.  Run alongside the base repro:
 *
 *   cd ui-tui && node_modules/.bin/vitest run src/__tests__/tableRepro.test.ts src/__tests__/tableFuzz.test.ts
 *
 * These all pass (they're repro, not correctness assertions) — the visual
 * output in the console shows the breakage.
 */
import { PassThrough } from 'stream'

import { Box, renderSync, stringWidth } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { Md } from '../components/markdown.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

const t = DEFAULT_THEME

const BEL = String.fromCharCode(7)
const ESC = String.fromCharCode(27)
const CSI_RE = new RegExp(`${ESC}\\[[0-?]*[ -/]*[@-~]`, 'g')
const OSC_RE = new RegExp(`${ESC}\\][\\s\\S]*?(?:${BEL}|${ESC}\\\\)`, 'g')

const renderAtWidth = (md: string, columns: number): string[] => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns, isTTY: false, rows: 24 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', (chunk: Buffer) => {
    output += chunk.toString()
  })

  const node = React.createElement(
    Box,
    { width: columns },
    React.createElement(Md, { cols: columns, t, text: md })
  )

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

const nonEmpty = (lines: string[]) => lines.filter(l => l.trim().length > 0)

const vis = (lines: string[], width: number, label: string) => {
  console.log(`\n${'═'.repeat(Math.min(width, 120))}`)
  console.log(`${label} @ ${width} cols`)
  console.log('═'.repeat(Math.min(width, 120)))
  for (const l of lines) {
    const fill = Math.max(0, width - l.length)
    console.log(`│${l}${'░'.repeat(Math.min(fill, 200))}│`)
  }
  console.log('═'.repeat(Math.min(width, 120)))
}

// ═══════════════════════════════════════════════════════════════════════
// FIXTURES — each targets a specific rendering edge case
// ═══════════════════════════════════════════════════════════════════════

// ── 1. Single-column table (degenerate) ────────────────────────────────
// renderTable builds column widths from rows[0].  A 1-col table means
// the separator has no `  ` gaps — does it still render?
const SINGLE_COL = [
  '| Item |',
  '|------|',
  '| Apple |',
  '| Banana |',
  '| Cherry |'
].join('\n')

// ── 2. Empty cells ─────────────────────────────────────────────────────
// What happens when some cells are completely empty?
const EMPTY_CELLS = [
  '| A | B | C |',
  '|---|---|---|',
  '|   | hello |   |',
  '| world |   |   |',
  '|   |   | end |'
].join('\n')

// ── 3. Cell content is just pipes / special chars ──────────────────────
// Escaped pipes, backtick-wrapped pipes, table-hostile characters
const PIPE_HELL = [
  '| Expression | Result |',
  '|------------|--------|',
  '| `a \\| b`   | true   |',
  '| `x \\|\\| y` | false  |',
  '| ──────     | ═══    |',
  '| `│` char   | box    |'
].join('\n')

// ── 4. CJK + emoji mixed with ASCII ───────────────────────────────────
// CJK glyphs are 2 cells wide.  Emoji with VS16 are also 2 cells.
// Column alignment must use display width, not string length.
const CJK_EMOJI = [
  '| 名前 | Status | Description |',
  '|------|--------|-------------|',
  '| 田中太郎 | ✅ Active | 東京都渋谷区のエンジニア |',
  '| Alice | ❌ Inactive | Regular ASCII text here |',
  '| 李明 | 🔥 Hot | 北京市朝阳区 short |',
  '| Bob | ⚠️ Warning | Mixed 中英文 content here |'
].join('\n')

// ── 5. One column is 1 char, another is 300+ chars ─────────────────────
// Extreme width disparity — the wide column dominates everything.
const EXTREME_DISPARITY = [
  '| # | Description |',
  '|---|-------------|',
  `| 1 | ${'A'.repeat(300)} |`,
  `| 2 | ${'B'.repeat(150)}${'C'.repeat(150)} |`,
  '| 3 | short |'
].join('\n')

// ── 6. 20 columns — many narrow columns ───────────────────────────────
// Total natural width will be huge even with short cells.
const MANY_COLUMNS = (() => {
  const hdr = Array.from({ length: 20 }, (_, i) => `C${i}`).join(' | ')
  const div = Array.from({ length: 20 }, () => '---').join(' | ')
  const row1 = Array.from({ length: 20 }, (_, i) => `${i * 3}`).join(' | ')
  const row2 = Array.from({ length: 20 }, (_, i) => `${'x'.repeat(i + 1)}`).join(' | ')
  return `| ${hdr} |\n| ${div} |\n| ${row1} |\n| ${row2} |`
})()

// ── 7. Inline markdown inside cells ────────────────────────────────────
// Bold, italic, code, links — cellWidth uses stripInlineMarkup but
// the rendered width includes the Ink formatting which might differ.
const INLINE_MARKUP_CELLS = [
  '| Feature | Status | Notes |',
  '|---------|--------|-------|',
  '| **Bold feature** | `done` | See [docs](https://example.com/very/long/path/to/documentation/page) |',
  '| *Italic thing* | ~~cancelled~~ | Has `inline code` and **bold** mixed |',
  '| $E = mc^2$ | ==highlighted== | Math: $\\alpha + \\beta = \\gamma$ |',
  '| Normal | Normal | [link text that is quite long](https://example.com) |'
].join('\n')

// ── 8. All cells identical width — separator alignment test ────────────
// If every cell is exactly 10 chars, the separator should line up perfectly.
const UNIFORM_CELLS = [
  '| AAAAAAAAAA | BBBBBBBBBB | CCCCCCCCCC |',
  '|------------|------------|------------|',
  '| 0123456789 | 0123456789 | 0123456789 |',
  '| abcdefghij | abcdefghij | abcdefghij |'
].join('\n')

// ── 9. Table immediately after a code fence ────────────────────────────
// Tests that the parser correctly transitions from code block to table.
const TABLE_AFTER_CODE = [
  '```python',
  'def foo():',
  '    return "bar"',
  '```',
  '',
  '| Function | Returns |',
  '|----------|---------|',
  '| foo()    | "bar"   |',
  '| baz()    | "qux"   |'
].join('\n')

// ── 10. Table with cells containing newline-like content ───────────────
// Literal \n in cells (not actual newlines) — should stay on one line.
const ESCAPED_NEWLINES = [
  '| Pattern | Replacement |',
  '|---------|-------------|',
  '| `\\n`    | newline     |',
  '| `\\t`    | tab         |',
  '| `\\r\\n`  | CRLF        |',
  '| `\\\\`    | backslash   |'
].join('\n')

// ── 11. Table where header has fewer cols than body ────────────────────
// Ragged table — some rows have more pipes than the header.
const RAGGED_ROWS = [
  '| A | B |',
  '|---|---|',
  '| 1 | 2 | 3 | 4 |',
  '| x | y |',
  '| p | q | r |'
].join('\n')

// ── 12. Table with very tall cells (many words that wrap) ──────────────
// At narrow widths, a single cell can produce 10+ visual lines.
const TALL_CELL = [
  '| ID | Description |',
  '|-----|-------------|',
  `| 1   | ${'The quick brown fox jumps over the lazy dog. '.repeat(8).trim()} |`,
  '| 2   | Short. |'
].join('\n')

// ── 13. Separator-only rows (no data) ──────────────────────────────────
// A table that's just headers + separator, no body.  Edge case for
// the `rows.length > 1` check on line 235 of markdown.tsx.
const HEADER_ONLY = [
  '| Name | Age | City |',
  '|------|-----|------|'
].join('\n')

// ── 14. Unicode box-drawing chars in cells ─────────────────────────────
// These are the same chars used for the separator — does it confuse
// the table divider detection?
const BOX_CHARS_IN_CELLS = [
  '| Symbol | Name | Width |',
  '|--------|------|-------|',
  '| ─      | horizontal | 1 |',
  '| │      | vertical | 1 |',
  '| ┌┐└┘   | corners | 4 |',
  '| ═══    | double | 3 |',
  '| ╔╗╚╝   | double corners | 4 |'
].join('\n')

// ── 15. Table with URLs that are longer than the terminal ──────────────
const LONG_URLS = [
  '| Service | Endpoint |',
  '|---------|----------|',
  '| Auth    | https://api.example.com/v2/authentication/oauth2/authorize?client_id=abc123&redirect_uri=https%3A%2F%2Fapp.example.com%2Fcallback&scope=read%20write%20admin&state=xyz789 |',
  '| Data    | https://data.example.com/api/v3/datasets/my-very-long-dataset-name/versions/2024-01-15/records?filter=status%3Dactive&limit=100&offset=0&sort=created_at%3Adesc |'
].join('\n')

// ── 16. Numbers with mixed alignment ───────────────────────────────────
// Right-aligned numbers next to left-aligned text — the current renderer
// ignores alignment markers entirely.
const ALIGNMENT = [
  '| Item | Quantity | Price |',
  '|:-----|:--------:|------:|',
  '| Widget A | 1,234 | $99.99 |',
  '| Gadget B | 42 | $1,299.00 |',
  '| Thingamajig C | 999,999 | $0.01 |'
].join('\n')

// ── 17. Table with only separator-style content ────────────────────────
// Cells that look like separators — does isTableDivider false-positive?
const SEPARATOR_LOOKALIKES = [
  '| Pattern | Matches |',
  '|---------|---------|',
  '| ---     | yes     |',
  '| :---:   | center  |',
  '| ---:    | right   |',
  '| :---    | left    |',
  '| ----    | yes     |'
].join('\n')

// ── 18. Table adjacent to another table (no gap) ───────────────────────
// Two tables with only a blank line between them.
const ADJACENT_TABLES = [
  '| A | B |',
  '|---|---|',
  '| 1 | 2 |',
  '',
  '| X | Y | Z |',
  '|---|---|---|',
  '| a | b | c |'
].join('\n')

// ── 19. Table with backtick-code cells that contain pipes ──────────────
// The pipe inside backticks should NOT be treated as a column separator.
const CODE_WITH_PIPES = [
  '| Command | Description |',
  '|---------|-------------|',
  '| `echo "hello" \\| grep h` | Filter output |',
  '| `cat file \\| wc -l` | Count lines |',
  '| `ps aux \\| head -5` | First 5 processes |'
].join('\n')

// ── 20. The "stress test" — everything at once ─────────────────────────
const KITCHEN_SINK = [
  '| # | 名前 | Path | **Status** | $Math$ | Link | Long Description |',
  '|---|------|------|------------|--------|------|-----------------|',
  `| 1 | 田中 | /home/user/very/long/path/to/deeply/nested/config.yaml | \`active\` | $\\alpha$ | [docs](https://example.com/very/long/docs) | ${'This is an extremely long description that will definitely overflow any reasonable terminal width and cause wrapping issues. '.repeat(2).trim()} |`,
  '| 2 | Bob | /tmp | ~~dead~~ | $E=mc^2$ | [x](https://x.com) | Short |',
  `| 3 | 李明 | /home/user/projects/backend/src/controllers/authentication/user_management_service_v2.py | ==HOT== | $\\sum_{i=0}^{n}$ | [link](https://example.com) | ${'Mixed 中英文 content with emoji 🔥 and special chars ─│┌┐ '.repeat(3).trim()} |`
].join('\n')

// ═══════════════════════════════════════════════════════════════════════
// TEST SUITE
// ═══════════════════════════════════════════════════════════════════════

const FIXTURES: Array<[string, string]> = [
  ['single-column', SINGLE_COL],
  ['empty-cells', EMPTY_CELLS],
  ['pipe-hell', PIPE_HELL],
  ['cjk-emoji', CJK_EMOJI],
  ['extreme-disparity', EXTREME_DISPARITY],
  ['many-columns-20', MANY_COLUMNS],
  ['inline-markup', INLINE_MARKUP_CELLS],
  ['uniform-cells', UNIFORM_CELLS],
  ['table-after-code', TABLE_AFTER_CODE],
  ['escaped-newlines', ESCAPED_NEWLINES],
  ['ragged-rows', RAGGED_ROWS],
  ['tall-cell', TALL_CELL],
  ['header-only', HEADER_ONLY],
  ['box-chars-in-cells', BOX_CHARS_IN_CELLS],
  ['long-urls', LONG_URLS],
  ['alignment-markers', ALIGNMENT],
  ['separator-lookalikes', SEPARATOR_LOOKALIKES],
  ['adjacent-tables', ADJACENT_TABLES],
  ['code-with-pipes', CODE_WITH_PIPES],
  ['kitchen-sink', KITCHEN_SINK]
]

const WIDTHS = [40, 60, 80, 120]

describe('Adversarial table fuzzing', () => {
  describe.each(WIDTHS)('at %d columns', (width) => {
    it.each(FIXTURES)('%s', (name, md) => {
      const lines = renderAtWidth(md, width)

      vis(lines, width, name)

      // Structural: rendered content must exist
      expect(nonEmpty(lines).length).toBeGreaterThanOrEqual(1)

      // Structural: no line should exceed the allocated width (using stringWidth for CJK)
      for (const line of nonEmpty(lines)) {
        expect(stringWidth(line)).toBeLessThanOrEqual(width)
      }
    })
  })
})
