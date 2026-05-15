/**
 * Table rendering reproduction suite.
 *
 * These tests exist purely to *demonstrate* the current rendering bugs at
 * various terminal widths.  They are NOT assertions about correct behavior —
 * they snapshot what the output looks like today so a future fix can A/B
 * against a known-broken baseline.
 *
 * Run:
 *   cd ui-tui && npx vitest run src/__tests__/tableRepro.test.ts
 *
 * Each test prints the rendered output to the console so you can eyeball it.
 * The actual `expect()` calls just lock in structural properties of the
 * current (broken) behavior for diffing later.
 */
import { PassThrough } from 'stream'

import { Box, renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { Md } from '../components/markdown.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

const t = DEFAULT_THEME

// ── render helper — parameterized by column width ──────────────────────
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
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const node = React.createElement(
    Box,
    { width: columns },
    React.createElement(Md, { t, text: md })
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

// ── markdown table fixtures ────────────────────────────────────────────

/** Simple 3-column table that fits at 80 cols */
const SIMPLE_TABLE = [
  '| Name          | Age | City          |',
  '|---------------|-----|---------------|',
  '| Alice Johnson | 28  | San Francisco |',
  '| Bob Smith     | 34  | New York      |',
  '| Carol Davis   | 42  | Chicago       |',
  '| David Wilson  | 31  | Austin        |'
].join('\n')

/** 6-column table — wider than 80 cols, fits at ~140 */
const WIDE_TABLE = [
  '| Project Name       | Description                    | Technology Stack              | Timeline | Budget Range | Status   |',
  '|--------------------|--------------------------------|-------------------------------|----------|--------------|----------|',
  '| E-commerce Platform | Modern online shopping experience | React, Node.js, PostgreSQL, Redis | 6 months | $150K-200K  | In Progress |',
  '| Mobile Banking App | Secure financial transactions   | Flutter, Firebase, Kubernetes | 8 months | $300K-400K  | Planning |',
  '| AI Content Generator | Automated marketing copy creation | Python, TensorFlow, FastAPI  | 4 months | $80K-120K   | Complete |',
  '| IoT Dashboard      | Real-time sensor data visualization | Vue.js, InfluxDB, Docker, AWS | 5 months | $200K-250K  | Testing  |'
].join('\n')

/** Table with long file paths — columns naturally ~90+ chars */
const PATH_TABLE = [
  '| File Path | Description |',
  '|-----------|-------------|',
  '| /home/user/very/long/path/to/some/deeply/nested/file.py | Main application entry point with configuration loading |',
  '| /home/user/projects/backend/src/controllers/authentication/user_management.py | User authentication and session management controller |',
  '| /home/user/workspace/frontend/components/dashboard/widgets/analytics/chart_renderer.tsx | React component for rendering interactive analytics charts |',
  '| /home/user/dev/microservices/payment-service/src/handlers/stripe/webhook_processor.py | Stripe webhook event processor for payment notifications |'
].join('\n')

/** Table with extreme width disparity: 2-3 char codes vs 100+ char descriptions */
const LOPSIDED_TABLE = [
  '| Code | Country Name   | Detailed Description |',
  '|------|----------------|----------------------|',
  '| US   | United States  | A federal republic consisting of 50 states and various territories, known for its diverse geography, technological innovation, and significant global economic influence |',
  '| UK   | United Kingdom | A sovereign country comprising England, Scotland, Wales, and Northern Ireland, historically significant for its maritime empire, industrial revolution, and continuing influence |',
  '| DE   | Germany        | A federal parliamentary republic in Central Europe, recognized as Europes largest economy and a leader in automotive manufacturing, engineering, and renewable energy technology |',
  '| JP   | Japan          | An island nation in East Asia known for its advanced technology sector, automotive industry, unique cultural traditions, and significant contributions to electronics and robotics |'
].join('\n')

/** Table with mixed content: numbers, inline code, URLs */
const MIXED_TABLE = [
  '| ID   | Revenue    | Endpoint                       | Code Snippet                                | Score |',
  '|------|------------|--------------------------------|---------------------------------------------|-------|',
  "| 1001 | $2,450,000 | https://api.example.com/users  | `fetch('/api/data').then(r => r.json())`    | 94.7  |",
  "| 1002 | $875,320   | https://api.example.com/analytics | `def process(df): return df.groupby('cat')` | 87.2  |",
  "| 1003 | $1,200,000 | https://api.example.com/payments | `SELECT * FROM orders WHERE status = 'ok'`  | 91.8  |"
].join('\n')

/** Reproduction of the user's original bug report — paths + "What it is" descriptions */
const AGENTS_MD_TABLE = [
  '| Path | What it is |',
  '|------|------------|',
  '| environments/ (43 files) | Entire RL environments directory — base env, agent loop, tool context, patches, SWE env, terminal test env, benchmarks (TerminalBench2, TBLite, YC bench), tool call parsers (hermes, llama, mistral, deepseek, qwen, kimi, glm, etc.) |',
  '| rl_cli.py (~410 lines) | Standalone CLI for RL training via Tinker-Atropos |',
  '| tools/rl_training_tool.py (~1400 lines) | All 10 rl_* tools (list/select/start/stop/configure/status/results/etc.) |'
].join('\n')

// ── test widths ────────────────────────────────────────────────────────
const WIDTHS = [60, 80, 100, 120, 160]

// ── helpers ────────────────────────────────────────────────────────────

/** Count how many rendered lines a table occupies (non-empty) */
const nonEmpty = (lines: string[]) => lines.filter(l => l.trim().length > 0)

/**
 * Check if the separator line occupies more than 1 rendered line.
 * A healthy separator is a single line of ─ characters.
 * A broken one wraps across 2+ lines.
 */
const separatorLines = (lines: string[]) =>
  lines.filter(l => /^[\s─…]+$/.test(l) && l.includes('─'))

/**
 * Detect "column bleed" — when cell content from column N appears on the
 * same visual position as column N+1's header.  Crude heuristic: find the
 * header row, measure where each column header starts, then check if any
 * data row has non-space characters before that column start.
 */

// ── the actual repro tests ─────────────────────────────────────────────

describe('Table rendering reproduction', () => {
  describe.each(WIDTHS)('at %d columns', (width) => {
    it('simple 3-col table', () => {
      const lines = renderAtWidth(SIMPLE_TABLE, width)
      const content = nonEmpty(lines)

      console.log(`\n${'═'.repeat(width)}`)
      console.log(`SIMPLE TABLE @ ${width} cols`)
      console.log('═'.repeat(width))
      lines.forEach(l => console.log(`│${l}${'░'.repeat(Math.max(0, width - l.length))}│`))
      console.log('═'.repeat(width))

      // At any reasonable width this table should fit — 4 data rows + 1 header
      expect(content.length).toBeGreaterThanOrEqual(5)
    })

    it('wide 6-col table', () => {
      const lines = renderAtWidth(WIDE_TABLE, width)
      const seps = separatorLines(lines)

      console.log(`\n${'═'.repeat(width)}`)
      console.log(`WIDE TABLE @ ${width} cols`)
      console.log('═'.repeat(width))
      lines.forEach(l => console.log(`│${l}${'░'.repeat(Math.max(0, width - l.length))}│`))
      console.log('═'.repeat(width))

      // Document: does the separator wrap to multiple lines?
      console.log(`  → separator lines: ${seps.length}`)
      // Just lock in what happens — don't assert correctness
      expect(seps.length).toBeGreaterThanOrEqual(1)
    })

    it('long file paths table', () => {
      const lines = renderAtWidth(PATH_TABLE, width)

      console.log(`\n${'═'.repeat(width)}`)
      console.log(`PATH TABLE @ ${width} cols`)
      console.log('═'.repeat(width))
      lines.forEach(l => console.log(`│${l}${'░'.repeat(Math.max(0, width - l.length))}│`))
      console.log('═'.repeat(width))

      expect(nonEmpty(lines).length).toBeGreaterThanOrEqual(1)
    })

    it('lopsided table (short code + long description)', () => {
      const lines = renderAtWidth(LOPSIDED_TABLE, width)

      console.log(`\n${'═'.repeat(width)}`)
      console.log(`LOPSIDED TABLE @ ${width} cols`)
      console.log('═'.repeat(width))
      lines.forEach(l => console.log(`│${l}${'░'.repeat(Math.max(0, width - l.length))}│`))
      console.log('═'.repeat(width))

      expect(nonEmpty(lines).length).toBeGreaterThanOrEqual(1)
    })

    it('mixed content table (code, URLs, numbers)', () => {
      const lines = renderAtWidth(MIXED_TABLE, width)

      console.log(`\n${'═'.repeat(width)}`)
      console.log(`MIXED TABLE @ ${width} cols`)
      console.log('═'.repeat(width))
      lines.forEach(l => console.log(`│${l}${'░'.repeat(Math.max(0, width - l.length))}│`))
      console.log('═'.repeat(width))

      expect(nonEmpty(lines).length).toBeGreaterThanOrEqual(1)
    })

    it('original bug report table (AGENTS.md RL paths)', () => {
      const lines = renderAtWidth(AGENTS_MD_TABLE, width)

      console.log(`\n${'═'.repeat(width)}`)
      console.log(`AGENTS.MD TABLE @ ${width} cols`)
      console.log('═'.repeat(width))
      lines.forEach(l => console.log(`│${l}${'░'.repeat(Math.max(0, width - l.length))}│`))
      console.log('═'.repeat(width))

      expect(nonEmpty(lines).length).toBeGreaterThanOrEqual(1)
    })
  })
})
