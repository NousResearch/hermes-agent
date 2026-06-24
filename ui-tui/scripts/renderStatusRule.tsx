// Status rule visual validation: render the StatusRule at multiple column
// widths and dump the plain-text output so we can compare before/after the
// separator + color refinements.  Run: npx tsx scripts/renderStatusRule.tsx

import { existsSync, mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { PassThrough } from 'stream'

import { Box, renderSync } from '@hermes/ink'
import React from 'react'

import { StatusRule } from '../src/components/appChrome.js'
import { DEFAULT_THEME } from '../src/theme.js'

// `no-control-regex` warns on the literal escape, but stripping ANSI is the
// whole point of this helper — the pattern is correct.
// eslint-disable-next-line no-control-regex
const stripAnsi = (s: string) => s.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '')

const renderOnce = (node: React.ReactNode, cols: number): string => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: cols, isTTY: false, rows: 50 })
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
}

const baseProps = {
  busy: false,
  cwdLabel: '~/src/hermes-agent/ui-tui (feat/refine-statusrule)',
  liveSessionCount: 2,
  model: 'gpt-5.4-mini',
  showCost: true,
  status: 'ready',
  statusColor: DEFAULT_THEME.color.ok,
  t: DEFAULT_THEME,
  turnStartedAt: null,
  usage: { context_max: 272_000, context_percent: 45, context_used: 121_800, total: 121_800, cost_usd: 0.0123, compressions: 2 },
  voiceLabel: 'voice off',
  bgCount: 1,
  lastTurnEndedAt: Date.now() - 95_000,
  sessionStartedAt: Date.now() - 923_000
}

const buildTree = (cols: number, opts: { busy: boolean }): React.ReactNode => {
  return React.createElement(
    Box,
    { flexDirection: 'column' },
    React.createElement(StatusRule, { ...baseProps, ...opts, cols })
  )
}

const WIDTHS = [200, 140, 100, 80, 60, 44] as const

const main = () => {
  const outDir = '/tmp/opencode/statusrule-snapshots'

  if (!existsSync(outDir)) {
    mkdirSync(outDir, { recursive: true })
  }

  for (const w of WIDTHS) {
    for (const mode of ['idle', 'busy'] as const) {
      const tree = buildTree(w, { busy: mode === 'busy' })
      const raw = renderOnce(tree, w)

      const text = stripAnsi(raw)
        .split('\n')
        .map(l => l.replace(/\s+$/g, ''))
        .filter(l => l.length > 0)
        .join('\n')

      const file = join(outDir, `${w}-${mode}.txt`)
      writeFileSync(file, text)
       
      console.log(`wrote ${file}\n${text}\n---`)
    }
  }
}

main()
