// Composer + StatusRule visual validation: render the input wrapper with the
// new intentional `borderStyle="round"` so we can confirm the "empty box"
// effect becomes a real, visible frame.
//
// We can't render the full App (it needs a real GatewayClient), so we
// approximate the composer pane: StatusRule above the prompt + input
// wrapper, matching the AppLayout's flex column order.
//
// Run: npx tsx scripts/renderComposerBorder.tsx

import { existsSync, mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { PassThrough } from 'stream'

import { Box, renderSync, Text } from '@hermes/ink'
import React from 'react'

import { StatusRule } from '../src/components/appChrome.js'
import { DEFAULT_THEME } from '../src/theme.js'

// `no-control-regex` warns on the literal escape, but stripping ANSI is the
// whole point of this helper — the pattern is correct.
// eslint-disable-next-line no-control-regex
const stripAnsi = (s: string) => s.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '')

const renderOnce = (node: React.ReactNode, cols: number, rows = 50): string => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: cols, isTTY: false, rows })
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
  showCost: false,
  status: 'ready',
  statusColor: DEFAULT_THEME.color.ok,
  t: DEFAULT_THEME,
  turnStartedAt: null,
  usage: { context_max: 272_000, context_percent: 45, context_used: 121_800, total: 121_800, compressions: 0 },
  voiceLabel: 'voice off',
  bgCount: 0,
  lastTurnEndedAt: null,
  sessionStartedAt: Date.now() - 923_000
}

const buildTree = (cols: number): React.ReactNode => {
  const t = DEFAULT_THEME
  const innerWidth = Math.max(1, cols - 2)
  // Border consumes 1 cell each side, so the inner content area is innerWidth - 2.
  const inputWidth = Math.max(1, innerWidth - 2)

  return React.createElement(
    Box,
    { flexDirection: 'column' },
    // StatusRule above (mirrors `ui.statusBar === 'top'`)
    React.createElement(
      Box,
      { paddingX: 1 },
      React.createElement(StatusRule, { ...baseProps, cols })
    ),
    // Prompt line
    React.createElement(
      Box,
      { paddingX: 1 },
      React.createElement(Text, { color: t.color.muted }, 'assistente >')
    ),
    // Input wrapper with the new intentional border
    React.createElement(
      Box,
      { paddingX: 1 },
      React.createElement(
        Box,
        {
          backgroundColor: t.color.completionCurrentBg,
          borderColor: t.color.border,
          borderStyle: 'round',
          width: innerWidth
        },
        React.createElement(Text, { color: t.color.muted, dimColor: true }, 'type a message…')
      )
    ),
    // StatusRule below (mirrors `ui.statusBar === 'bottom'`)
    React.createElement(
      Box,
      { paddingX: 1 },
      React.createElement(StatusRule, { ...baseProps, cols })
    )
  )
}

// Local Text import to avoid touching the top of the file
const WIDTHS = (process.argv.slice(2).map(Number).filter(n => n > 0) as number[]).length
  ? (process.argv.slice(2).map(Number).filter(n => n > 0) as number[])
  : [200, 140, 100, 80, 60, 44]

const main = () => {
  const outDir = '/tmp/opencode/composer-snapshots'

  if (!existsSync(outDir)) {
    mkdirSync(outDir, { recursive: true })
  }

  for (const w of WIDTHS) {
    const tree = buildTree(w)
    const raw = renderOnce(tree, w, 8)

    const text = stripAnsi(raw)
      .split('\n')
      .map(l => l.replace(/\s+$/g, ''))
      .join('\n')

    const file = join(outDir, `${w}.txt`)
    writeFileSync(file, text)
     
    console.log(`wrote ${file}\n${text}\n---`)
  }
}

main()
