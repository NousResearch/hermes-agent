/* Visual self-verification tool: `npm run visual` renders real TUI surfaces
 * across theme x background scenes to /tmp/tui-visual.html, then shot.mjs
 * screenshots it to /tmp/tui-visual.png for eyeball + agent review.
 *
 * Original note: : render real TUI surfaces with ANSI colors intact,
 * convert to HTML on the actual background, and screenshot in a browser. */
process.env.FORCE_COLOR = '3'
process.env.COLORTERM = 'truecolor'

import '../../src/lib/forceTruecolor.js'

import { writeFileSync } from 'fs'
import { PassThrough } from 'stream'

import { Box, renderSync, Text } from '@hermes/ink'
import React, { type ReactElement } from 'react'

import { GatewayProvider } from '../../src/app/gatewayContext.js'
import { patchOverlayState, resetOverlayState } from '../../src/app/overlayStore.js'
import { patchUiState, resetUiState } from '../../src/app/uiStore.js'
import { FloatingOverlays } from '../../src/components/appOverlays.js'
import { Banner, SessionPanel } from '../../src/components/branding.js'
import { fromSkin, type Theme } from '../../src/theme.js'
import type { SessionInfo } from '../../src/types.js'

const noop = () => {}
const pending = () => new Promise<never>(() => {})
 
const fakeGateway = { gw: { notify: noop, off: noop, on: noop, request: pending }, rpc: pending } as any

const SLATE = {
  banner_border: '#4169e1',
  banner_title: '#7eb8f6',
  banner_accent: '#8EA8FF',
  banner_dim: '#4b5563',
  banner_text: '#c9d1d9',
  ui_accent: '#7eb8f6',
  ui_label: '#8EA8FF',
  ui_ok: '#63D0A6',
  ui_error: '#F7A072',
  ui_warn: '#e6a855',
  prompt: '#c9d1d9',
  session_label: '#7eb8f6',
  session_border: '#545E6B',
  status_bar_bg: '#151C2F',
  status_bar_text: '#C9D1D9'
}

// The regenerated slate light_colors block from hermes_cli/skin_engine.py
// (relight recipe: vivid hue-preserved accents, airy capped-saturation text,
// darker calm dims).

const info: SessionInfo = {
  cwd: '/Users/brooklyn/www/hermes-agent',
  mcp_servers: [{ connected: true, name: 'figma', tools: 12, transport: 'sse' }],
  model: 'claude-opus-4.8-fast',
  skills: {
    devops: ['docker', 'kubernetes', 'terraform'],
    github: ['pr-review', 'issue-triage'],
    productivity: ['powerpoint', 'excel', 'notion-sync']
  },
  tools: {
    browser: ['browser_back', 'browser_click', 'browser_console', 'browser_get_images'],
    clarify: ['clarify'],
    code_execution: ['execute_code'],
    cronjob: ['cronjob'],
    delegation: ['delegate_task'],
    file: ['patch', 'read_file', 'search_files', 'write_file']
  },
  update_behind: 1,
  version: '3.2.1'
}

const completions = [
  { display: '/new', meta: 'Start a new session (fresh session ID + history)', text: '/new' },
  { display: '/reset', meta: 'Start a new session (alias for /new)', text: '/reset' },
  { display: '/clear', meta: 'Clear screen and start a new session', text: '/clear' },
  { display: '/redraw', meta: 'Force a full UI repaint', text: '/redraw' },
  { display: '/history', meta: 'Show conversation history', text: '/history' }
]

function renderAnsi(node: ReactElement, columns: number): string {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()

  let output = ''

  ;(process.stdout as unknown as { columns: number }).columns = columns
  Object.assign(stdout, { columns, isTTY: false, rows: 60 })
  Object.assign(stdin, {
    isTTY: true,
    pause: noop,
    ref: noop,
    resume: noop,
    setEncoding: noop,
    setRawMode: noop,
    unref: noop
  })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(
    <GatewayProvider value={fakeGateway}>
      <Box flexDirection="column" width={columns}>{node}</Box>
    </GatewayProvider>,
    {
      exitOnCtrlC: false,
      patchConsole: false,
      stderr: stderr as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as unknown as NodeJS.WriteStream
    }
  )

  instance.unmount()
  instance.cleanup()

  return output
}

// ── ANSI → HTML (handles ink's SGR set: 38;2/48;2 truecolor, named resets, bold/dim/italic/inverse) ──

const escapeHtml = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

function ansiToHtml(raw: string, defaultFg: string, defaultBg: string): string {
  let fg = defaultFg
  let bg = defaultBg
  let bold = false
  let dim = false
  let italic = false
  let inverse = false
  let html = ''

  const openSpan = () => {
    const f = inverse ? bg : fg
    const b = inverse ? fg : bg
    const styles = [`color:${f}`]

    if (b !== defaultBg || inverse) {
      styles.push(`background-color:${b}`)
    }

    if (bold) {
      styles.push('font-weight:bold')
    }

    if (dim) {
      styles.push('opacity:0.55')
    }

    if (italic) {
      styles.push('font-style:italic')
    }

    return `<span style="${styles.join(';')}">`
  }

  // eslint-disable-next-line no-control-regex
  const parts = raw.split(/(\x1b\[[0-9;]*m)/)

  html += openSpan()

  for (const part of parts) {
    // eslint-disable-next-line no-control-regex
    const m = /^\x1b\[([0-9;]*)m$/.exec(part)

    if (!m) {
      // Drop non-SGR escapes (cursor moves etc.) — renderSync output for a
      // static frame is line-oriented, so this is safe for inspection.
      // eslint-disable-next-line no-control-regex
      html += escapeHtml(part.replace(/\x1b\[[^m]*[A-Za-z]/g, ''))

      continue
    }

    const codes = (m[1] || '0').split(';').map(Number)

    for (let i = 0; i < codes.length; i++) {
      const c = codes[i]!

      if (c === 0) {
        fg = defaultFg
        bg = defaultBg
        bold = dim = italic = inverse = false
      } else if (c === 1) {bold = true}
      else if (c === 2) {dim = true}
      else if (c === 3) {italic = true}
      else if (c === 7) {inverse = true}
      else if (c === 22) { bold = false; dim = false }
      else if (c === 23) {italic = false}
      else if (c === 27) {inverse = false}
      else if (c === 39) {fg = defaultFg}
      else if (c === 49) {bg = defaultBg}
      else if (c === 38 && codes[i + 1] === 2) {
        fg = `rgb(${codes[i + 2]},${codes[i + 3]},${codes[i + 4]})`
        i += 4
      } else if (c === 48 && codes[i + 1] === 2) {
        bg = `rgb(${codes[i + 2]},${codes[i + 3]},${codes[i + 4]})`
        i += 4
      } else if (c === 38 && codes[i + 1] === 5) {
        fg = `var(--a${codes[i + 2]}, #888)`
        i += 2
      } else if (c === 48 && codes[i + 1] === 5) {
        bg = `var(--a${codes[i + 2]}, #888)`
        i += 2
      }
    }

    html += `</span>${openSpan()}`
  }

  return html + '</span>'
}

// ── Scenes ──

interface Scene {
  bg: string
  fg: string
  name: string
  theme: Theme
}

const setup = (bgHex: string) => {
  process.env.HERMES_TUI_BACKGROUND = bgHex
  resetOverlayState()
  resetUiState()
}

const scenes: Scene[] = []

const addScene = (name: string, bgHex: string, skin: Record<string, string>) => {
  setup(bgHex)

  const theme = fromSkin(skin, {})

  scenes.push({ bg: bgHex, fg: theme.color.text, name, theme })
}

addScene('default · dark terminal', '#101014', {})
addScene('default · light terminal (Cursor)', '#ffffff', {})
addScene('slate · dark terminal', '#101014', SLATE)
addScene('slate · light terminal (raw palette + display shim)', '#ffffff', SLATE)

let page = `<!doctype html><meta charset="utf-8"><body style="margin:0;background:#666;font:13px/1.35 Menlo,monospace"><div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:16px">`

for (const scene of scenes) {
  setup(scene.bg)
  patchUiState({ sid: 'd2a6ecf8', theme: scene.theme })

  const intro = renderAnsi(
    <Box flexDirection="column">
      <Banner maxWidth={86} t={scene.theme} />
      <SessionPanel info={info} maxWidth={86} sid="d2a6ecf8" t={scene.theme} />
    </Box>,
    88
  )

  patchOverlayState({})

  const comps = renderAnsi(
    <Box flexDirection="column" height={10} position="relative" width={88}>
      <Box flexGrow={1} />
      <FloatingOverlays
        cols={88}
        compIdx={0}
        completions={completions}
        onActiveSessionClose={pending}
        onActiveSessionSelect={noop}
        onModelSelect={noop}
        onNewLiveSession={noop}
        onNewPromptSession={noop}
        onResumeSelect={noop}
        pagerPageSize={8}
      />
    </Box>,
    88
  )

  const statusLine = renderAnsi(
    <Box flexDirection="column">
      <Text>
        <Text color={scene.theme.color.statusGood}>— ready </Text>
        <Text color={scene.theme.color.muted}>| opus 4.8 fast | 4s | voice off</Text>
      </Text>
      <Text>
        <Text color={scene.theme.color.muted}>{scene.theme.brand.prompt} </Text>
        <Text backgroundColor={scene.theme.color.muted} color={scene.bg}>T</Text>
        <Text color={scene.theme.color.muted}>ry &quot;fix the lint errors&quot;</Text>
      </Text>
    </Box>,
    88
  )

  page += `<div style="background:${scene.bg};color:${scene.fg};padding:14px;border-radius:6px">`
  page += `<div style="font:bold 12px sans-serif;opacity:.6;margin-bottom:8px;color:${scene.fg}">${scene.name}</div>`
  page += `<pre style="margin:0;white-space:pre">${ansiToHtml(intro, scene.fg, scene.bg)}</pre>`
  page += `<pre style="margin:8px 0 0;white-space:pre">${ansiToHtml(comps, scene.fg, scene.bg)}</pre>`
  page += `<pre style="margin:8px 0 0;white-space:pre">${ansiToHtml(statusLine, scene.fg, scene.bg)}</pre>`
  page += `</div>`
}

page += '</div></body>'
writeFileSync('/tmp/tui-visual.html', page)
console.log('wrote /tmp/tui-visual.html')
process.exit(0)
