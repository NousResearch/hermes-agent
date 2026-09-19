import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import { stripAnsi } from '@hermes/shared/ansi'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { sessionCommands } from '../app/slash/commands/session.js'
import { getUiState } from '../app/uiStore.js'
import { StatusRule } from '../components/appChrome.js'
import { MessageLine } from '../components/messageLine.js'
import { ToolTrail } from '../components/thinking.js'
import { statusGlyph } from '../lib/subagentGlyph.js'
import {
  DEFAULT_GLYPH_PRESET,
  DEFAULT_THEME,
  getGlyphPreset,
  GLYPH_PRESETS,
  GLYPH_TABLES,
  glyphTable,
  normalizeGlyphPreset,
  setGlyphPreset,
  spinnerFrames,
  themeWithGlyphPreset
} from '../theme.js'

type GlyphTable = (typeof GLYPH_TABLES)[keyof typeof GLYPH_TABLES]

const isAscii = (value: string) => [...value].every(ch => (ch.codePointAt(0) ?? 0x80) < 0x7f)

/** Every string a preset table can draw: slots, spinner frames, status map. */
const tableStrings = (table: GlyphTable): string[] =>
  Object.values(table).flatMap(value =>
    typeof value === 'string'
      ? [value]
      : Array.isArray(value)
        ? [...(value as readonly string[])]
        : Object.values(value as Record<string, string>)
  )

const ascii = themeWithGlyphPreset(DEFAULT_THEME, 'ascii')

const renderText = (element: React.ReactElement, columns = 80) => {
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

  const instance = renderSync(element, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  const text = stripAnsi(output)

  instance.unmount()

  return text
}

// StatusRule is a pure prop→element function, so glyph assertions can read the
// element tree directly instead of driving a terminal (same approach as
// appChromeStatusRule.test.tsx).
type ReactNodeLike = React.ReactNode

const textContent = (node: ReactNodeLike): string => {
  if (node === null || node === undefined || typeof node === 'boolean') {
    return ''
  }

  if (typeof node === 'string' || typeof node === 'number') {
    return String(node)
  }

  if (Array.isArray(node)) {
    return node.map(textContent).join('')
  }

  if (React.isValidElement(node)) {
    return textContent(node.props.children)
  }

  return ''
}

const statusRuleProps = {
  bgCount: 2,
  busy: false,
  cols: 200,
  compacting: false,
  cwdLabel: '~/repo',
  liveSessionCount: 2,
  model: 'opus-4.8',
  sessionStartedAt: Date.now() - 60_000,
  status: 'ready',
  statusColor: DEFAULT_THEME.color.ok,
  turnStartedAt: null,
  usage: {
    active_subagents: 2,
    avg_latency_s: 1.4,
    avg_tps: 42,
    cache_hit_pct: 80,
    compressions: 3,
    context_max: 200_000,
    context_percent: 25,
    context_used: 50_000,
    total: 50_000
  },
  voiceLabel: ''
}

describe('tui.glyph_preset — resolution', () => {
  it('ships exactly the nerd / unicode / ascii tiers', () => {
    expect([...GLYPH_PRESETS]).toEqual(['ascii', 'nerd', 'unicode'])
    expect(Object.keys(GLYPH_TABLES).sort()).toEqual(['ascii', 'nerd', 'unicode'])
  })

  it('falls back to the unicode tier for a missing or unknown preset', () => {
    expect(DEFAULT_GLYPH_PRESET).toBe('unicode')

    for (const raw of [undefined, null, '', '   ', 'Rainbow', 42, {}, [], true]) {
      expect(normalizeGlyphPreset(raw), `raw=${String(raw)}`).toBe(DEFAULT_GLYPH_PRESET)
      expect(glyphTable(raw)).toBe(GLYPH_TABLES.unicode)
    }
  })

  it('accepts the three configured names, case- and whitespace-insensitively', () => {
    for (const preset of GLYPH_PRESETS) {
      expect(normalizeGlyphPreset(preset)).toBe(preset)
      expect(normalizeGlyphPreset(` ${preset.toUpperCase()} `)).toBe(preset)
      expect(glyphTable(preset)).toBe(GLYPH_TABLES[preset])
    }
  })

  it('covers the same slots and status keys in every tier', () => {
    const reference = GLYPH_TABLES.unicode

    for (const table of Object.values(GLYPH_TABLES)) {
      expect(Object.keys(table).sort()).toEqual(Object.keys(reference).sort())
      expect(Object.keys(table.status).sort()).toEqual(Object.keys(reference.status).sort())
      expect(table.spinner.length).toBeGreaterThan(1)
      // One column per frame keeps the status-bar width budget honest.
      expect(table.spinner.every(frame => frame.length === 1)).toBe(true)
    }
  })
})

describe('default tier is the classic Hermes chrome', () => {
  const unicode = GLYPH_TABLES.unicode

  it('keeps every glyph the pre-preset TUI drew', () => {
    expect(unicode.status.running).toBe('●')
    expect(unicode.status.queued).toBe('○')
    expect(unicode.status.completed).toBe('✓')
    expect(unicode.status.failed).toBe('✗')
    expect(unicode.status.error).toBe('⚠')
    expect(unicode.ok).toBe('✓')
    expect(unicode.fail).toBe('✗')
    expect(unicode.warn).toBe('⚠')
    expect(unicode.railMid).toBe('├─')
    expect(unicode.railLast).toBe('└─')
    expect(unicode.railPipe).toBe('│')
    expect(unicode.dot).toBe('·')
    expect(unicode.tool).toBe('⚡')
    expect(unicode.spinner).toContain('⠋')
  })

  it('is what DEFAULT_THEME renders with, so no existing setup changes', () => {
    expect(DEFAULT_THEME.glyphs).toBe(unicode)
  })

  it('treats the nerd tier as Nerd Font icons over standard-unicode rails', () => {
    const nerd = GLYPH_TABLES.nerd

    // Nerd Font icon slots live in the private-use area.
    expect(nerd.ok.codePointAt(0)).toBe(0xf00c)
    expect(nerd.fail.codePointAt(0)).toBe(0xf00d)
    expect(nerd.status.running.codePointAt(0)).toBe(0xf110)
    // Patched fonts keep the standard box-drawing/braille ranges.
    expect(nerd.railMid).toBe(unicode.railMid)
    expect(nerd.railLast).toBe(unicode.railLast)
    expect(nerd.spinner).toEqual(unicode.spinner)
  })
})

describe('ascii tier never emits a glyph a 7-bit terminal cannot draw', () => {
  it('every string in the table is ASCII', () => {
    for (const value of tableStrings(GLYPH_TABLES.ascii)) {
      expect(isAscii(value), JSON.stringify(value)).toBe(true)
    }
  })

  it('uses the documented floor tokens', () => {
    const table = GLYPH_TABLES.ascii

    expect(table.ok).toBe('[ok]')
    expect(table.fail).toBe('[x]')
    expect(table.warn).toBe('[!]')
    expect(table.status.running).toBe('[~]')
    expect(table.status.completed).toBe('[ok]')
    expect(table.railMid).toBe('|-')
    expect(table.railLast).toBe('`-')
    expect(table.railPipe).toBe('|')
    expect(table.spinner).toEqual(['|', '/', '-', '\\'])
  })

  it('swaps decorative braille art for the preset frames', () => {
    const art = ['⠋', '⠙']

    expect(spinnerFrames(GLYPH_TABLES.ascii, art)).toEqual(GLYPH_TABLES.ascii.spinner)
    // nerd/unicode keep the caller's animation.
    expect(spinnerFrames(GLYPH_TABLES.unicode, art)).toEqual(art)
    expect(spinnerFrames(GLYPH_TABLES.nerd, art)).toEqual(art)
    // …and always have a cycle even with no art to fall back on.
    expect(spinnerFrames(GLYPH_TABLES.unicode)).toEqual(GLYPH_TABLES.unicode.spinner)
  })
})

describe('ascii tier in the transcript', () => {
  it('renders a tool trail with zero non-ASCII glyphs', () => {
    const text = renderText(
      React.createElement(ToolTrail, {
        reasoning: 'Thinking about the shape of the fix.',
        reasoningActive: true,
        sections: { thinking: 'expanded', tools: 'expanded' },
        t: ascii,
        trail: ['read_file(src/a.ts) (0.1s) ✓', 'grep(pattern) (0.0s) ✗']
      } as never)
    )

    expect(text).toContain('Thinking about the shape of the fix.')
    expect(isAscii(text), text).toBe(true)
  })

  it('renders the response rail and role gutter with zero non-ASCII glyphs', () => {
    const text = renderText(
      React.createElement(
        React.Fragment,
        null,
        React.createElement(MessageLine, {
          cols: 80,
          msg: { role: 'assistant', text: 'Done.', thinking: 'Checked the diff.', tools: ['edit(x.ts) 5ms ✓'] },
          t: ascii
        }),
        React.createElement(MessageLine, {
          cols: 80,
          msg: { kind: 'event', role: 'system', text: 'model switch: opus' },
          t: ascii
        })
      )
    )

    expect(isAscii(text), text).toBe(true)
  })

  it('renders every subagent status icon in ASCII', () => {
    const statuses = [
      'cancelled',
      'completed',
      'dispatched',
      'error',
      'failed',
      'finalizing',
      'interrupted',
      'queued',
      'rejected',
      'running',
      'timeout',
      'something-new'
    ]

    for (const status of statuses) {
      const { glyph } = statusGlyph(status, ascii)

      expect(isAscii(glyph), `${status} → ${JSON.stringify(glyph)}`).toBe(true)
    }
  })

  it('renders the whole status rule in ASCII, including the busy spinner', () => {
    const idle = textContent(
      StatusRule({
        ...statusRuleProps,
        battery: { available: true, category: 'good', percent: 80, plugged: false },
        focusView: true,
        lastTurnEndedAt: Date.now() - 5000,
        t: ascii
      } as never)
    )

    const busy = textContent(
      StatusRule({
        ...statusRuleProps,
        busy: true,
        indicatorStyle: 'unicode',
        turnStartedAt: Date.now() - 8000,
        t: ascii
      } as never)
    )

    expect(idle).toContain('ready')
    expect(isAscii(idle), idle).toBe(true)
    expect(isAscii(busy), busy).toBe(true)
    // The busy slot is the `unicode` indicator style, which draws the preset's
    // spinner cycle — never braille under the ascii tier.
    expect(busy).not.toContain('⠋')
  })

  it('keeps the classic chrome rendered under the default tier', () => {
    expect(isAscii(textContent(StatusRule({ ...statusRuleProps, t: DEFAULT_THEME } as never)))).toBe(false)
  })
})

describe('/glyphs picker', () => {
  const command = sessionCommands.find(cmd => cmd.name === 'glyphs')!

  const buildCtx = (result: unknown) => {
    const sys = vi.fn()
    const rpc = vi.fn(() => Promise.resolve(result))

    const ctx = {
      gateway: { rpc },
      guarded:
        <T>(fn: (r: T) => void) =>
        (r: null | T) => {
          if (r) {
            fn(r)
          }
        },
      guardedErr: vi.fn(),
      sid: 'sid-1',
      stale: () => false,
      transcript: { sys }
    }

    const run = async (arg: string) => {
      command.run(arg, ctx as never, 'glyphs')
      await Promise.resolve()
      await Promise.resolve()
      await Promise.resolve()
    }

    return { printed: () => sys.mock.calls.map(c => String(c[0])).join('\n'), rpc, run, sys }
  }

  afterEach(() => setGlyphPreset(DEFAULT_GLYPH_PRESET))

  it('is registered with every tier spelled out in its usage', () => {
    expect(command.usage).toContain(GLYPH_PRESETS.join('|'))
    expect(command.help).toContain('nerd')
  })

  it('prints one live sample row per tier, so the pick is made in this terminal', async () => {
    const { printed, run, sys } = buildCtx({ value: 'unicode' })

    await run('')

    const text = printed()

    expect(sys).toHaveBeenCalled()
    expect(text).toContain('glyphs: unicode')

    for (const preset of GLYPH_PRESETS) {
      expect(text).toContain(preset)
      // Each row shows that tier's own marks, e.g. ascii's `[ok]` next to nerd's icon.
      expect(text).toContain(GLYPH_TABLES[preset].ok)
      expect(text).toContain(GLYPH_TABLES[preset].status.running)
    }
  })

  it('persists a valid tier and re-renders the running session from it', async () => {
    const { rpc, run } = buildCtx({ key: 'glyphs', value: 'ascii' })

    expect(getUiState().theme.glyphs).toBe(GLYPH_TABLES.unicode)

    await run('ascii')

    expect(rpc).toHaveBeenCalledWith('config.set', { key: 'glyphs', value: 'ascii' })
    // No restart: the committed theme carries the new table, which is what
    // every chrome consumer reads on its next render.
    expect(getGlyphPreset()).toBe('ascii')
    expect(getUiState().theme.glyphs).toBe(GLYPH_TABLES.ascii)
    expect(getUiState().theme.brand.tool).toBe(GLYPH_TABLES.ascii.gutter)
  })

  it('refuses an unknown tier without touching config', async () => {
    const { rpc, run, sys } = buildCtx({ value: 'wingdings' })

    await run('wingdings')

    expect(rpc).not.toHaveBeenCalled()
    expect(String(sys.mock.calls[0]?.[0])).toContain('usage: /glyphs')
    expect(getGlyphPreset()).toBe(DEFAULT_GLYPH_PRESET)
  })
})
