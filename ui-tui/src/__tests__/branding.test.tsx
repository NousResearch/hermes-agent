import { renderSync } from '@hermes/ink'
import React from 'react'
import { PassThrough } from 'stream'
import { describe, expect, it } from 'vitest'

import { Banner, SessionPanel } from '../components/branding.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME, fromSkin, type Theme } from '../theme.js'
import type { SessionInfo } from '../types.js'

function renderNode(node: React.ReactElement, columns = 80) {
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

  const instance = renderSync(node, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  instance.unmount()
  instance.cleanup()

  return stripAnsi(output)
}

function renderBanner(t = DEFAULT_THEME, columns = 80) {
  return renderNode(React.createElement(Banner, { t }), columns)
}

function renderSessionPanel(t = DEFAULT_THEME, columns = 120) {
  const info: SessionInfo = {
    cwd: '/tmp/project',
    model: 'openai/gpt-5.4',
    skills: {},
    tools: {},
    version: '0.0.1'
  }

  return renderNode(React.createElement(SessionPanel, { info, sid: 'sess-1', t }), columns)
}

function customBrandTheme(name = 'HELIX', overrides: Partial<Theme> = {}) {
  const t = fromSkin({}, { agent_name: name })

  return {
    ...t,
    ...overrides,
    brand: {
      ...t.brand,
      ...(overrides.brand ?? {})
    }
  }
}

describe('Banner branding', () => {
  it('keeps default branding in narrow mode', () => {
    const rendered = renderBanner(DEFAULT_THEME, 60)

    expect(rendered).toContain('NOUS HERMES')
    expect(rendered).toContain('Nous Research · Messenger of the Digital Gods')
  })

  it('uses display.skin.branding.agent_name in narrow mode instead of hardcoded Nous strings', () => {
    const rendered = renderBanner(customBrandTheme(), 60)

    expect(rendered).toContain('HELIX')
    expect(rendered).toContain('HELIX · AI Agent')
    expect(rendered).not.toContain('NOUS HERMES')
    expect(rendered).not.toContain('Nous Research · Messenger of the Digital Gods')
  })

  it('falls back to text branding for custom brands in wide mode', () => {
    const rendered = renderBanner(customBrandTheme(), 200)

    expect(rendered).toContain('HELIX')
    expect(rendered).toContain('HELIX · AI Agent')
    expect(rendered).not.toContain('NOUS HERMES')
  })

  it('still renders explicit custom banner logo art for custom brands', () => {
    const rendered = renderBanner(customBrandTheme('HELIX', {
      bannerLogo: '[#ffffff]CUSTOM HELIX ART[/]'
    }), 200)

    expect(rendered).toContain('CUSTOM HELIX ART')
    expect(rendered).not.toContain('NOUS HERMES')
  })
})

describe('SessionPanel branding', () => {
  it('replaces the hardcoded Nous Research label for custom brands in wide mode', () => {
    const rendered = renderSessionPanel(customBrandTheme(), 140)

    expect(rendered).toContain('gpt-5.4 · HELIX')
    expect(rendered).not.toContain('Nous Research')
  })

  it('falls back to text branding instead of default hero art for custom brands without bannerHero', () => {
    const rendered = renderSessionPanel(customBrandTheme(), 140)

    expect(rendered).toContain('⚕ HELIX')
    expect(rendered).not.toContain('⢀⣀⡀')
  })
})
