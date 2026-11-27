import { describe, expect, it } from 'vitest'

import { artWidth, logo, parseRichMarkup } from '../banner.js'

const hasMarkupLeak = (lines: [string, string][]) =>
  lines.some(([, text]) => /\[(?:bold\s+|dim\s+)?#|\[\/\]/.test(text))

// Multi-line block shape used by real skins (kaylee-agent, etc.): one open tag
// on the first line, art in the middle, closing [/] on the last line.
const MULTILINE_LOGO = `[bold #1a5fb4]██╗  ██╗ █████╗ ██╗   ██╗██╗     ███████╗███████╗     █████╗  ██████╗ ███████╗███╗   ██╗████████╗
██║ ██╔╝██╔══██╗╚██╗ ██╔╝██║     ██╔════╝██╔════╝    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
█████╔╝ ███████║ ╚████╔╝ ██║     █████╗  █████╗      ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║
██╔═██╗ ██╔══██║  ╚██╔╝  ██║     ██╔══╝  ██╔══╝      ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║
██║  ██╗██║  ██║   ██║   ███████╗███████╗███████╗    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║
╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚══════╝╚══════╝    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝[/]`

const MULTILINE_HERO = `[#62a0ea]⡏⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⢉⣉⣉⣉⣉⣉⣉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⢹
⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣤⣤⣤⣤⣀⠉⠙⠻⣿⣿⣿⣷⣶⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸
⣇⣀⣀⣀⣀⣀⣀⣀⣠⣾⣿⣿⣿⣛⣻⣣⣋⣀⣀⣠⣊⣰⣋⣀⣀⣀⣀⣀⣀⣀⣀⣀⣛⣀⣀⣀⣀⣀⣸[/]`

describe('parseRichMarkup', () => {
  it('keeps per-line open/close tags (classic Hermes logo style)', () => {
    const markup = ['[bold #FFD700]LINE ONE[/]', '[#FFBF00]LINE TWO[/]', '[dim #CD7F32]LINE THREE[/]'].join(
      '\n'
    )

    const lines = parseRichMarkup(markup)
    expect(lines).toEqual([
      ['#FFD700', 'LINE ONE'],
      ['#FFBF00', 'LINE TWO'],
      ['#CD7F32', 'LINE THREE']
    ])
    expect(hasMarkupLeak(lines)).toBe(false)
  })

  it('carries a multi-line open tag until the closing [/]', () => {
    const markup = ['[bold #1a5fb4]AAA', 'BBB', 'CCC[/]'].join('\n')

    const lines = parseRichMarkup(markup)
    expect(lines).toEqual([
      ['#1a5fb4', 'AAA'],
      ['#1a5fb4', 'BBB'],
      ['#1a5fb4', 'CCC']
    ])
    expect(hasMarkupLeak(lines)).toBe(false)
  })

  it('does not emit a row for a trailing close-only line', () => {
    const markup = ['[#62a0ea]art', '[/]'].join('\n')
    expect(parseRichMarkup(markup)).toEqual([['#62a0ea', 'art']])
  })

  it('preserves plain lines and blank rows', () => {
    expect(parseRichMarkup('plain\n\nstill plain')).toEqual([
      ['', 'plain'],
      ['', ' '],
      ['', 'still plain']
    ])
  })

  it('resets color after [/] mid-stream', () => {
    const markup = ['[#ff0000]red', 'still red[/]', 'plain again'].join('\n')
    expect(parseRichMarkup(markup)).toEqual([
      ['#ff0000', 'red'],
      ['#ff0000', 'still red'],
      ['', 'plain again']
    ])
  })

  it('parses multi-line skin banner_logo without leaking tags', () => {
    const lines = parseRichMarkup(MULTILINE_LOGO)
    expect(lines).toHaveLength(6)
    expect(hasMarkupLeak(lines)).toBe(false)
    expect(lines.every(([color]) => color === '#1a5fb4')).toBe(true)
    expect(lines[0]![1]).toMatch(/^██/)
    expect(lines[5]![1]).toMatch(/╚═╝/)
    expect(artWidth(lines)).toBeGreaterThan(80)
  })

  it('parses multi-line skin banner_hero without leaking tags', () => {
    const lines = parseRichMarkup(MULTILINE_HERO)
    expect(lines).toHaveLength(3)
    expect(hasMarkupLeak(lines)).toBe(false)
    expect(lines.every(([color]) => color === '#62a0ea')).toBe(true)
  })

  it('logo() routes custom markup through the parser', () => {
    const colors = {
      primary: '#fff',
      accent: '#0ff',
      border: '#888',
      muted: '#444',
      text: '#eee',
      danger: '#f00',
      ok: '#0f0',
      warn: '#ff0',
      surface: '#111',
      background: '#000'
    }
    const lines = logo(colors as never, '[#00b4d8]HI\nTHERE[/]')
    expect(lines).toEqual([
      ['#00b4d8', 'HI'],
      ['#00b4d8', 'THERE']
    ])
  })
})
