import { createContext, type ReactNode, useContext } from 'react'

export const GLYPH_PRESETS = ['nerd', 'unicode', 'ascii'] as const
export type GlyphPreset = (typeof GLYPH_PRESETS)[number]
export const DEFAULT_GLYPH_PRESET: GlyphPreset = 'unicode'

export interface ChromeGlyphs {
  disclosureClosed: string
  disclosureOpen: string
  error: string
  info: string
  skipped: string
  success: string
  toolBullet: string
  treeLast: string
  treeMid: string
  treePipe: string
  warn: string
}

const UNICODE: ChromeGlyphs = {
  disclosureClosed: '▸ ',
  disclosureOpen: '▾ ',
  error: '✗',
  info: '·',
  skipped: '○',
  success: '✓',
  toolBullet: '● ',
  treeLast: '└─ ',
  treeMid: '├─ ',
  treePipe: '│ ',
  warn: '!'
}

const NERD: ChromeGlyphs = {
  disclosureClosed: ' ',
  disclosureOpen: ' ',
  error: '',
  info: '',
  skipped: '',
  success: '',
  toolBullet: ' ',
  treeLast: '└─ ',
  treeMid: '├─ ',
  treePipe: '│ ',
  warn: ''
}

const ASCII: ChromeGlyphs = {
  disclosureClosed: '> ',
  disclosureOpen: 'v ',
  error: '[x]',
  info: '.',
  skipped: '-',
  success: '[ok]',
  toolBullet: '* ',
  treeLast: '\\- ',
  treeMid: '+- ',
  treePipe: '| ',
  warn: '!'
}

const BY_PRESET: Readonly<Record<GlyphPreset, ChromeGlyphs>> = {
  ascii: ASCII,
  nerd: NERD,
  unicode: UNICODE
}

export const glyphsForPreset = (preset: GlyphPreset): ChromeGlyphs => BY_PRESET[preset]

const GlyphContext = createContext<ChromeGlyphs>(UNICODE)

export function GlyphProvider({ children, preset }: { children: ReactNode; preset: GlyphPreset }) {
  return <GlyphContext.Provider value={glyphsForPreset(preset)}>{children}</GlyphContext.Provider>
}

export const useChromeGlyphs = (): ChromeGlyphs => useContext(GlyphContext)
