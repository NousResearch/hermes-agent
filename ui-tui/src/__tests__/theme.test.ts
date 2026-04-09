import { describe, expect, it } from 'vitest'

import { DEFAULT_THEME, fromSkin } from '../theme.js'


describe('DEFAULT_THEME', () => {

  it('has brand defaults', () => {
    expect(DEFAULT_THEME.brand.name).toBe('Hermes Agent')
    expect(DEFAULT_THEME.brand.prompt).toBe('❯')
    expect(DEFAULT_THEME.brand.tool).toBe('┊')
  })

  it('has color palette', () => {
    expect(DEFAULT_THEME.color.gold).toBe('#FFD700')
    expect(DEFAULT_THEME.color.error).toBe('#ef5350')
  })
})


describe('fromSkin', () => {

  it('overrides banner colors', () => {
    expect(fromSkin({ banner_title: '#FF0000' }, {}).color.gold).toBe('#FF0000')
  })

  it('preserves unset colors', () => {
    expect(fromSkin({ banner_title: '#FF0000' }, {}).color.amber).toBe(DEFAULT_THEME.color.amber)
  })

  it('overrides branding', () => {
    const { brand } = fromSkin({}, { agent_name: 'TestBot', prompt_symbol: '$' })
    expect(brand.name).toBe('TestBot')
    expect(brand.prompt).toBe('$')
  })

  it('defaults for empty skin', () => {
    expect(fromSkin({}, {}).color).toEqual(DEFAULT_THEME.color)
    expect(fromSkin({}, {}).brand.icon).toBe(DEFAULT_THEME.brand.icon)
  })

  it('passes banner logo/hero', () => {
    expect(fromSkin({}, {}, 'LOGO', 'HERO').bannerLogo).toBe('LOGO')
    expect(fromSkin({}, {}, 'LOGO', 'HERO').bannerHero).toBe('HERO')
  })

  it('maps ui_ color keys + cascades to status', () => {
    const { color } = fromSkin({ ui_ok: '#008000' }, {})
    expect(color.ok).toBe('#008000')
    expect(color.statusGood).toBe('#008000')
  })
})
