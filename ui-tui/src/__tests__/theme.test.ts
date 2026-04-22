import { describe, expect, it } from 'vitest'

import { DARK_THEME, DEFAULT_THEME, detectLightMode, fromSkin, LIGHT_THEME } from '../theme.js'

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

describe('LIGHT_THEME', () => {
  it('avoids bright-yellow accents unreadable on white backgrounds (#11300)', () => {
    expect(LIGHT_THEME.color.gold).not.toBe('#FFD700')
    expect(LIGHT_THEME.color.amber).not.toBe('#FFBF00')
    expect(LIGHT_THEME.color.dim).not.toBe('#B8860B')
    expect(LIGHT_THEME.color.statusWarn).not.toBe('#FFD700')
  })

  it('keeps the same shape as DARK_THEME', () => {
    expect(Object.keys(LIGHT_THEME.color).sort()).toEqual(Object.keys(DARK_THEME.color).sort())
    expect(LIGHT_THEME.brand).toEqual(DARK_THEME.brand)
  })
})

describe('DEFAULT_THEME aliasing', () => {
  it('defaults to DARK_THEME when nothing signals light', () => {
    expect(DEFAULT_THEME).toBe(DARK_THEME)
  })
})

describe('detectLightMode', () => {
  it('returns false on empty env', () => {
    expect(detectLightMode({})).toBe(false)
  })

  it('honors HERMES_TUI_LIGHT on/off', () => {
    expect(detectLightMode({ HERMES_TUI_LIGHT: '1' })).toBe(true)
    expect(detectLightMode({ HERMES_TUI_LIGHT: 'true' })).toBe(true)
    expect(detectLightMode({ HERMES_TUI_LIGHT: 'on' })).toBe(true)
    expect(detectLightMode({ HERMES_TUI_LIGHT: '0' })).toBe(false)
    expect(detectLightMode({ HERMES_TUI_LIGHT: 'off' })).toBe(false)
  })

  it('sniffs COLORFGBG bg slots 7 and 15 as light (#11300)', () => {
    expect(detectLightMode({ COLORFGBG: '0;15' })).toBe(true)
    expect(detectLightMode({ COLORFGBG: '0;default;15' })).toBe(true)
    expect(detectLightMode({ COLORFGBG: '0;7' })).toBe(true)
    expect(detectLightMode({ COLORFGBG: '15;0' })).toBe(false)
    expect(detectLightMode({ COLORFGBG: '7;default;0' })).toBe(false)
  })

  it('lets HERMES_TUI_LIGHT=0 override a light COLORFGBG', () => {
    expect(detectLightMode({ COLORFGBG: '0;15', HERMES_TUI_LIGHT: '0' })).toBe(false)
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
