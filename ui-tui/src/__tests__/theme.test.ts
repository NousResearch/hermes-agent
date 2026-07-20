import { afterEach, describe, expect, it, vi } from 'vitest'

// `theme.js` reads `process.env` at module-load to compute DEFAULT_THEME,
// and `fromSkin` closes over DEFAULT_THEME.  A developer shell with
// HERMES_TUI_THEME=light (or HERMES_TUI_BACKGROUND set to something
// bright) would flip the base and turn these assertions into a local-
// only failure.  We sterilize the relevant env vars + dynamically
// import the module fresh so EVERY symbol that closes over the env
// (DEFAULT_THEME, DARK_THEME, LIGHT_THEME, fromSkin) is loaded against
// a known-empty environment.
//
// `detectLightMode` takes env as an explicit arg, so it's safe to import
// statically — but we stay consistent and dynamic-import it too.
const RELEVANT_ENV = [
  'HERMES_TUI_LIGHT',
  'HERMES_TUI_THEME',
  'HERMES_TUI_BACKGROUND',
  'COLORFGBG',
  'COLORTERM',
  'TERM_PROGRAM'
] as const

async function importThemeWithEnv(env: Partial<Record<(typeof RELEVANT_ENV)[number], string>> = {}) {
  for (const key of RELEVANT_ENV) {
    vi.stubEnv(key, env[key] ?? '')
  }

  vi.resetModules()

  return import('../theme.js')
}

async function importThemeWithCleanEnv() {
  return importThemeWithEnv()
}

afterEach(() => {
  vi.unstubAllEnvs()
  vi.resetModules()
})

describe('DEFAULT_THEME', () => {
  it('has brand defaults', async () => {
    const { DEFAULT_THEME } = await importThemeWithCleanEnv()

    expect(DEFAULT_THEME.brand.name).toBe('Hermes Agent')
    expect(DEFAULT_THEME.brand.prompt).toBe('❯')
    expect(DEFAULT_THEME.brand.tool).toBe('┊')
  })

  it('has color palette', async () => {
    const { DEFAULT_THEME } = await importThemeWithCleanEnv()

    expect(DEFAULT_THEME.color.primary).toBe('#FFD700')
    expect(DEFAULT_THEME.color.error).toBe('#ef5350')
  })
})

describe('LIGHT_THEME', () => {
  it('avoids bright-yellow accents unreadable on white backgrounds (#11300)', async () => {
    const { LIGHT_THEME } = await importThemeWithCleanEnv()

    expect(LIGHT_THEME.color.primary).not.toBe('#FFD700')
    expect(LIGHT_THEME.color.accent).not.toBe('#FFBF00')
    expect(LIGHT_THEME.color.muted).not.toBe('#B8860B')
    expect(LIGHT_THEME.color.statusWarn).not.toBe('#FFD700')
  })

  it('keeps the same shape as DARK_THEME', async () => {
    const { DARK_THEME, LIGHT_THEME } = await importThemeWithCleanEnv()

    expect(Object.keys(LIGHT_THEME.color).sort()).toEqual(Object.keys(DARK_THEME.color).sort())
    expect(LIGHT_THEME.brand).toEqual(DARK_THEME.brand)
  })
})

describe('DEFAULT_THEME aliasing', () => {
  it('defaults to DARK_THEME when nothing signals light', async () => {
    const { DEFAULT_THEME, DARK_THEME: DARK } = await importThemeWithCleanEnv()

    expect(DEFAULT_THEME).toBe(DARK)
  })
})

describe('detectLightMode', () => {
  it('returns false on empty env', async () => {
    const { detectLightMode } = await importThemeWithCleanEnv()

    expect(detectLightMode({})).toBe(false)
  })

  it('defaults Apple Terminal to light when no stronger signal is present', async () => {
    const { detectLightMode } = await importThemeWithCleanEnv()

    expect(detectLightMode({ TERM_PROGRAM: 'Apple_Terminal' })).toBe(true)
  })

  it('honors HERMES_TUI_LIGHT on/off', async () => {
    const { detectLightMode } = await importThemeWithCleanEnv()

    expect(detectLightMode({ HERMES_TUI_LIGHT: '1' })).toBe(true)
    expect(detectLightMode({ HERMES_TUI_LIGHT: 'true' })).toBe(true)
    expect(detectLightMode({ HERMES_TUI_LIGHT: 'on' })).toBe(true)
    expect(detectLightMode({ HERMES_TUI_LIGHT: '0' })).toBe(false)
    expect(detectLightMode({ HERMES_TUI_LIGHT: 'off' })).toBe(false)
  })

  it('sniffs COLORFGBG bg slots 7 and 15 as light (#11300)', async () => {
    const { detectLightMode } = await importThemeWithCleanEnv()

    expect(detectLightMode({ COLORFGBG: '0;15' })).toBe(true)
    expect(detectLightMode({ COLORFGBG: '0;default;15' })).toBe(true)
    expect(detectLightMode({ COLORFGBG: '0;7' })).toBe(true)
    expect(detectLightMode({ COLORFGBG: '15;0' })).toBe(false)
    expect(detectLightMode({ COLORFGBG: '7;default;0' })).toBe(false)
  })

  it('falls through on malformed COLORFGBG with empty/non-numeric trailing field', async () => {
    const { detectLightMode } = await importThemeWithCleanEnv()
    // `Number('')` is 0, so `'15;'` would have been read as bg=0
    // (authoritative dark) and incorrectly blocked TERM_PROGRAM.
    // The strict /^\d+$/ guard makes these fall through instead.
    const allowList = new Set(['Apple_Terminal'])

    expect(detectLightMode({ COLORFGBG: '15;', TERM_PROGRAM: 'Apple_Terminal' }, allowList)).toBe(true)
    expect(detectLightMode({ COLORFGBG: 'default;default', TERM_PROGRAM: 'Apple_Terminal' }, allowList)).toBe(true)
    // Without an allow-list match, fall-through still defaults to dark.
    expect(detectLightMode({ COLORFGBG: '15;' })).toBe(false)
  })

  it('lets HERMES_TUI_LIGHT=0 override a light COLORFGBG', async () => {
    const { detectLightMode } = await importThemeWithCleanEnv()

    expect(detectLightMode({ COLORFGBG: '0;15', HERMES_TUI_LIGHT: '0' })).toBe(false)
  })

  it('honors HERMES_TUI_THEME=light/dark as a symmetric explicit override', async () => {
    const { detectLightMode } = await importThemeWithCleanEnv()

    expect(detectLightMode({ HERMES_TUI_THEME: 'light' })).toBe(true)
    expect(detectLightMode({ HERMES_TUI_THEME: 'dark' })).toBe(false)
    expect(detectLightMode({ COLORFGBG: '0;15', HERMES_TUI_THEME: 'dark' })).toBe(false)
    expect(detectLightMode({ COLORFGBG: '15;0', HERMES_TUI_THEME: 'light' })).toBe(true)
  })

  it('uses HERMES_TUI_BACKGROUND luminance when COLORFGBG is missing', async () => {
    const { detectLightMode } = await importThemeWithCleanEnv()

    expect(detectLightMode({ HERMES_TUI_BACKGROUND: '#ffffff' })).toBe(true)
    expect(detectLightMode({ HERMES_TUI_BACKGROUND: '#000000' })).toBe(false)
    expect(detectLightMode({ HERMES_TUI_BACKGROUND: '#1e1e1e' })).toBe(false)
    // Three-char hex normalises like CSS.
    expect(detectLightMode({ HERMES_TUI_BACKGROUND: '#fff' })).toBe(true)
    // Garbage falls through to the default-dark path.
    expect(detectLightMode({ HERMES_TUI_BACKGROUND: 'not-a-colour' })).toBe(false)
  })

  it('rejects partially-invalid hex instead of silently truncating', async () => {
    const { detectLightMode } = await importThemeWithCleanEnv()
    // `parseInt('fffgff'.slice(2,4), 16)` would return 15 — the strict
    // regex must reject these inputs so they fall through to default-
    // dark instead of producing a false-positive light reading.
    expect(detectLightMode({ HERMES_TUI_BACKGROUND: '#fffgff' })).toBe(false)
    expect(detectLightMode({ HERMES_TUI_BACKGROUND: 'ffggff' })).toBe(false)
    expect(detectLightMode({ HERMES_TUI_BACKGROUND: '#xyz' })).toBe(false)
    // Wrong length also rejected (no implicit padding/truncation).
    expect(detectLightMode({ HERMES_TUI_BACKGROUND: '#fffff' })).toBe(false)
    expect(detectLightMode({ HERMES_TUI_BACKGROUND: '#fffffff' })).toBe(false)
  })

  it('treats COLORFGBG as authoritative when present so it dominates the TERM_PROGRAM allow-list', async () => {
    const { detectLightMode } = await importThemeWithCleanEnv()
    // Injecting the allow-list keeps this precedence rule explicit even if
    // production defaults change.
    const allowList = new Set(['Apple_Terminal'])

    // Sanity: the allow-list alone WOULD turn this terminal light.
    expect(detectLightMode({ TERM_PROGRAM: 'Apple_Terminal' }, allowList)).toBe(true)

    // Dark COLORFGBG must beat the allow-list.
    expect(detectLightMode({ COLORFGBG: '15;0', TERM_PROGRAM: 'Apple_Terminal' }, allowList)).toBe(false)
  })
})

describe('fromSkin', () => {
  // `fromSkin` closes over DEFAULT_THEME (which is env-derived), so we
  // must dynamic-import it after sterilizing env — otherwise an ambient
  // HERMES_TUI_THEME=light would flip the base palette and make these
  // assertions order-dependent on the developer's shell.

  it('overrides banner colors', async () => {
    const { fromSkin } = await importThemeWithCleanEnv()

    expect(fromSkin({ banner_title: '#FF0000' }, {}).color.primary).toBe('#FF0000')
  })

  it('preserves unset colors', async () => {
    const { DEFAULT_THEME, fromSkin } = await importThemeWithCleanEnv()

    expect(fromSkin({ banner_title: '#FF0000' }, {}).color.accent).toBe(DEFAULT_THEME.color.accent)
  })

  it('derives completion current background from resolved completion background (polarity-compatible)', async () => {
    // Light terminal + light-authored menu fill: the skin's fill is honored
    // and the current-row derivation mixes off it.
    const { fromSkin } = await importThemeWithEnv({ HERMES_TUI_BACKGROUND: '#ffffff' })

    const theme = fromSkin({ banner_accent: '#000000', completion_menu_bg: '#ffffff' }, {})

    expect(theme.color.completionBg).toBe('#ffffff')
    expect(theme.color.completionCurrentBg).toBe('#bfbfbf')
  })

  it('rejects wrong-polarity fills even when skin-authored (terminal owns the canvas)', async () => {
    // Dark terminal + white menu fill: unlike the desktop app, the TUI cannot
    // paint its own canvas, so cross-polarity fills fall back to the base.
    const { DARK_THEME, fromSkin } = await importThemeWithCleanEnv()

    const theme = fromSkin({ banner_accent: '#000000', completion_menu_bg: '#ffffff' }, {})

    expect(theme.color.completionBg).toBe(DARK_THEME.color.completionBg)
    expect(theme.color.completionCurrentBg).toBe(DARK_THEME.color.completionCurrentBg)
  })

  it('uses active completion color as the selection highlight fallback', async () => {
    const { fromSkin } = await importThemeWithCleanEnv()

    const theme = fromSkin({ completion_menu_current_bg: '#123456' }, {})

    expect(theme.color.selectionBg).toBe('#123456')
  })

  it('maps completion meta background colors from skins', async () => {
    const { fromSkin } = await importThemeWithCleanEnv()

    const theme = fromSkin(
      {
        completion_menu_meta_bg: '#111111',
        completion_menu_meta_current_bg: '#222222'
      },
      {}
    )

    expect(theme.color.completionMetaBg).toBe('#111111')
    expect(theme.color.completionMetaCurrentBg).toBe('#222222')
  })

  it('lets selection_bg override completion highlight colors', async () => {
    const { fromSkin } = await importThemeWithCleanEnv()

    const theme = fromSkin({ completion_menu_current_bg: '#123456', selection_bg: '#654321' }, {})

    expect(theme.color.selectionBg).toBe('#654321')
  })

  it('overrides branding', async () => {
    const { fromSkin } = await importThemeWithCleanEnv()
    const { brand } = fromSkin({}, { agent_name: 'TestBot', prompt_symbol: '$' })

    expect(brand.name).toBe('TestBot')
    expect(brand.prompt).toBe('$')
  })

  it('normalizes skin prompt symbols to trimmed single-line text', async () => {
    const { DEFAULT_THEME, fromSkin } = await importThemeWithCleanEnv()

    expect(fromSkin({}, { prompt_symbol: ' ⚔ ❯ \n' }).brand.prompt).toBe('⚔ ❯')
    expect(fromSkin({}, { prompt_symbol: ' Ψ > \n' }).brand.prompt).toBe('Ψ >')
    expect(fromSkin({}, { prompt_symbol: '\n\t' }).brand.prompt).toBe(DEFAULT_THEME.brand.prompt)
  })

  it('defaults for empty skin', async () => {
    const { DEFAULT_THEME, fromSkin } = await importThemeWithCleanEnv()

    expect(fromSkin({}, {}).color).toEqual(DEFAULT_THEME.color)
    expect(fromSkin({}, {}).brand.icon).toBe(DEFAULT_THEME.brand.icon)
  })

  it('normalizes non-banner foregrounds on light Apple Terminal', async () => {
    const { fromSkin } = await importThemeWithEnv({ TERM_PROGRAM: 'Apple_Terminal' })

    const theme = fromSkin(
      {
        banner_accent: '#FFBF00',
        banner_border: '#CD7F32',
        banner_dim: '#B8860B',
        banner_text: '#FFF8DC',
        banner_title: '#FFD700',
        prompt: '#FFF8DC'
      },
      {}
    )

    expect(theme.color.primary).toBe('#FFD700')
    expect(theme.color.accent).toBe('#FFBF00')
    expect(theme.color.border).toBe('#CD7F32')
    expect(theme.color.muted).toBe('ansi256(245)')
    expect(theme.color.text).toBe('ansi256(136)')
    expect(theme.color.prompt).toBe('ansi256(136)')
  })

  it('keeps truecolor light Apple Terminal in truecolor (adapting, not ansi256-bucketing)', async () => {
    const { fromSkin } = await importThemeWithEnv({ COLORTERM: 'truecolor', TERM_PROGRAM: 'Apple_Terminal' })
    const theme = fromSkin({ banner_text: '#FFF8DC' }, {})

    // No ansi256 bucketing on truecolor terminals — but a cream foreground on
    // a light background is exactly the washout the adaptation exists to fix,
    // so the value is clamped to a readable truecolor hex rather than kept.
    expect(theme.color.text).toMatch(/^#[0-9a-f]{6}$/i)
    expect(luminance(theme.color.text)).toBeLessThanOrEqual(0.45)
  })

  it('normalizes Apple Terminal names before matching', async () => {
    const { fromSkin } = await importThemeWithEnv({ TERM_PROGRAM: ' Apple_Terminal ' })
    const theme = fromSkin({ banner_text: '#FFF8DC' }, {})

    expect(theme.color.text).toBe('ansi256(136)')
  })

  it('passes banner logo/hero', async () => {
    const { fromSkin } = await importThemeWithCleanEnv()

    expect(fromSkin({}, {}, 'LOGO', 'HERO').bannerLogo).toBe('LOGO')
    expect(fromSkin({}, {}, 'LOGO', 'HERO').bannerHero).toBe('HERO')
  })

  it('maps ui_ color keys + cascades to status', async () => {
    const { fromSkin } = await importThemeWithCleanEnv()
    const { color } = fromSkin({ ui_ok: '#008000' }, {})

    // The exact value may be contrast-lifted against the background; the
    // contract is the cascade (ok drives statusGood) and the hue surviving.
    expect(color.statusGood).toBe(color.ok)
    expect(color.ok).toMatch(/^#[0-9a-f]{6}$/i)
    expect(luminance(color.ok)).toBeGreaterThan(0)
  })
})

// Rec. 709-ish relative luminance, local to the test so assertions are
// independent of the implementation under test.
const luminance = (hex: string): number => {
  const n = parseInt(hex.replace('#', ''), 16)

  const channel = (v: number) => {
    const c = v / 255

    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
  }

  return 0.2126 * channel((n >> 16) & 0xff) + 0.7152 * channel((n >> 8) & 0xff) + 0.0722 * channel(n & 0xff)
}

// The bundled slate skin's actual color block — dark-authored (pale pastels,
// no completion/selection backgrounds defined).
const SLATE_COLORS = {
  banner_accent: '#8EA8FF',
  banner_border: '#4169e1',
  banner_dim: '#4b5563',
  banner_text: '#c9d1d9',
  banner_title: '#7eb8f6',
  prompt: '#c9d1d9',
  session_border: '#4b5563',
  session_label: '#7eb8f6',
  ui_accent: '#7eb8f6',
  ui_error: '#F7A072',
  ui_label: '#8EA8FF',
  ui_ok: '#63D0A6',
  ui_warn: '#e6a855'
}

describe('background-aware adaptation (OSC-11 light terminals)', () => {
  it('keeps a dark-authored skin readable on a light background', async () => {
    const { contrastRatio, fromSkin } = await importThemeWithEnv({ HERMES_TUI_BACKGROUND: '#ffffff' })
    const { color } = fromSkin(SLATE_COLORS, {})

    // Foreground roles must clear WCAG contrast against the actual white
    // background — hue survives, washout doesn't.
    for (const key of ['text', 'prompt', 'accent', 'label', 'ok', 'error', 'primary'] as const) {
      expect(contrastRatio(color[key], '#ffffff'), `${key} ${color[key]}`).toBeGreaterThanOrEqual(3.8)
    }

    // Softer roles (muted/warn/border) get a looser but real floor.
    for (const key of ['muted', 'warn', 'border'] as const) {
      expect(contrastRatio(color[key], '#ffffff'), `${key} ${color[key]}`).toBeGreaterThanOrEqual(2.7)
    }

    // Background roles the skin never defined must be light-polarity fills,
    // not the dark base's navy.
    for (const key of ['completionBg', 'completionCurrentBg', 'statusBg', 'selectionBg'] as const) {
      expect(luminance(color[key]), `${key} ${color[key]}`).toBeGreaterThanOrEqual(0.4)
    }
  })

  it('leaves the same skin untouched on a dark background', async () => {
    const { fromSkin } = await importThemeWithEnv({ HERMES_TUI_BACKGROUND: '#1e1e2e' })
    const { color } = fromSkin(SLATE_COLORS, {})

    expect(color.text).toBe('#c9d1d9')
    expect(color.accent).toBe('#7eb8f6')
    expect(luminance(color.completionBg)).toBeLessThanOrEqual(0.35)
  })

  it('empty skin on a light background resolves to the light base palette', async () => {
    const { fromSkin, LIGHT_THEME } = await importThemeWithEnv({ HERMES_TUI_BACKGROUND: '#ffffff' })

    expect(fromSkin({}, {}).color).toEqual(LIGHT_THEME.color)
  })

  it('base palettes are fixed points of the adaptation', async () => {
    const dark = await importThemeWithCleanEnv()

    expect(dark.fromSkin({}, {}).color).toEqual(dark.DARK_THEME.color)

    const light = await importThemeWithEnv({ HERMES_TUI_BACKGROUND: '#fafafa' })

    expect(light.fromSkin({}, {}).color).toEqual(light.LIGHT_THEME.color)
  })

  it('defaultThemeForCurrentBackground follows a late HERMES_TUI_BACKGROUND write', async () => {
    const { DEFAULT_THEME, defaultThemeForCurrentBackground, LIGHT_THEME } = await importThemeWithCleanEnv()

    // Module loaded dark (clean env)…
    expect(DEFAULT_THEME.color.completionBg).toBe('#1a1a2e')

    // …then the OSC-11 answer lands and is cached into the env slot.
    expect(defaultThemeForCurrentBackground({ HERMES_TUI_BACKGROUND: '#ffffff' }).color).toEqual(LIGHT_THEME.color)
  })
})
