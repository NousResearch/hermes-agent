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
  'TERM_PROGRAM',
] as const

async function importThemeWithCleanEnv() {
  for (const key of RELEVANT_ENV) {
    vi.stubEnv(key, '')
  }
  vi.resetModules()
  return import('../theme.js')
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

    expect(DEFAULT_THEME.color.gold).toBe('#FFD700')
    expect(DEFAULT_THEME.color.error).toBe('#ef5350')
  })
})

describe('LIGHT_THEME', () => {
  it('avoids bright-yellow accents unreadable on white backgrounds (#11300)', async () => {
    const { LIGHT_THEME } = await importThemeWithCleanEnv()

    expect(LIGHT_THEME.color.gold).not.toBe('#FFD700')
    expect(LIGHT_THEME.color.amber).not.toBe('#FFBF00')
    expect(LIGHT_THEME.color.dim).not.toBe('#B8860B')
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
    // Inject a light-default allow-list so the precedence test is
    // meaningful even though the production allow-list is empty.
    const allowList = new Set(['Apple_Terminal'])

    // Sanity: the allow-list alone WOULD turn this terminal light.
    expect(detectLightMode({ TERM_PROGRAM: 'Apple_Terminal' }, allowList)).toBe(true)

    // Dark COLORFGBG must beat the allow-list.
    expect(
      detectLightMode({ COLORFGBG: '15;0', TERM_PROGRAM: 'Apple_Terminal' }, allowList),
    ).toBe(false)
  })
})

describe('fromSkin', () => {
  // `fromSkin` closes over DEFAULT_THEME (which is env-derived), so we
  // must dynamic-import it after sterilizing env — otherwise an ambient
  // HERMES_TUI_THEME=light would flip the base palette and make these
  // assertions order-dependent on the developer's shell.

  it('overrides banner colors', async () => {
    const { fromSkin } = await importThemeWithCleanEnv()

    expect(fromSkin({ banner_title: '#FF0000' }, {}).color.gold).toBe('#FF0000')
  })

  it('preserves unset colors', async () => {
    const { DEFAULT_THEME, fromSkin } = await importThemeWithCleanEnv()

    expect(fromSkin({ banner_title: '#FF0000' }, {}).color.amber).toBe(DEFAULT_THEME.color.amber)
  })

  it('overrides branding', async () => {
    const { fromSkin } = await importThemeWithCleanEnv()
    const { brand } = fromSkin({}, { agent_name: 'TestBot', prompt_symbol: '$' })

    expect(brand.name).toBe('TestBot')
    expect(brand.prompt).toBe('$')
  })

  it('defaults for empty skin', async () => {
    const { DEFAULT_THEME, fromSkin } = await importThemeWithCleanEnv()

    expect(fromSkin({}, {}).color).toEqual(DEFAULT_THEME.color)
    expect(fromSkin({}, {}).brand.icon).toBe(DEFAULT_THEME.brand.icon)
  })

  it('passes banner logo/hero', async () => {
    const { fromSkin } = await importThemeWithCleanEnv()

    expect(fromSkin({}, {}, 'LOGO', 'HERO').bannerLogo).toBe('LOGO')
    expect(fromSkin({}, {}, 'LOGO', 'HERO').bannerHero).toBe('HERO')
  })

  it('maps ui_ color keys + cascades to status', async () => {
    const { fromSkin } = await importThemeWithCleanEnv()
    const { color } = fromSkin({ ui_ok: '#008000' }, {})

    expect(color.ok).toBe('#008000')
    expect(color.statusGood).toBe('#008000')
  })
})
