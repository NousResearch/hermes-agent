import { describe, expect, it } from 'vitest'

import {
  DEFAULT_VOICE_RECORD_KEY,
  formatVoiceRecordKey,
  isActionMod,
  isCopyShortcut,
  isMac,
  isMacActionFallback,
  isVoiceToggleKey,
  parseVoiceRecordKey
} from '../lib/platform.js'

// These cover the non-macOS (host-native on the Linux CI lane) arms. The
// macOS arms key off a module-level `isMac` and would need the interpreter to
// believe it is on darwin — see AGENTS.md "Don't fake the host OS".
const describeHost = describe.skipIf(isMac)

describeHost('platform action modifier', () => {
  it('still uses Ctrl as the action modifier on non-macOS', () => {
    expect(isActionMod({ ctrl: true, meta: false, super: false })).toBe(true)
    expect(isActionMod({ ctrl: false, meta: false, super: true })).toBe(false)
  })
})

describeHost('isCopyShortcut', () => {
  it('keeps Ctrl+C as the local non-macOS copy chord', () => {
    expect(isCopyShortcut({ ctrl: true, meta: false, super: false }, 'c', {})).toBe(true)
  })

  it('accepts client Cmd+C over SSH even when running on Linux', () => {
    const env = { SSH_CONNECTION: '1 2 3 4' } as NodeJS.ProcessEnv

    expect(isCopyShortcut({ ctrl: false, meta: false, super: true }, 'c', env)).toBe(true)
    expect(isCopyShortcut({ ctrl: false, meta: true, super: false }, 'c', env)).toBe(true)
  })

  it('does not treat local Linux Alt+C as copy', () => {
    expect(isCopyShortcut({ ctrl: false, meta: true, super: false }, 'c', {})).toBe(false)
  })
})

describeHost('isVoiceToggleKey', () => {
  it('matches Ctrl+B on non-macOS platforms', () => {
    expect(isVoiceToggleKey({ ctrl: true, meta: false, super: false }, 'b')).toBe(true)
    expect(isVoiceToggleKey({ ctrl: true, meta: false, super: false }, 'B')).toBe(true)
  })

  it('does not match unmodified b or other Ctrl combos', () => {
    expect(isVoiceToggleKey({ ctrl: false, meta: false, super: false }, 'b')).toBe(false)
    expect(isVoiceToggleKey({ ctrl: true, meta: false, super: false }, 'a')).toBe(false)
    expect(isVoiceToggleKey({ ctrl: true, meta: false, super: false }, 'c')).toBe(false)
  })
})

describeHost('parseVoiceRecordKey (#18994)', () => {
  it('falls back to Ctrl+B for empty input', () => {
    expect(parseVoiceRecordKey('')).toEqual(DEFAULT_VOICE_RECORD_KEY)
  })

  it('parses ctrl+<letter> bindings', () => {
    expect(parseVoiceRecordKey('ctrl+o')).toEqual({ ch: 'o', mod: 'ctrl', raw: 'ctrl+o' })
    expect(parseVoiceRecordKey('Ctrl+R')).toEqual({ ch: 'r', mod: 'ctrl', raw: 'ctrl+r' })
  })

  it('parses alt/super aliases', () => {
    expect(parseVoiceRecordKey('alt+b').mod).toBe('alt')
    expect(parseVoiceRecordKey('option+b').mod).toBe('alt')
    expect(parseVoiceRecordKey('super+b').mod).toBe('super')
    expect(parseVoiceRecordKey('win+b').mod).toBe('super')
  })

  it('treats ambiguous mac modifiers (meta / cmd / command) as unrecognised', () => {
    // ``meta`` / ``cmd`` / ``command`` are ambiguous on the wire:
    // hermes-ink sets ``key.meta`` for plain Alt on every platform AND
    // for Cmd on legacy macOS terminals. Accepting any of them would
    // produce a display/binding mismatch (Copilot round-6 review on
    // #19835). Users on modern kitty-style terminals spell the
    // platform action modifier ``super`` / ``win``.
    expect(parseVoiceRecordKey('meta+b')).toEqual(DEFAULT_VOICE_RECORD_KEY)
    expect(parseVoiceRecordKey('cmd+b')).toEqual(DEFAULT_VOICE_RECORD_KEY)
    expect(parseVoiceRecordKey('command+b')).toEqual(DEFAULT_VOICE_RECORD_KEY)
  })

  it('parses named keys (space, enter, tab, escape, backspace, delete)', () => {
    // Every named token from the CLI's prompt_toolkit ``c-<name>`` set is
    // accepted with both the canonical name and its common alias.
    expect(parseVoiceRecordKey('ctrl+space')).toEqual({
      ch: 'space',
      mod: 'ctrl',
      named: 'space',
      raw: 'ctrl+space'
    })
    expect(parseVoiceRecordKey('alt+enter').named).toBe('enter')
    expect(parseVoiceRecordKey('alt+return').named).toBe('enter') // ``return`` ↔ ``enter``
    expect(parseVoiceRecordKey('ctrl+tab').named).toBe('tab')
    expect(parseVoiceRecordKey('ctrl+escape').named).toBe('escape')
    expect(parseVoiceRecordKey('ctrl+esc').named).toBe('escape') // ``esc`` alias
    expect(parseVoiceRecordKey('ctrl+backspace').named).toBe('backspace')
    expect(parseVoiceRecordKey('ctrl+delete').named).toBe('delete')
    expect(parseVoiceRecordKey('ctrl+del').named).toBe('delete') // ``del`` alias
  })

  it('falls back to Ctrl+B for unrecognised multi-character tokens', () => {
    // Typos / unsupported names (``ctrl+spcae``, ``ctrl+f5``, …) fall back
    // to the documented Ctrl+B default rather than silently disabling the
    // binding.
    expect(parseVoiceRecordKey('ctrl+spcae')).toEqual(DEFAULT_VOICE_RECORD_KEY)
    expect(parseVoiceRecordKey('ctrl+f5')).toEqual(DEFAULT_VOICE_RECORD_KEY)
  })

  // Round-3 Copilot review regressions on #19835.
  it('does not throw on non-string YAML scalars — falls back instead', () => {
    // ``config.get full`` surfaces raw YAML values; ``voice.record_key: 1``
    // or ``voice.record_key: true`` would otherwise crash ``.trim()``.
    expect(parseVoiceRecordKey(1 as unknown as string)).toEqual(DEFAULT_VOICE_RECORD_KEY)
    expect(parseVoiceRecordKey(true as unknown as string)).toEqual(DEFAULT_VOICE_RECORD_KEY)
    expect(parseVoiceRecordKey(null as unknown as string)).toEqual(DEFAULT_VOICE_RECORD_KEY)
    expect(parseVoiceRecordKey(undefined as unknown as string)).toEqual(DEFAULT_VOICE_RECORD_KEY)
    expect(parseVoiceRecordKey({} as unknown as string)).toEqual(DEFAULT_VOICE_RECORD_KEY)
  })

  it('rejects multi-modifier chords rather than silently dropping extras', () => {
    // Previously ``ctrl+alt+r`` parsed as ``ctrl+r`` and ``cmd+ctrl+b`` as
    // ``super+b`` — a typo silently bound a different shortcut. Now a
    // multi-modifier spelling falls back to the documented default.
    expect(parseVoiceRecordKey('ctrl+alt+r')).toEqual(DEFAULT_VOICE_RECORD_KEY)
    expect(parseVoiceRecordKey('cmd+ctrl+b')).toEqual(DEFAULT_VOICE_RECORD_KEY)
    expect(parseVoiceRecordKey('alt+ctrl+space')).toEqual(DEFAULT_VOICE_RECORD_KEY)
  })

  // Round-4 Copilot review regressions on #19835.
  it('rejects bare-char configs without an explicit modifier', () => {
    // The classic CLI's prompt_toolkit binds raw-char configs to the key
    // itself (``c-o`` requires an explicit modifier); rewriting ``o``
    // → ``ctrl+o`` would silently diverge the two runtimes. Refuse.
    expect(parseVoiceRecordKey('o')).toEqual(DEFAULT_VOICE_RECORD_KEY)
    expect(parseVoiceRecordKey('b')).toEqual(DEFAULT_VOICE_RECORD_KEY)
    expect(parseVoiceRecordKey('space')).toEqual(DEFAULT_VOICE_RECORD_KEY)
    expect(parseVoiceRecordKey('escape')).toEqual(DEFAULT_VOICE_RECORD_KEY)
  })

  it('rejects ctrl+c / ctrl+d / ctrl+l — reserved by the TUI input handler', () => {
    // ``useInputHandlers()`` intercepts these before the voice check,
    // so a binding like ``ctrl+c`` would be advertised but never fire.
    // Fall back to the documented default instead of lying to the user.
    expect(parseVoiceRecordKey('ctrl+c')).toEqual(DEFAULT_VOICE_RECORD_KEY)
    expect(parseVoiceRecordKey('ctrl+d')).toEqual(DEFAULT_VOICE_RECORD_KEY)
    expect(parseVoiceRecordKey('ctrl+l')).toEqual(DEFAULT_VOICE_RECORD_KEY)
    // Alt-modifier versions of those letters are NOT intercepted, so
    // they remain usable.
    expect(parseVoiceRecordKey('alt+c').mod).toBe('alt')
    // ``ctrl+x`` is intentionally allowed — only intercepted during
    // queue-edit (``queueEditIdx !== null``), so the voice binding
    // works for most of the session (Copilot round-8 review).
    expect(parseVoiceRecordKey('ctrl+x').mod).toBe('ctrl')
    expect(parseVoiceRecordKey('ctrl+x').ch).toBe('x')
  })

  it('allows super+{c,d,l,v} on Linux/Windows — those globals key off Ctrl, not Super', () => {
    // Kitty/CSI-u users on non-mac report Cmd/Super as ``key.super``,
    // but the TUI's global shortcuts (copy/exit/clear/paste) key off
    // Ctrl there, so ``super+<letter>`` doesn't collide. Reject would
    // silently coerce valid configs to Ctrl+B (Copilot round-8 review).
    expect(parseVoiceRecordKey('super+c').mod).toBe('super')
    expect(parseVoiceRecordKey('super+d').mod).toBe('super')
    expect(parseVoiceRecordKey('super+l').mod).toBe('super')
    expect(parseVoiceRecordKey('super+v').mod).toBe('super')
  })

  it('allows alt+{c,d,l} on Linux/Windows — non-mac isAction keys off Ctrl', () => {
    // On Linux/Windows ``isActionMod`` ignores key.meta, so alt+<letter>
    // doesn't collide with copy/exit/clear. Those configs stay usable.
    expect(parseVoiceRecordKey('alt+c').mod).toBe('alt')
    expect(parseVoiceRecordKey('alt+d').mod).toBe('alt')
    expect(parseVoiceRecordKey('alt+l').mod).toBe('alt')
  })

  it('ctrl+<key> rejects chords with extra alt / meta / super bits', () => {
    const ctrlO = parseVoiceRecordKey('ctrl+o')

    // ``ctrl+o`` must fire ONLY on literal Ctrl+O, not on
    // Ctrl+Alt+O / Ctrl+Cmd+O / Ctrl+Meta+O — otherwise the runtime
    // matches a different chord than the parser would let you
    // configure.
    expect(isVoiceToggleKey({ alt: true, ctrl: true, meta: false, super: false }, 'o', ctrlO)).toBe(false)
    expect(isVoiceToggleKey({ ctrl: true, meta: true, super: false }, 'o', ctrlO)).toBe(false)
    expect(isVoiceToggleKey({ ctrl: true, meta: false, super: true }, 'o', ctrlO)).toBe(false)
    // Sanity: plain Ctrl+O still fires.
    expect(isVoiceToggleKey({ ctrl: true, meta: false, super: false }, 'o', ctrlO)).toBe(true)
  })

  it('super+<key> rejects chords with extra ctrl / alt / meta bits', () => {
    const superB = parseVoiceRecordKey('super+b')

    expect(isVoiceToggleKey({ alt: true, ctrl: false, meta: false, super: true }, 'b', superB)).toBe(false)
    expect(isVoiceToggleKey({ ctrl: false, meta: true, super: true }, 'b', superB)).toBe(false)
    expect(isVoiceToggleKey({ ctrl: true, meta: false, super: true }, 'b', superB)).toBe(false)
    // Sanity: plain Super+B still fires.
    expect(isVoiceToggleKey({ ctrl: false, meta: false, super: true }, 'b', superB)).toBe(true)
  })

  it('rejects matches when Shift is held (different chord than configured)', () => {
    // Parser rejects multi-modifier configs like ``ctrl+shift+tab``,
    // so the runtime matcher must also reject Shift-held events —
    // otherwise ``ctrl+tab`` would fire on Ctrl+Shift+Tab.
    const ctrlTab = parseVoiceRecordKey('ctrl+tab')
    const altEnter = parseVoiceRecordKey('alt+enter')
    const ctrlO = parseVoiceRecordKey('ctrl+o')

    expect(isVoiceToggleKey({ ctrl: true, meta: false, shift: true, super: false, tab: true }, '', ctrlTab)).toBe(false)
    expect(
      isVoiceToggleKey({ alt: true, ctrl: false, meta: false, return: true, shift: true, super: false }, '', altEnter)
    ).toBe(false)
    expect(isVoiceToggleKey({ ctrl: true, meta: false, shift: true, super: false }, 'o', ctrlO)).toBe(false)

    // Sanity: same events without Shift still fire.
    expect(isVoiceToggleKey({ ctrl: true, meta: false, shift: false, super: false, tab: true }, '', ctrlTab)).toBe(true)
    expect(isVoiceToggleKey({ ctrl: true, meta: false, shift: false, super: false }, 'o', ctrlO)).toBe(true)
  })
})

describeHost('formatVoiceRecordKey (#18994)', () => {
  it('renders as the user expects in /voice status', () => {
    expect(formatVoiceRecordKey(parseVoiceRecordKey('ctrl+b'))).toBe('Ctrl+B')
    expect(formatVoiceRecordKey(parseVoiceRecordKey('ctrl+o'))).toBe('Ctrl+O')
    expect(formatVoiceRecordKey(parseVoiceRecordKey('alt+r'))).toBe('Alt+R')
    // ``super``/``win`` render as ``Super`` on non-mac so the hint
    // doesn't tell Linux/Windows users to press a Cmd key they don't
    // have.
    expect(formatVoiceRecordKey(parseVoiceRecordKey('super+b'))).toBe('Super+B')
  })

  it('renders named keys in title case (Ctrl+Space, Ctrl+Enter)', () => {
    expect(formatVoiceRecordKey(parseVoiceRecordKey('ctrl+space'))).toBe('Ctrl+Space')
    expect(formatVoiceRecordKey(parseVoiceRecordKey('alt+enter'))).toBe('Alt+Enter')
    expect(formatVoiceRecordKey(parseVoiceRecordKey('ctrl+esc'))).toBe('Ctrl+Escape')
    expect(formatVoiceRecordKey(parseVoiceRecordKey('super+space'))).toBe('Super+Space')
  })
})

describeHost('isVoiceToggleKey honours configured record key (#18994)', () => {
  it('binds the configured letter, not hardcoded b', () => {
    const ctrlO = parseVoiceRecordKey('ctrl+o')

    expect(isVoiceToggleKey({ ctrl: true, meta: false, super: false }, 'o', ctrlO)).toBe(true)
    // The old hardcoded 'b' must NOT match when the user configured 'o'.
    expect(isVoiceToggleKey({ ctrl: true, meta: false, super: false }, 'b', ctrlO)).toBe(false)
  })

  it('alt+<letter> binding matches alt OR meta (terminal-protocol parity)', () => {
    const altR = parseVoiceRecordKey('alt+r')

    expect(isVoiceToggleKey({ alt: true, ctrl: false, meta: false, super: false }, 'r', altR)).toBe(true)
    expect(isVoiceToggleKey({ ctrl: false, meta: true, super: false }, 'r', altR)).toBe(true)
    expect(isVoiceToggleKey({ ctrl: false, meta: false, super: false }, 'r', altR)).toBe(false)
  })

  it('binds named keys via ink event flags (space → ch === " ", enter → key.return, …)', () => {
    const ctrlSpace = parseVoiceRecordKey('ctrl+space')
    expect(isVoiceToggleKey({ ctrl: true, meta: false, super: false }, ' ', ctrlSpace)).toBe(true)
    // Single-char ``b`` must NOT match a ``space``-configured binding.
    expect(isVoiceToggleKey({ ctrl: true, meta: false, super: false }, 'b', ctrlSpace)).toBe(false)
    // Space without the configured modifier must not fire either.
    expect(isVoiceToggleKey({ ctrl: false, meta: false, super: false }, ' ', ctrlSpace)).toBe(false)

    const ctrlEnter = parseVoiceRecordKey('ctrl+enter')
    expect(isVoiceToggleKey({ ctrl: true, meta: false, return: true, super: false }, '', ctrlEnter)).toBe(true)
    expect(isVoiceToggleKey({ ctrl: true, meta: false, return: false, super: false }, '', ctrlEnter)).toBe(false)

    const altTab = parseVoiceRecordKey('alt+tab')
    expect(isVoiceToggleKey({ alt: true, ctrl: false, meta: false, super: false, tab: true }, '', altTab)).toBe(true)
    expect(isVoiceToggleKey({ alt: false, ctrl: false, meta: false, super: false, tab: true }, '', altTab)).toBe(false)

    const ctrlEscape = parseVoiceRecordKey('ctrl+escape')
    expect(isVoiceToggleKey({ ctrl: true, escape: true, meta: false, super: false }, '', ctrlEscape)).toBe(true)
    expect(isVoiceToggleKey({ ctrl: true, escape: false, meta: false, super: false }, '', ctrlEscape)).toBe(false)

    const ctrlBackspace = parseVoiceRecordKey('ctrl+backspace')
    expect(isVoiceToggleKey({ backspace: true, ctrl: true, meta: false, super: false }, '', ctrlBackspace)).toBe(true)

    const ctrlDelete = parseVoiceRecordKey('ctrl+delete')
    expect(isVoiceToggleKey({ ctrl: true, delete: true, meta: false, super: false }, '', ctrlDelete)).toBe(true)
  })

  it('omitted configured key falls back to ctrl+b (back-compat)', () => {
    // No third arg → DEFAULT_VOICE_RECORD_KEY → Ctrl+B behaviour.
    expect(isVoiceToggleKey({ ctrl: true, meta: false, super: false }, 'b')).toBe(true)
    expect(isVoiceToggleKey({ ctrl: true, meta: false, super: false }, 'o')).toBe(false)
  })
})

describeHost('isMacActionFallback', () => {
  it('is a no-op on non-macOS (Linux routes Ctrl+K/W through isActionMod directly)', () => {
    expect(isMacActionFallback({ ctrl: true, meta: false, super: false }, 'k', 'k')).toBe(false)
    expect(isMacActionFallback({ ctrl: true, meta: false, super: false }, 'w', 'w')).toBe(false)
  })
})
