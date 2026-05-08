/** Platform-aware keybinding helpers.
 *
 * On macOS the "action" modifier is Cmd. Modern terminals that support kitty
 * keyboard protocol report Cmd as `key.super`; legacy terminals often surface it
 * as `key.meta`. Some macOS terminals also translate Cmd+Left/Right/Backspace
 * into readline-style Ctrl+A/Ctrl+E/Ctrl+U before the app sees them.
 * On other platforms the action modifier is Ctrl.
 * Ctrl+C stays the interrupt key on macOS. On non-mac terminals it can also
 * copy an active TUI selection, matching common terminal selection behavior.
 */

export const isMac = process.platform === 'darwin'

/** True when the platform action-modifier is pressed (Cmd on macOS, Ctrl elsewhere). */
export const isActionMod = (key: { ctrl: boolean; meta: boolean; super?: boolean }): boolean =>
  isMac ? key.meta || key.super === true : key.ctrl

/**
 * Accept raw Ctrl+<letter> as an action shortcut on macOS, where `isActionMod`
 * otherwise means Cmd. Two motivations:
 *   - Some macOS terminals rewrite Cmd navigation/deletion into readline control
 *     keys (Cmd+Left → Ctrl+A, Cmd+Right → Ctrl+E, Cmd+Backspace → Ctrl+U).
 *   - Ctrl+K (kill-to-end) and Ctrl+W (delete-word-back) are standard readline
 *     bindings that users expect to work regardless of platform, even though
 *     no terminal rewrites Cmd into them.
 */
export const isMacActionFallback = (
  key: { ctrl: boolean; meta: boolean; super?: boolean },
  ch: string,
  target: 'a' | 'e' | 'u' | 'k' | 'w'
): boolean => isMac && key.ctrl && !key.meta && key.super !== true && ch.toLowerCase() === target

/** Match action-modifier + a single character (case-insensitive). */
export const isAction = (key: { ctrl: boolean; meta: boolean; super?: boolean }, ch: string, target: string): boolean =>
  isActionMod(key) && ch.toLowerCase() === target

export const isRemoteShell = (env: NodeJS.ProcessEnv = process.env): boolean =>
  Boolean(env.SSH_CONNECTION || env.SSH_CLIENT || env.SSH_TTY)

export const isCopyShortcut = (
  key: { ctrl: boolean; meta: boolean; super?: boolean },
  ch: string,
  env: NodeJS.ProcessEnv = process.env
): boolean =>
  ch.toLowerCase() === 'c' &&
  (isAction(key, ch, 'c') ||
    (isRemoteShell(env) && (key.meta || key.super === true)) ||
    // VS Code/Cursor/Windsurf terminal setup forwards Cmd+C as a CSI-u
    // sequence with the super bit plus a benign ctrl bit. Accept that shape
    // even though raw Ctrl+C should remain interrupt on local macOS.
    (isMac && key.ctrl && (key.meta || key.super === true)))

/**
 * Voice recording toggle key — configurable via ``voice.record_key`` in
 * ``config.yaml`` (default ``ctrl+b``).
 *
 * Documented in tips.py, the Python CLI prompt_toolkit handler, and the
 * config.yaml default. The TUI honours the same config knob (#18994);
 * when ``voice.record_key`` is e.g. ``ctrl+o`` the TUI binds Ctrl+O.
 *
 * Only the documented default (``ctrl+b``) additionally accepts the
 * macOS action modifier (Cmd+B) — custom bindings like ``ctrl+o``
 * require the literal Ctrl bit so Cmd+O can't steal the shortcut.
 */
export type VoiceRecordKeyMod = 'alt' | 'ctrl' | 'super'

/** Named (multi-character) keys we support, matching the CLI's
 * prompt_toolkit binding shape (``c-space``, ``c-enter``, etc.) so a
 * config value like ``ctrl+space`` binds in both runtimes. */
export type VoiceRecordKeyNamed = 'backspace' | 'delete' | 'enter' | 'escape' | 'space' | 'tab'

export interface ParsedVoiceRecordKey {
  /** Single character (``'b'``, ``'o'``) when ``named`` is undefined,
   * otherwise the named-key token (``'space'``, ``'enter'``…). Kept as
   * one field for back-compat with the v1 ``{ ch, mod, raw }`` shape. */
  ch: string
  mod: VoiceRecordKeyMod
  named?: VoiceRecordKeyNamed
  raw: string
}

export const DEFAULT_VOICE_RECORD_KEY: ParsedVoiceRecordKey = {
  ch: 'b',
  mod: 'ctrl',
  raw: 'ctrl+b'
}

/** Modifier aliases.
 *
 * ``meta`` / ``cmd`` / ``command`` are intentionally absent.
 * hermes-ink sets ``key.meta`` for plain Alt/Option on every platform
 * AND for Cmd on some legacy macOS terminals (Terminal.app without
 * kitty-protocol passthrough). Accepting any of those as a literal
 * modifier would produce a display/binding mismatch — a config like
 * ``cmd+b`` would render as ``Cmd+B`` but silently fire on Alt+B, or
 * never fire at all on legacy terminals even though the UI advertises
 * it (Copilot round-6 review on #19835). Users on modern kitty-style
 * terminals (iTerm2 CSI-u, Ghostty, Kitty, WezTerm, Alacritty) spell
 * the platform action modifier ``super`` / ``win``, which match the
 * unambiguous ``key.super`` bit. macOS users on Terminal.app stick
 * with the documented ``ctrl+b``.
 *
 * Cross-runtime parity: the ``ctrl`` / ``control`` / ``alt`` / ``option`` /
 * ``opt`` spellings are normalized identically in the classic CLI
 * (``hermes_cli/voice.py::normalize_voice_record_key_for_prompt_toolkit``)
 * so one ``voice.record_key`` value binds the same shortcut in both
 * runtimes (Copilot round-9 review on #19835). The ``super`` /
 * ``win`` / ``windows`` spellings are TUI-only — prompt_toolkit has no
 * super modifier, so the CLI falls back to the documented default and
 * logs a warning at startup (Copilot round-11 review on #19835). */
const _MOD_ALIASES: Record<string, VoiceRecordKeyMod> = {
  alt: 'alt',
  control: 'ctrl',
  ctrl: 'ctrl',
  option: 'alt',
  opt: 'alt',
  super: 'super',
  win: 'super',
  windows: 'super'
}

/** Map config-string named tokens to the canonical name used at match time.
 *
 * Aliases mirror what prompt_toolkit accepts (``return`` ↔ ``enter``,
 * ``esc`` ↔ ``escape``) so a config that round-trips through the CLI also
 * binds in the TUI. */
const _NAMED_KEY_ALIASES: Record<string, VoiceRecordKeyNamed> = {
  backspace: 'backspace',
  bs: 'backspace',
  del: 'delete',
  delete: 'delete',
  enter: 'enter',
  esc: 'escape',
  escape: 'escape',
  ret: 'enter',
  return: 'enter',
  space: 'space',
  spc: 'space',
  tab: 'tab'
}

/** ``useInputHandlers()`` intercepts these unconditionally before the
 * voice check runs, so a binding like ``ctrl+c`` (interrupt),
 * ``ctrl+d`` (quit), or ``ctrl+l`` (clear screen) would be advertised
 * in /voice status but never fire push-to-talk. Reject at parse time
 * so the user gets the documented Ctrl+B instead of a dead shortcut
 * (Copilot round-4 review on #19835).
 *
 * ``ctrl+x`` is intentionally NOT here — it's only claimed during
 * queue-edit (``queueEditIdx !== null``), so the voice binding works
 * for most of the session and matches CLI parity for ``ctrl+<letter>``
 * bindings (Copilot round-8 review on #19835). */
const _RESERVED_CTRL_CHARS = new Set(['c', 'd', 'l'])

/** On macOS the action-modifier intercepts these editor chords via
 * ``isCopyShortcut`` / ``isAction`` in ``useInputHandlers()``:
 *  - super+c → copy
 *  - super+d → exit
 *  - super+l → clear screen
 *  - super+v → paste (also claimed at the TextInput layer)
 * On Linux/Windows those globals key off Ctrl instead of Super, so
 * super+<letter> bindings don't collide. Gate the rejection to darwin
 * at parse time so kitty/CSI-u ``super+<key>`` configs still work for
 * non-mac users (Copilot round-8 review on #19835). */
const _RESERVED_SUPER_CHARS = new Set(['c', 'd', 'l', 'v'])

/** On macOS ``isActionMod`` accepts ``key.meta`` as the action
 * modifier — but hermes-ink reports Alt as ``key.meta`` on many
 * terminals. So on darwin a configured ``alt+c`` / ``alt+d`` / ``alt+l``
 * gets swallowed by ``isCopyShortcut`` / ``isAction`` before the voice
 * check runs. Block at parse time so /voice status doesn't advertise
 * a shortcut that actually copies / quits / clears (Copilot round-12
 * review on #19835). */
const _RESERVED_ALT_CHARS_MAC = new Set(['c', 'd', 'l'])

interface RuntimeKeyEvent {
  alt?: boolean
  backspace?: boolean
  ctrl: boolean
  delete?: boolean
  escape?: boolean
  meta: boolean
  return?: boolean
  shift?: boolean
  super?: boolean
  tab?: boolean
}

/** Match an ink ``key`` event against a parsed named key. The ink runtime
 * sets one boolean per named key; ``space`` is a printable char so it
 * arrives as ``ch === ' '`` rather than a dedicated ``key.space`` flag. */
const _matchesNamedKey = (
  named: VoiceRecordKeyNamed,
  key: RuntimeKeyEvent,
  ch: string
): boolean => {
  switch (named) {
    case 'backspace':
      return key.backspace === true
    case 'delete':
      return key.delete === true
    case 'enter':
      return key.return === true
    case 'escape':
      return key.escape === true
    case 'space':
      return ch === ' '
    case 'tab':
      return key.tab === true
  }
}

/**
 * Parse a config-string voice record key like ``ctrl+b`` / ``alt+r`` /
 * ``ctrl+space`` into ``{mod, ch, named?}``. Accepts single characters
 * AND the named tokens declared in ``_NAMED_KEY_ALIASES`` (``space``,
 * ``enter``/``return``, ``tab``, ``escape``/``esc``, ``backspace``,
 * ``delete``) — matching the keys prompt_toolkit accepts on the CLI
 * side via the ``c-<name>`` rewrite in ``cli.py``.
 *
 * Accepts ``unknown`` because the source is raw YAML via
 * ``config.get full`` — a hand-edited ``voice.record_key: 1`` or
 * ``voice.record_key: true`` would otherwise crash ``.trim()`` on a
 * non-string scalar (Copilot round-3 review on #19835). Non-string /
 * empty / unrecognised values fall back to the documented Ctrl+B
 * default so a typo never silently disables the shortcut.
 */
export const parseVoiceRecordKey = (raw: unknown): ParsedVoiceRecordKey => {
  if (typeof raw !== 'string') {
    return DEFAULT_VOICE_RECORD_KEY
  }

  const lower = raw.trim().toLowerCase()

  if (!lower) {
    return DEFAULT_VOICE_RECORD_KEY
  }

  const parts = lower.split('+').map(p => p.trim()).filter(Boolean)

  if (!parts.length) {
    return DEFAULT_VOICE_RECORD_KEY
  }

  const last = parts[parts.length - 1]
  const modCandidates = parts.slice(0, -1)

  // Reject multi-modifier chords (``ctrl+alt+r``, ``cmd+ctrl+b``) rather
  // than silently dropping the extra modifier — the previous
  // single-token validator made a typo bind a different shortcut than
  // the user configured (Copilot round-3 review on #19835). The classic
  // CLI only supports single-modifier bindings via prompt_toolkit's
  // ``c-x`` / ``a-x`` rewrite in ``cli.py``, so this matches CLI parity.
  if (modCandidates.length > 1) {
    return DEFAULT_VOICE_RECORD_KEY
  }

  // Require an explicit modifier. A bare ``o`` / ``space`` / ``escape``
  // has no sensible mapping: the CLI's prompt_toolkit binds the raw
  // key (no rewrite) so bare-char configs would silently diverge
  // between the two runtimes (Copilot round-4 review on #19835).
  // Fall back to the documented default.
  if (modCandidates.length === 0) {
    return DEFAULT_VOICE_RECORD_KEY
  }

  const norm = _MOD_ALIASES[modCandidates[0]]

  // Unknown modifier token (e.g. bare ``meta+b`` which is ambiguous on
  // the wire) falls back to the documented default rather than
  // silently coercing to Ctrl and producing a misleading bind.
  if (!norm) {
    return DEFAULT_VOICE_RECORD_KEY
  }

  const mod = norm

  // Block bindings the TUI input handler intercepts before the voice
  // check — ``ctrl+c`` / ``ctrl+d`` / ``ctrl+l`` would never actually
  // fire push-to-talk, so advertising them in /voice status is a lie.
  if (mod === 'ctrl' && last.length === 1 && _RESERVED_CTRL_CHARS.has(last)) {
    return DEFAULT_VOICE_RECORD_KEY
  }

  // Same for ``super+c`` / ``super+d`` / ``super+l`` / ``super+v`` on
  // macOS only — those are copy / exit / clear / paste and get claimed
  // by ``isCopyShortcut`` / ``isAction`` / the TextInput paste layer
  // before voice has a chance to toggle. On Linux/Windows the TUI
  // globals key off Ctrl (not Super), so kitty/CSI-u ``super+<letter>``
  // bindings stay usable for non-mac users.
  if (isMac && mod === 'super' && last.length === 1 && _RESERVED_SUPER_CHARS.has(last)) {
    return DEFAULT_VOICE_RECORD_KEY
  }

  // On macOS hermes-ink reports Alt as ``key.meta``, which ``isActionMod``
  // accepts as the mac action modifier. So ``alt+c`` / ``alt+d`` / ``alt+l``
  // collide with copy / exit / clear in ``useInputHandlers()`` before the
  // voice check. Reject at parse time on darwin only — non-mac ``alt+<letter>``
  // bindings are still usable (Copilot round-12 review on #19835).
  if (isMac && mod === 'alt' && last.length === 1 && _RESERVED_ALT_CHARS_MAC.has(last)) {
    return DEFAULT_VOICE_RECORD_KEY
  }

  if (last.length === 1) {
    return { ch: last, mod, raw: lower }
  }

  const named = _NAMED_KEY_ALIASES[last]

  if (named) {
    return { ch: named, mod, named, raw: lower }
  }

  // Unknown multi-character token (e.g. typo'd ``ctrl+spcae``) — fall back
  // to the doc default rather than silently disabling the binding.
  return DEFAULT_VOICE_RECORD_KEY
}

/** Render a parsed key back as ``Ctrl+B`` / ``Ctrl+Space`` for status text.
 *
 * Platform-aware for the ``super`` modifier: renders ``Cmd`` on macOS and
 * ``Super`` elsewhere. Previously rendered ``Cmd`` universally, which told
 * Linux/Windows users the wrong modifier to press (Copilot review, round
 * 2 on #19835). */
export const formatVoiceRecordKey = (parsed: ParsedVoiceRecordKey): string => {
  const modLabel =
    parsed.mod === 'super' ? (isMac ? 'Cmd' : 'Super') : parsed.mod[0].toUpperCase() + parsed.mod.slice(1)
  // Named tokens render in title case (Ctrl+Space, Ctrl+Enter); single
  // chars render upper-case to match the existing Ctrl+B convention.
  const keyLabel = parsed.named
    ? parsed.named[0].toUpperCase() + parsed.named.slice(1)
    : parsed.ch.toUpperCase()

  return `${modLabel}+${keyLabel}`
}

/** Whether the parsed binding is the documented default (ctrl+b).
 *
 * Compare on the parsed spec rather than ``raw`` so semantically-equal
 * aliases (``control+b``, ``ctrl + b``) still get the macOS Cmd+B
 * muscle-memory fallback (Copilot review, round 2 on #19835). */
const _isDefaultVoiceKey = (parsed: ParsedVoiceRecordKey): boolean =>
  parsed.mod === DEFAULT_VOICE_RECORD_KEY.mod &&
  parsed.ch === DEFAULT_VOICE_RECORD_KEY.ch &&
  parsed.named === DEFAULT_VOICE_RECORD_KEY.named

export const isVoiceToggleKey = (
  key: RuntimeKeyEvent,
  ch: string,
  configured: ParsedVoiceRecordKey = DEFAULT_VOICE_RECORD_KEY
): boolean => {
  // Match the configured key first (single-char compare or named-key
  // event-property check). Bail out before evaluating modifier shape
  // so the wrong key never reaches the modifier guard.
  if (configured.named) {
    if (!_matchesNamedKey(configured.named, key, ch)) {
      return false
    }
  } else if (ch.toLowerCase() !== configured.ch) {
    return false
  }

  // The parser rejects multi-modifier configs (``ctrl+shift+b`` etc.),
  // so at match time Shift must always be clear — otherwise
  // ``ctrl+tab`` would also fire on Ctrl+Shift+Tab and ``alt+enter``
  // on Alt+Shift+Enter, triggering a different chord than configured
  // (Copilot round-5 review on #19835).
  if (key.shift === true) {
    return false
  }

  switch (configured.mod) {
    case 'alt':
      // Most terminals surface Alt as either ``alt`` or ``meta``; accept
      // both so the binding works across xterm-style and kitty-style
      // protocols. Guard against ctrl/super bits so a chord like
      // Ctrl+Alt+<key> or Cmd+Alt+<key> doesn't spuriously fire the
      // alt binding.
      //
      // Bare Escape on hermes-ink can arrive as ``key.meta=true`` on some
      // terminals, so a configured ``alt+escape`` must not match that shape;
      // require an explicit alt bit for escape chords (Copilot round-7
      // follow-up on #19835).
      return (key.alt === true || (key.meta && key.escape !== true)) && !key.ctrl && key.super !== true
    case 'ctrl':
      // Require the Ctrl bit AND a clear Alt/Super so a chord like
      // Ctrl+Alt+<key> / Ctrl+Cmd+<key> doesn't spuriously match
      // ``ctrl+<key>`` (Copilot round-6 review on #19835).
      //
      // The documented default (``ctrl+b``) additionally accepts the
      // explicit ``key.super`` bit on macOS for Cmd+B muscle memory —
      // but ONLY ``key.super`` (kitty-style), never ``key.meta``, since
      // ``key.meta`` is hermes-ink's Alt signal and accepting it would
      // fire the binding on Alt+B.
      if (key.ctrl) {
        return !key.alt && !key.meta && key.super !== true
      }

      return _isDefaultVoiceKey(configured) && isMac && key.super === true && !key.alt && !key.meta
    case 'super':
      // Require the explicit ``key.super`` bit (kitty-style protocol)
      // AND clear Ctrl/Alt/Meta so Ctrl+Cmd+X or Alt+Cmd+X don't
      // spuriously fire the super binding (Copilot round-6 review on
      // #19835). Legacy-terminal users whose Cmd arrives as
      // ``key.meta`` need a kitty-protocol terminal — see the
      // _MOD_ALIASES doc-comment for the rationale.
      return key.super === true && !key.ctrl && !key.alt && !key.meta
  }
}
