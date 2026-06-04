import { nonAlphanumericKeys, type ParsedKey } from '../parse-keypress.js'

import { Event } from './event.js'

const inputForSpecialSequence = (name: string): string =>
  name === 'space' ? ' ' : name === 'return' || name === 'escape' ? '' : name

// SGR mouse-report fragment that leaked into a nameless text/sequence token.
// In alt-screen Ink enables MOUSE_ANY (DEC 1003), so every pixel of motion
// emits a CSI mouse report (ESC[<btn;col;row M/m). When a heavy React commit
// blocks the event loop past App's 50ms flush watchdog, that CSI can be split
// across stdin chunks at ANY byte boundary. The tokenizer flush force-emits
// the buffered prefix and resets to ground, so BOTH halves can surface as
// unparseable tokens that parseKeypress can't classify (name=''):
//
//   - flushed prefix   — ESC[< / [< / < + partial params, no terminator yet
//                        (e.g. `ESC[<0;35;`, `[<0;`, `<0;35;46`)
//   - ESC-less tail    — 1-, 2-, or 3-field digit run ending in M/m
//                        (e.g. `46M`, `;46M`, `35;46M`, `;35;46M`, `0;35;46M`)
//
// One regex covers every split position. The leading-`;` and 1-/2-field tails
// are the cases the older `/^\[<\d+;\d+;\d+[Mm]/` guard missed, which is how
// `46M35;40M...` ends up typed into the prompt during long sessions.
//
// Safety: the `(?=…\d)` lookahead requires at least one digit, so a typed `<`,
// `[`, `;`, or `M` (none of which carry a coordinate digit) is never matched;
// the embedded `M`/`m` in `[\d;]+` means a run like `1;2;3M9;10M` (two stuck-
// together fragments / prose) can't satisfy the `$` anchor and is left intact.
// Combined with the caller's `!keypress.name` gate — real typing arrives one
// char per chunk with a name set — no genuine keystroke is swallowed.
// eslint-disable-next-line no-control-regex
const SGR_MOUSE_FRAGMENT_LEAK_RE = /^(?:\x1b)?(?=(?:\[<|<)?[\d;]*\d)(?:\[<|<)?[\d;]+[Mm]?$/

export type Key = {
  upArrow: boolean
  downArrow: boolean
  leftArrow: boolean
  rightArrow: boolean
  pageDown: boolean
  pageUp: boolean
  wheelUp: boolean
  wheelDown: boolean
  home: boolean
  end: boolean
  return: boolean
  escape: boolean
  ctrl: boolean
  shift: boolean
  fn: boolean
  tab: boolean
  backspace: boolean
  delete: boolean
  meta: boolean
  super: boolean
}

function parseKey(keypress: ParsedKey): [Key, string] {
  const key: Key = {
    upArrow: keypress.name === 'up',
    downArrow: keypress.name === 'down',
    leftArrow: keypress.name === 'left',
    rightArrow: keypress.name === 'right',
    pageDown: keypress.name === 'pagedown',
    pageUp: keypress.name === 'pageup',
    wheelUp: keypress.name === 'wheelup',
    wheelDown: keypress.name === 'wheeldown',
    home: keypress.name === 'home',
    end: keypress.name === 'end',
    return: keypress.name === 'return',
    escape: keypress.name === 'escape',
    fn: keypress.fn,
    ctrl: keypress.ctrl,
    shift: keypress.shift,
    tab: keypress.name === 'tab',
    backspace: keypress.name === 'backspace',
    delete: keypress.name === 'delete',
    // `parseKeypress` parses \u001B\u001B[A (meta + up arrow) as meta = false
    // but with option = true, so we need to take this into account here
    // to avoid breaking changes in Ink.
    // TODO(vadimdemedes): consider removing this in the next major version.
    meta: keypress.meta || keypress.name === 'escape' || keypress.option,
    // Super (Cmd on macOS / Win key) — only arrives via kitty keyboard
    // protocol CSI u sequences. Distinct from meta (Alt/Option) so
    // bindings like cmd+c can be expressed separately from opt+c.
    super: keypress.super
  }

  let input = keypress.ctrl ? keypress.name : keypress.sequence

  // Handle undefined input case
  if (input === undefined) {
    input = ''
  }

  // When ctrl is set, keypress.name for space is the literal word "space".
  // Convert to actual space character for consistency with the CSI u branch
  // (which maps 'space' → ' '). Without this, ctrl+space leaks the literal
  // word "space" into text input.
  if (keypress.ctrl && input === 'space') {
    input = ' '
  }

  // Suppress unrecognized escape sequences that were parsed as function keys
  // (matched by FN_KEY_RE) but have no name in the keyName map.
  // Examples: ESC[25~ (F13/Right Alt on Windows), ESC[26~ (F14), etc.
  // Without this, the ESC prefix is stripped below and the remainder (e.g.,
  // "[25~") leaks into the input as literal text.
  if (keypress.code && !keypress.name) {
    input = ''
  }

  // Suppress SGR mouse-report fragments left over from a flush-boundary split
  // (see SGR_MOUSE_FRAGMENT_LEAK_RE). Both the flushed CSI prefix and the
  // ESC-less remainder reach here as nameless tokens that parseKeypress can't
  // classify, so without this sink they leak into the prompt as `46M35;40M…`.
  // This is the same defensive sink as the F13 guard above; the underlying
  // tokenizer-flush race is upstream of this layer.
  if (!keypress.name && SGR_MOUSE_FRAGMENT_LEAK_RE.test(input)) {
    input = ''
  }

  // Strip meta if it's still remaining after `parseKeypress`
  // TODO(vadimdemedes): remove this in the next major version.
  if (input.startsWith('\u001B')) {
    input = input.slice(1)
  }

  // Track whether we've already processed this as a special sequence
  // that converted input to the key name (CSI u or application keypad mode).
  // For these, we don't want to clear input with nonAlphanumericKeys check.
  let processedAsSpecialSequence = false

  // Handle CSI u sequences (Kitty keyboard protocol): after stripping ESC,
  // we're left with "[codepoint;modifieru" (e.g., "[98;3u" for Alt+b).
  // Use the parsed key name instead for input handling. Require a digit
  // after [ — real CSI u is always [<digits>…u, and a bare startsWith('[')
  // false-matches X10 mouse at row 85 (Cy = 85+32 = 'u'), leaking the
  // literal text "mouse" into the prompt via processedAsSpecialSequence.
  if (/^\[\d/.test(input) && input.endsWith('u')) {
    if (!keypress.name) {
      // Unmapped Kitty functional key (Caps Lock 57358, F13–F35, KP nav,
      // bare modifiers, etc.) — keycodeToName() returned undefined. Swallow
      // so the raw "[57358u" doesn't leak into the prompt. See #38781.
      input = ''
    } else {
      input = inputForSpecialSequence(keypress.name)
    }

    processedAsSpecialSequence = true
  }

  // Handle xterm modifyOtherKeys sequences: after stripping ESC, we're left
  // with "[27;modifier;keycode~" (e.g., "[27;3;98~" for Alt+b). Same
  // extraction as CSI u — without this, printable-char keycodes (single-letter
  // names) skip the nonAlphanumericKeys clear and leak "[27;..." as input.
  if (input.startsWith('[27;') && input.endsWith('~')) {
    if (!keypress.name) {
      // Unmapped modifyOtherKeys keycode — swallow for consistency with
      // the CSI u handler above. Practically untriggerable today (xterm
      // modifyOtherKeys only sends ASCII keycodes, all mapped), but
      // guards against future terminal behavior.
      input = ''
    } else {
      input = inputForSpecialSequence(keypress.name)
    }

    processedAsSpecialSequence = true
  }

  // Handle application keypad mode sequences: after stripping ESC,
  // we're left with "O<letter>" (e.g., "Op" for numpad 0, "Oy" for numpad 9).
  // Use the parsed key name (the digit character) for input handling.
  if (input.startsWith('O') && input.length === 2 && keypress.name && keypress.name.length === 1) {
    input = keypress.name
    processedAsSpecialSequence = true
  }

  // Clear input for non-alphanumeric keys (arrows, function keys, etc.)
  // Skip this for CSI u and application keypad mode sequences since
  // those were already converted to their proper input characters.
  if (!processedAsSpecialSequence && keypress.name && nonAlphanumericKeys.includes(keypress.name)) {
    input = ''
  }

  // Set shift=true for uppercase letters (A-Z)
  // Must check it's actually a letter, not just any char unchanged by toUpperCase
  if (input.length === 1 && typeof input[0] === 'string' && input[0] >= 'A' && input[0] <= 'Z') {
    key.shift = true
  }

  return [key, input]
}

export class InputEvent extends Event {
  readonly keypress: ParsedKey
  readonly key: Key
  readonly input: string

  constructor(keypress: ParsedKey) {
    super()
    const [key, input] = parseKey(keypress)

    this.keypress = keypress
    this.key = key
    this.input = input
  }
}
