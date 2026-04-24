/**
 * OSC (Operating System Command) Types and Parser
 */

import { Buffer } from 'buffer'

import { env } from '../../utils/env.js'
import { execFileNoThrow } from '../../utils/execFileNoThrow.js'

import { BEL, ESC, ESC_TYPE, SEP } from './ansi.js'
import type { Action, Color, TabStatusAction } from './types.js'

export const OSC_PREFIX = ESC + String.fromCharCode(ESC_TYPE.OSC)
const ENV_ON_RE = /^(?:1|true|yes|on)$/i
const ENV_OFF_RE = /^(?:0|false|no|off)$/i

/** String Terminator (ESC \) - alternative to BEL for terminating OSC */
export const ST = ESC + '\\'

/** Generate an OSC sequence: ESC ] p1;p2;...;pN <terminator>
 * Uses ST terminator for Kitty (avoids beeps), BEL for others */
export function osc(...parts: (string | number)[]): string {
  const terminator = env.terminal === 'kitty' ? ST : BEL

  return `${OSC_PREFIX}${parts.join(SEP)}${terminator}`
}

/**
 * Wrap an escape sequence for terminal multiplexer passthrough.
 * tmux and GNU screen intercept escape sequences; DCS passthrough
 * tunnels them to the outer terminal unmodified.
 *
 * tmux 3.3+ gates this behind `allow-passthrough` (default off). When off,
 * tmux silently drops the whole DCS — no junk, no worse than unwrapped OSC.
 * Users who want passthrough set it in their .tmux.conf; we don't mutate it.
 *
 * Do NOT wrap BEL: raw \x07 triggers tmux's bell-action (window flag);
 * wrapped \x07 is opaque DCS payload and tmux never sees the bell.
 */
export function wrapForMultiplexer(sequence: string): string {
  if (process.env['TMUX']) {
    const escaped = sequence.replaceAll('\x1b', '\x1b\x1b')

    return `\x1bPtmux;${escaped}\x1b\\`
  }

  if (process.env['STY']) {
    return `\x1bP${sequence}\x1b\\`
  }

  return sequence
}

/**
 * Which path setClipboard() will take, based on env state. Synchronous so
 * callers can show an honest toast without awaiting the copy itself.
 *
 * - 'native': pbcopy (or equivalent) will run — high-confidence system
 *   clipboard write. tmux buffer may also be loaded as a bonus.
 * - 'tmux-buffer': tmux load-buffer will run, but no native tool — paste
 *   with prefix+] works. System clipboard depends on tmux's set-clipboard
 *   option + outer terminal OSC 52 support; can't know from here.
 * - 'osc52': only the raw OSC 52 sequence will be written to stdout.
 *   Best-effort; iTerm2 disables OSC 52 by default.
 *
 * pbcopy gating uses SSH_CONNECTION specifically, not SSH_TTY — tmux panes
 * inherit SSH_TTY forever even after local reattach, but SSH_CONNECTION is
 * in tmux's default update-environment set and gets cleared.
 */
export type ClipboardPath = 'native' | 'tmux-buffer' | 'osc52'

export function getClipboardPath(): ClipboardPath {
  const nativeAvailable = process.platform === 'darwin' && !process.env['SSH_CONNECTION']

  if (nativeAvailable) {
    return 'native'
  }

  if (process.env['TMUX']) {
    return 'tmux-buffer'
  }

  return 'osc52'
}

export function shouldEmitClipboardSequence(env: NodeJS.ProcessEnv = process.env): boolean {
  const override = (env.HERMES_TUI_CLIPBOARD_OSC52 ?? env.HERMES_TUI_COPY_OSC52 ?? '').trim()

  if (ENV_ON_RE.test(override)) {
    return true
  }

  if (ENV_OFF_RE.test(override)) {
    return false
  }

  return !!env['SSH_CONNECTION'] || (!env['TMUX'] && !env['STY'])
}

/**
 * Wrap a payload in tmux's DCS passthrough: ESC P tmux ; <payload> ESC \
 * tmux forwards the payload to the outer terminal, bypassing its own parser.
 * Inner ESCs must be doubled. Requires `set -g allow-passthrough on` in
 * ~/.tmux.conf; without it, tmux silently drops the whole DCS (no regression).
 */
function tmuxPassthrough(payload: string): string {
  return `${ESC}Ptmux;${payload.replaceAll(ESC, ESC + ESC)}${ST}`
}

/**
 * Load text into tmux's paste buffer via `tmux load-buffer`.
 * -w (tmux 3.2+) propagates to the outer terminal's clipboard via tmux's
 * own OSC 52 emission. -w is dropped for iTerm2: tmux's OSC 52 emission
 * crashes the iTerm2 session over SSH.
 *
 * Returns true if the buffer was loaded successfully.
 */
export async function tmuxLoadBuffer(text: string): Promise<boolean> {
  if (!process.env['TMUX']) {
    return false
  }

  const args = process.env['LC_TERMINAL'] === 'iTerm2' ? ['load-buffer', '-'] : ['load-buffer', '-w', '-']

  const { code } = await execFileNoThrow('tmux', args, {
    input: text,
    useCwd: false,
    timeout: 2000
  })

  return code === 0
}

/**
 * OSC 52 clipboard write: ESC ] 52 ; c ; <base64> BEL/ST
 * 'c' selects the clipboard (vs 'p' for primary selection on X11).
 *
 * When inside tmux ($TMUX set), `tmux load-buffer -w -` is the primary
 * path. tmux's buffer is always reachable — works over SSH, survives
 * detach/reattach, immune to stale env vars. The -w flag (tmux 3.2+) tells
 * tmux to also propagate to the outer terminal via its own OSC 52 path,
 * which tmux wraps correctly for the attached client. On older tmux, -w is
 * ignored and the buffer is still loaded. -w is dropped for iTerm2 (#22432)
 * because tmux's own OSC 52 emission (empty selection param: ESC]52;;b64)
 * crashes iTerm2 over SSH.
 *
 * After load-buffer succeeds, we ALSO return a DCS-passthrough-wrapped
 * OSC 52 for the caller to write to stdout. Our sequence uses explicit `c`
 * (not tmux's crashy empty-param variant), so it sidesteps the #22432 path.
 * With `allow-passthrough on` + an OSC-52-capable outer terminal, selection
 * reaches the system clipboard; with either off, tmux silently drops the
 * DCS and prefix+] still works. See Greg Smith's "free pony" in
 * https://anthropic.slack.com/archives/C07VBSHV7EV/p1773177228548119.
 *
 * If load-buffer fails entirely, fall through to raw OSC 52.
 *
 * Outside tmux, write raw OSC 52 to stdout (caller handles the write).
 *
 * Local (no SSH_CONNECTION): also shell out to a native clipboard utility.
 * OSC 52 and tmux -w both depend on terminal settings — iTerm2 disables
 * OSC 52 by default, VS Code shows a permission prompt on first use. Native
 * utilities (pbcopy/wl-copy/xclip/xsel/clip.exe) always work locally. Over
 * SSH these would write to the remote clipboard — OSC 52 is the right path there.
 *
 * Returns the sequence for the caller to write to stdout (raw OSC 52
 * outside tmux, DCS-wrapped inside).
 */
export async function setClipboard(text: string): Promise<string> {
  const b64 = Buffer.from(text, 'utf8').toString('base64')
  const raw = osc(OSC.CLIPBOARD, 'c', b64)
  const emitSequence = shouldEmitClipboardSequence(process.env)

  // Native safety net — fire FIRST, before the tmux await, so a quick
  // focus-switch after selecting doesn't race pbcopy. Previously this ran
  // AFTER awaiting tmux load-buffer, adding ~50-100ms of subprocess latency
  // before pbcopy even started — fast cmd+tab → paste would beat it
  // (https://anthropic.slack.com/archives/C07VBSHV7EV/p1773943921788829).
  // Gated on SSH_CONNECTION (not SSH_TTY) since tmux panes inherit SSH_TTY
  // forever but SSH_CONNECTION is in tmux's default update-environment and
  // clears on local attach. Fire-and-forget.
  if (!process.env['SSH_CONNECTION']) {
    copyNative(text)
  }

  const tmuxBufferLoaded = await tmuxLoadBuffer(text)

  // Inner OSC uses BEL directly (not osc()) — ST's ESC would need doubling
  // too, and BEL works everywhere for OSC 52.
  if (tmuxBufferLoaded) {
    return emitSequence ? tmuxPassthrough(`${ESC}]52;c;${b64}${BEL}`) : ''
  }

  return emitSequence ? raw : ''
}

// Linux clipboard tool: undefined = not yet probed, null = none available.
// Probe order: wl-copy (Wayland) → xclip (X11) → xsel (X11 fallback).
// Cached after first attempt so repeated mouse-ups skip the probe chain.
let linuxCopy: 'wl-copy' | 'xclip' | 'xsel' | null | undefined

/**
 * Shell out to a native clipboard utility as a safety net for OSC 52.
 * Only called when not in an SSH session (over SSH, these would write to
 * the remote machine's clipboard — OSC 52 is the right path there).
 * Fire-and-forget: failures are silent since OSC 52 may have succeeded.
 */
function copyNative(text: string): void {
  const opts = { input: text, useCwd: false, timeout: 2000 }

  switch (process.platform) {
    case 'darwin':
      void execFileNoThrow('pbcopy', [], opts)

      return
    case 'linux': {
      if (linuxCopy === null) {
        return
      }

      if (linuxCopy === 'wl-copy') {
        void execFileNoThrow('wl-copy', [], opts)

        return
      }

      if (linuxCopy === 'xclip') {
        void execFileNoThrow('xclip', ['-selection', 'clipboard'], opts)

        return
      }

      if (linuxCopy === 'xsel') {
        void execFileNoThrow('xsel', ['--clipboard', '--input'], opts)

        return
      }

      // First call: probe wl-copy (Wayland) then xclip/xsel (X11), cache winner.
      void execFileNoThrow('wl-copy', [], opts).then(r => {
        if (r.code === 0) {
          linuxCopy = 'wl-copy'

          return
        }

        void execFileNoThrow('xclip', ['-selection', 'clipboard'], opts).then(r2 => {
          if (r2.code === 0) {
            linuxCopy = 'xclip'

            return
          }

          void execFileNoThrow('xsel', ['--clipboard', '--input'], opts).then(r3 => {
            linuxCopy = r3.code === 0 ? 'xsel' : null
          })
        })
      })

      return
    }

    case 'win32':
      // clip.exe is always available on Windows. Unicode handling is
      // imperfect (system locale encoding) but good enough for a fallback.
      void execFileNoThrow('clip', [], opts)

      return
  }
}

/** @internal test-only */
export function _resetLinuxCopyCache(): void {
  linuxCopy = undefined
}

/**
 * OSC command numbers
 */
export const OSC = {
  SET_TITLE_AND_ICON: 0,
  SET_ICON: 1,
  SET_TITLE: 2,
  SET_COLOR: 4,
  SET_CWD: 7,
  HYPERLINK: 8,
  ITERM2: 9, // iTerm2 proprietary sequences
  SET_FG_COLOR: 10,
  SET_BG_COLOR: 11,
  SET_CURSOR_COLOR: 12,
  CLIPBOARD: 52,
  KITTY: 99, // Kitty notification protocol
  RESET_COLOR: 104,
  RESET_FG_COLOR: 110,
  RESET_BG_COLOR: 111,
  RESET_CURSOR_COLOR: 112,
  SEMANTIC_PROMPT: 133,
  GHOSTTY: 777, // Ghostty notification protocol
  TAB_STATUS: 21337 // Tab status extension
} as const

/**
 * Parse an OSC sequence into an action
 *
 * @param content - The sequence content (without ESC ] and terminator)
 */
export function parseOSC(content: string): Action | null {
  const semicolonIdx = content.indexOf(';')
  const command = semicolonIdx >= 0 ? content.slice(0, semicolonIdx) : content
  const data = semicolonIdx >= 0 ? content.slice(semicolonIdx + 1) : ''

  const commandNum = parseInt(command, 10)

  // Window/icon title
  if (commandNum === OSC.SET_TITLE_AND_ICON) {
    return { type: 'title', action: { type: 'both', title: data } }
  }

  if (commandNum === OSC.SET_ICON) {
    return { type: 'title', action: { type: 'iconName', name: data } }
  }

  if (commandNum === OSC.SET_TITLE) {
    return { type: 'title', action: { type: 'windowTitle', title: data } }
  }

  // Hyperlinks (OSC 8)
  if (commandNum === OSC.HYPERLINK) {
    const parts = data.split(';')
    const paramsStr = parts[0] ?? ''
    const url = parts.slice(1).join(';')

    if (url === '') {
      return { type: 'link', action: { type: 'end' } }
    }

    const params: Record<string, string> = {}

    if (paramsStr) {
      for (const pair of paramsStr.split(':')) {
        const eqIdx = pair.indexOf('=')

        if (eqIdx >= 0) {
          params[pair.slice(0, eqIdx)] = pair.slice(eqIdx + 1)
        }
      }
    }

    return {
      type: 'link',
      action: {
        type: 'start',
        url,
        params: Object.keys(params).length > 0 ? params : undefined
      }
    }
  }

  // Tab status (OSC 21337)
  if (commandNum === OSC.TAB_STATUS) {
    return { type: 'tabStatus', action: parseTabStatus(data) }
  }

  return { type: 'unknown', sequence: `\x1b]${content}` }
}

/**
 * Parse an XParseColor-style color spec into an RGB Color.
 * Accepts `#RRGGBB` and `rgb:R/G/B` (1–4 hex digits per component, scaled
 * to 8-bit). Returns null on parse failure.
 */
export function parseOscColor(spec: string): Color | null {
  const hex = spec.match(/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i)

  if (hex) {
    return {
      type: 'rgb',
      r: parseInt(hex[1]!, 16),
      g: parseInt(hex[2]!, 16),
      b: parseInt(hex[3]!, 16)
    }
  }

  const rgb = spec.match(/^rgb:([0-9a-f]{1,4})\/([0-9a-f]{1,4})\/([0-9a-f]{1,4})$/i)

  if (rgb) {
    // XParseColor: N hex digits → value / (16^N - 1), scale to 0-255
    const scale = (s: string) => Math.round((parseInt(s, 16) / (16 ** s.length - 1)) * 255)

    return {
      type: 'rgb',
      r: scale(rgb[1]!),
      g: scale(rgb[2]!),
      b: scale(rgb[3]!)
    }
  }

  return null
}

/**
 * Parse OSC 21337 payload: `key=value;key=value;...` with `\;` and `\\`
 * escapes inside values. Bare key or `key=` clears that field; unknown
 * keys are ignored.
 */
function parseTabStatus(data: string): TabStatusAction {
  const action: TabStatusAction = {}

  for (const [key, value] of splitTabStatusPairs(data)) {
    switch (key) {
      case 'indicator':
        action.indicator = value === '' ? null : parseOscColor(value)

        break

      case 'status':
        action.status = value === '' ? null : value

        break

      case 'status-color':
        action.statusColor = value === '' ? null : parseOscColor(value)

        break
    }
  }

  return action
}

/** Split `k=v;k=v` honoring `\;` and `\\` escapes. Yields [key, unescapedValue]. */
function* splitTabStatusPairs(data: string): Generator<[string, string]> {
  let key = ''
  let val = ''
  let inVal = false
  let esc = false

  for (const c of data) {
    if (esc) {
      if (inVal) {
        val += c
      } else {
        key += c
      }

      esc = false
    } else if (c === '\\') {
      esc = true
    } else if (c === ';') {
      yield [key, val]
      key = ''
      val = ''
      inVal = false
    } else if (c === '=' && !inVal) {
      inVal = true
    } else if (inVal) {
      val += c
    } else {
      key += c
    }
  }

  if (key || inVal) {
    yield [key, val]
  }
}

// Output generators

/** Start a hyperlink (OSC 8). Auto-assigns an id= param derived from the URL
 *  so terminals group wrapped lines of the same link together (the spec says
 *  cells with matching URI *and* nonempty id are joined; without an id each
 *  wrapped line is a separate link — inconsistent hover, partial tooltips).
 *  Empty url = close sequence (empty params per spec). */
export function link(url: string, params?: Record<string, string>): string {
  if (!url) {
    return LINK_END
  }

  const p = { id: osc8Id(url), ...params }

  const paramStr = Object.entries(p)
    .map(([k, v]) => `${k}=${v}`)
    .join(':')

  return osc(OSC.HYPERLINK, paramStr, url)
}

function osc8Id(url: string): string {
  let h = 0

  for (let i = 0; i < url.length; i++) {
    h = ((h << 5) - h + url.charCodeAt(i)) | 0
  }

  return (h >>> 0).toString(36)
}

/** End a hyperlink (OSC 8) */
export const LINK_END = osc(OSC.HYPERLINK, '', '')

// iTerm2 OSC 9 subcommands

/** iTerm2 OSC 9 subcommand numbers */
export const ITERM2 = {
  NOTIFY: 0,
  BADGE: 2,
  PROGRESS: 4
} as const

/** Progress operation codes (for use with ITERM2.PROGRESS) */
export const PROGRESS = {
  CLEAR: 0,
  SET: 1,
  ERROR: 2,
  INDETERMINATE: 3
} as const

/**
 * Clear iTerm2 progress bar sequence (OSC 9;4;0;BEL)
 * Uses BEL terminator since this is for cleanup (not runtime notification)
 * and we want to ensure it's always sent regardless of terminal type.
 */
export const CLEAR_ITERM2_PROGRESS = `${OSC_PREFIX}${OSC.ITERM2};${ITERM2.PROGRESS};${PROGRESS.CLEAR};${BEL}`

/**
 * Clear terminal title sequence (OSC 0 with empty string + BEL).
 * Uses BEL terminator for cleanup — safe on all terminals.
 */
export const CLEAR_TERMINAL_TITLE = `${OSC_PREFIX}${OSC.SET_TITLE_AND_ICON};${BEL}`

/** Clear all three OSC 21337 tab-status fields. Used on exit. */
export const CLEAR_TAB_STATUS = osc(OSC.TAB_STATUS, 'indicator=;status=;status-color=')

/**
 * Gate for emitting OSC 21337 (tab-status indicator). Ant-only while the
 * spec is unstable. Terminals that don't recognize it discard silently, so
 * emission is safe unconditionally — we don't gate on terminal detection
 * since support is expected across several terminals.
 *
 * Callers must wrap output with wrapForMultiplexer() so tmux/screen
 * DCS-passthrough carries the sequence to the outer terminal.
 */
export function supportsTabStatus(): boolean {
  return process.env.USER_TYPE === 'ant'
}

/**
 * Emit an OSC 21337 tab-status sequence. Omitted fields are left unchanged
 * by the receiving terminal; `null` sends an empty value to clear.
 * `;` and `\` in status text are escaped per the spec.
 */
export function tabStatus(fields: TabStatusAction): string {
  const parts: string[] = []

  const rgb = (c: Color) =>
    c.type === 'rgb' ? `#${[c.r, c.g, c.b].map(n => n.toString(16).padStart(2, '0')).join('')}` : ''

  if ('indicator' in fields) {
    parts.push(`indicator=${fields.indicator ? rgb(fields.indicator) : ''}`)
  }

  if ('status' in fields) {
    parts.push(`status=${fields.status?.replaceAll('\\', '\\\\').replaceAll(';', '\\;') ?? ''}`)
  }

  if ('statusColor' in fields) {
    parts.push(`status-color=${fields.statusColor ? rgb(fields.statusColor) : ''}`)
  }

  return osc(OSC.TAB_STATUS, parts.join(';'))
}
