import { describe, expect, it } from 'vitest'

import { env, supportsOsc52Clipboard } from '../../utils/env.js'

import { shouldEmitClipboardSequence, shouldUseNativeClipboard } from './osc.js'

describe('shouldEmitClipboardSequence', () => {
  it('suppresses local multiplexer clipboard OSC by default', () => {
    expect(shouldEmitClipboardSequence({ TMUX: '/tmp/tmux-1/default,1,0' } as NodeJS.ProcessEnv)).toBe(false)
    expect(shouldEmitClipboardSequence({ STY: '1234.pts-0.host' } as NodeJS.ProcessEnv)).toBe(false)
  })

  it('keeps OSC enabled for remote or plain local terminals', () => {
    expect(
      shouldEmitClipboardSequence({ SSH_CONNECTION: '1', TMUX: '/tmp/tmux-1/default,1,0' } as NodeJS.ProcessEnv)
    ).toBe(true)
    expect(shouldEmitClipboardSequence({ TERM: 'xterm-256color' } as NodeJS.ProcessEnv)).toBe(true)
  })

  it('honors explicit env override', () => {
    expect(
      shouldEmitClipboardSequence({
        HERMES_TUI_CLIPBOARD_OSC52: '1',
        TMUX: '/tmp/tmux-1/default,1,0'
      } as NodeJS.ProcessEnv)
    ).toBe(true)
    expect(
      shouldEmitClipboardSequence({ HERMES_TUI_COPY_OSC52: '0', TERM: 'xterm-256color' } as NodeJS.ProcessEnv)
    ).toBe(false)
  })

  it('HERMES_TUI_FORCE_OSC52 takes precedence over TMUX suppression', () => {
    // Without the override, local-in-tmux suppresses the OSC 52 sequence
    // so the terminal multiplexer path wins. FORCE_OSC52=1 flips that
    // back on for users whose tmux config supports passthrough.
    expect(shouldEmitClipboardSequence({ TMUX: '/tmp/t,1,0' } as NodeJS.ProcessEnv)).toBe(false)
    expect(
      shouldEmitClipboardSequence({
        HERMES_TUI_FORCE_OSC52: '1',
        TMUX: '/tmp/t,1,0'
      } as NodeJS.ProcessEnv)
    ).toBe(true)
  })

  it('HERMES_TUI_FORCE_OSC52=0 suppresses OSC 52 even for remote or plain terminals', () => {
    expect(
      shouldEmitClipboardSequence({
        HERMES_TUI_FORCE_OSC52: '0',
        SSH_CONNECTION: '1'
      } as NodeJS.ProcessEnv)
    ).toBe(false)
  })
})

describe('supportsOsc52Clipboard', () => {
  // Terminals known to correctly implement OSC 52. On these, setClipboard()
  // skips the native-tool safety net (wl-copy/xclip/pbcopy) to avoid racing
  // the terminal's own clipboard write. Values must match what
  // detectTerminal() in utils/env.ts returns — TERM=xterm-ghostty normalises
  // to 'ghostty', TERM_PROGRAM=WezTerm stays 'WezTerm', etc.
  it.each(['ghostty', 'kitty', 'WezTerm', 'windows-terminal', 'vscode'])(
    'returns true for allowlisted terminal %s',
    terminal => {
      expect(supportsOsc52Clipboard(terminal)).toBe(true)
    }
  )

  // Intentionally conservative — iTerm2 disables OSC 52 by default; Alacritty
  // and GNOME Terminal detection is unreliable; xterm/Terminal.app lack
  // reliable OSC 52. These keep the existing native-safety-net behaviour.
  it.each(['iTerm.app', 'alacritty', 'Apple_Terminal', 'xterm', 'tmux', 'screen', 'cursor', 'WarpTerminal', ''])(
    'returns false for non-allowlisted terminal %s',
    terminal => {
      expect(supportsOsc52Clipboard(terminal)).toBe(false)
    }
  )

  it('returns false when terminal is null (detection failed)', () => {
    expect(supportsOsc52Clipboard(null)).toBe(false)
  })

  it('defaults to the module-level detected terminal when no argument is passed', () => {
    // With no argument, uses env.terminal detected at module load. We don't
    // know what that is in CI, but the call must return a boolean (not throw)
    // and the result must match calling with env.terminal explicitly.
    expect(typeof supportsOsc52Clipboard()).toBe('boolean')
    expect(supportsOsc52Clipboard()).toBe(supportsOsc52Clipboard(env.terminal))
  })
})

// shouldUseNativeClipboard() encodes the gating logic that setClipboard()
// uses to decide whether to fire copyNative(). Testing it directly (rather
// than mocking copyNative inside setClipboard) matches the package's
// existing style — tests pass env/terminal as arguments instead of using
// vi.mock — and gives broader coverage of the env x terminal matrix.
describe('shouldUseNativeClipboard', () => {
  it('returns false over SSH (native would write to remote clipboard)', () => {
    // Over SSH the user's terminal is on the local end of the pty;
    // pbcopy/wl-copy/xclip on the remote machine would write to the wrong
    // clipboard. OSC 52 is the right path. Existing behaviour, preserved.
    expect(shouldUseNativeClipboard({ SSH_CONNECTION: '1' } as NodeJS.ProcessEnv, 'xterm')).toBe(false)
    expect(shouldUseNativeClipboard({ SSH_CONNECTION: '1' } as NodeJS.ProcessEnv, 'ghostty')).toBe(false)
    expect(shouldUseNativeClipboard({ SSH_CONNECTION: '1' } as NodeJS.ProcessEnv, null)).toBe(false)
  })

  it('returns true on plain local terminals (existing behaviour)', () => {
    // Non-allowlisted terminals — xterm, GNOME Terminal, Apple_Terminal,
    // alacritty (detection unreliable), iTerm2 (OSC 52 off by default).
    // These keep the native safety net firing. This is the bulk of the
    // existing user base; behaviour must not regress.
    expect(shouldUseNativeClipboard({} as NodeJS.ProcessEnv, 'xterm')).toBe(true)
    expect(shouldUseNativeClipboard({} as NodeJS.ProcessEnv, 'iTerm.app')).toBe(true)
    expect(shouldUseNativeClipboard({} as NodeJS.ProcessEnv, 'Apple_Terminal')).toBe(true)
    expect(shouldUseNativeClipboard({} as NodeJS.ProcessEnv, 'alacritty')).toBe(true)
    expect(shouldUseNativeClipboard({} as NodeJS.ProcessEnv, null)).toBe(true)
  })

  it('returns false on allowlisted local terminals (the race-fix case)', () => {
    // Ghostty / kitty / WezTerm / Windows Terminal / VS Code — OSC 52
    // alone is reliable, native fallback racing it can corrupt the
    // clipboard (the wl-copy on Wayland symptom this PR fixes).
    expect(shouldUseNativeClipboard({} as NodeJS.ProcessEnv, 'ghostty')).toBe(false)
    expect(shouldUseNativeClipboard({} as NodeJS.ProcessEnv, 'kitty')).toBe(false)
    expect(shouldUseNativeClipboard({} as NodeJS.ProcessEnv, 'WezTerm')).toBe(false)
    expect(shouldUseNativeClipboard({} as NodeJS.ProcessEnv, 'windows-terminal')).toBe(false)
    expect(shouldUseNativeClipboard({} as NodeJS.ProcessEnv, 'vscode')).toBe(false)
  })

  it('returns true inside tmux even on allowlisted outer terminal', () => {
    // detectTerminal() prefers TERM_PROGRAM over TMUX, so a tmux session
    // inside Ghostty reports terminal='ghostty'. But setClipboard() goes
    // through tmux load-buffer there, not raw OSC 52 — the wl-copy race
    // doesn't apply. Native is still useful since tmux's outer-terminal
    // forwarding depends on `set -g set-clipboard` + `allow-passthrough`.
    expect(shouldUseNativeClipboard({ TMUX: '/tmp/t,1,0' } as NodeJS.ProcessEnv, 'ghostty')).toBe(true)
    expect(shouldUseNativeClipboard({ TMUX: '/tmp/t,1,0' } as NodeJS.ProcessEnv, 'kitty')).toBe(true)
    expect(shouldUseNativeClipboard({ TMUX: '/tmp/t,1,0' } as NodeJS.ProcessEnv, 'WezTerm')).toBe(true)
    expect(shouldUseNativeClipboard({ TMUX: '/tmp/t,1,0' } as NodeJS.ProcessEnv, 'vscode')).toBe(true)
  })

  it('returns true inside GNU screen even on allowlisted outer terminal', () => {
    // Same reasoning as TMUX — STY indicates we're inside screen, which
    // has its own escape-sequence handling and we don't emit raw OSC 52.
    expect(shouldUseNativeClipboard({ STY: '1234.pts-0.host' } as NodeJS.ProcessEnv, 'ghostty')).toBe(true)
    expect(shouldUseNativeClipboard({ STY: '1234.pts-0.host' } as NodeJS.ProcessEnv, 'kitty')).toBe(true)
  })

  it('returns true when OSC 52 emission is disabled via HERMES_TUI_FORCE_OSC52=0', () => {
    // If we suppress OSC 52 (user override) AND skip native, the clipboard
    // write becomes a no-op. So when OSC 52 is off, native is the only
    // remaining path — keep it on regardless of terminal allowlist.
    expect(shouldUseNativeClipboard({ HERMES_TUI_FORCE_OSC52: '0' } as NodeJS.ProcessEnv, 'ghostty')).toBe(true)
    expect(shouldUseNativeClipboard({ HERMES_TUI_FORCE_OSC52: '0' } as NodeJS.ProcessEnv, 'kitty')).toBe(true)
    expect(shouldUseNativeClipboard({ HERMES_TUI_CLIPBOARD_OSC52: '0' } as NodeJS.ProcessEnv, 'WezTerm')).toBe(true)
    expect(shouldUseNativeClipboard({ HERMES_TUI_COPY_OSC52: 'no' } as NodeJS.ProcessEnv, 'vscode')).toBe(true)
  })

  it('returns true under TMUX even with HERMES_TUI_FORCE_OSC52=1 on an allowlisted terminal (tmux load-buffer path)', () => {
    // FORCE_OSC52=1 is the user explicitly opting INTO OSC 52 (e.g. they
    // have tmux set up for passthrough). On an allowlisted terminal the
    // race-avoidance still applies.
    expect(
      shouldUseNativeClipboard({ HERMES_TUI_FORCE_OSC52: '1', TMUX: '/tmp/t,1,0' } as NodeJS.ProcessEnv, 'ghostty')
      // TMUX guard wins — native still fires because we're going through
      // tmux load-buffer, not raw OSC 52 to the terminal.
    ).toBe(true)
  })

  it('SSH_CONNECTION takes precedence over allowlisted terminal', () => {
    // Even on Ghostty, if we're SSH'd in we shouldn't run pbcopy on the
    // remote machine — the user's clipboard is on the other end.
    expect(shouldUseNativeClipboard({ SSH_CONNECTION: '1' } as NodeJS.ProcessEnv, 'ghostty')).toBe(false)
  })

  it('SSH_CONNECTION takes precedence over TMUX', () => {
    // Combined: SSH'd in and inside tmux on the remote. SSH_CONNECTION
    // gate fires first, native stays off (we use OSC 52 to reach the
    // local terminal).
    expect(shouldUseNativeClipboard({ SSH_CONNECTION: '1', TMUX: '/tmp/t,1,0' } as NodeJS.ProcessEnv, 'xterm')).toBe(
      false
    )
  })

  it('defaults env to process.env and terminal to the module-detected terminal when no args passed', () => {
    // Smoke test: no args is a valid call shape for the convenience seam.
    // shouldUseNativeClipboard() defaults `terminal` to envModule.terminal
    // (the module-level detected terminal), not null. Returns a boolean
    // without throwing.
    expect(typeof shouldUseNativeClipboard()).toBe('boolean')
  })
})
