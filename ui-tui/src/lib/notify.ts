import { execFile, spawn } from 'node:child_process'
import { platform } from 'node:os'

import { getAttentionHook } from '../app/attentionConfigStore.js'

export type AttentionEvent =
  | 'approval.needed'
  | 'input.needed'
  | 'sudo.needed'
  | 'turn.blocked'
  | 'turn.completed'

export type AttentionPayload = {
  event: AttentionEvent
  message: string
  session_id: null | string
  subtitle?: string
}

export type NotifyDependencies = {
  exec?: typeof execFile
  platform?: () => string
}

export const ATTENTION_EVENT_TITLES: Record<AttentionEvent, string> = {
  'approval.needed': 'Approval needed',
  'input.needed': 'Input needed',
  'sudo.needed': 'Sudo password needed',
  'turn.blocked': 'Turn failed',
  'turn.completed': 'Turn complete'
}

// Fire-and-forget `paplay` on Linux. Wayland compositors swallow BEL as a
// visual flash, so bell_* flags also play a system sound on PulseAudio /
// PipeWire systems (#25022). paplay ships with both; a missing binary raises
// the spawn error absorbed below. Shared by turn-complete and prompt-open
// notifications so the audible cue matches on both.
const playAttentionSound = (deps: NotifyDependencies): void => {
  if ((deps.platform?.() ?? platform()) !== 'linux') {
    return
  }

  const exec = deps.exec ?? execFile

  exec(
    'paplay',
    ['/usr/share/sounds/freedesktop/stereo/message-new-instant.oga'],
    { timeout: 10_000 },
    () => {
      // ENOENT and friends are the signal this box has no PulseAudio — no
      // point surfacing or retrying.
    }
  )
}

/**
 * One attention cue: BEL (terminals/tmux route it to bell-action; works over
 * SSH) plus the paplay fallback for hosts where BEL is silent.
 */
export const ringBell = (stdout?: NodeJS.WriteStream, deps: NotifyDependencies = {}): void => {
  if (stdout?.isTTY) {
    stdout.write('\x07')
  }

  playAttentionSound(deps)
}

// Message text carried out of the TUI process into a user-configured hook —
// collapse newlines and cap length so a hostile/verbose model turn can't turn
// the payload into a log-flooding or quoting hazard (#46357).
const ATTENTION_MESSAGE_MAX = 200

const sanitizeAttentionMessage = (text: string): string => {
  const flat = text.replace(/\s+/g, ' ').trim()

  return flat.length > ATTENTION_MESSAGE_MAX ? `${flat.slice(0, ATTENTION_MESSAGE_MAX)}…` : flat
}

/**
 * Fire the user's `display.tui_attention_hook` for one attention-worthy
 * moment (#46357). The command runs detached with stdio ignored, so it can
 * never block the renderer; spawn failures only surface in the debug log.
 * The payload reaches the command as argv (event, title, message) and as
 * HERMES_ATTENTION_* env vars — argv for `notify-send`-style one-liners, env
 * for scripts that want the structured fields.
 */
export const notifyAttention = (payload: AttentionPayload): void => {
  const hook = getAttentionHook()

  if (!hook.enabled || !hook.command) {
    return
  }

  const [file, ...baseArgs] = hook.command.trim().split(/\s+/)
  const message = sanitizeAttentionMessage(payload.message)

  try {
    const child = spawn(file, [...baseArgs, payload.event, ATTENTION_EVENT_TITLES[payload.event], message], {
      detached: true,
      env: {
        ...process.env,
        HERMES_ATTENTION_EVENT: payload.event,
        HERMES_ATTENTION_MESSAGE: message,
        HERMES_ATTENTION_SESSION_ID: payload.session_id ?? '',
        HERMES_ATTENTION_SUBTITLE: ATTENTION_EVENT_TITLES[payload.event],
        HERMES_ATTENTION_TITLE: 'Hermes'
      },
      stdio: 'ignore'
    })

    child.on('error', err => {
      process.stderr.write(`hermes-tui: attention hook failed: ${String(err)}\n`)
    })
    child.unref()
  } catch (err) {
    process.stderr.write(`hermes-tui: attention hook failed: ${String(err)}\n`)
  }
}
