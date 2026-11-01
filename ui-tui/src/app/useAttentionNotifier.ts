import { type TerminalNotification, useTerminalNotification } from '@hermes/ink'
import { useCallback, useState } from 'react'

import { messages } from '../i18n/runtime.js'

import {
  type AttentionKind,
  type NotificationTerminal,
  notificationTerminalForEnv,
  warpAgentPayload
} from './notificationTerminal.js'
import { $uiState } from './uiStore.js'

type Notifiers = Pick<TerminalNotification, 'notifyGhostty' | 'notifyITerm2' | 'notifyKitty'>

const SENDERS: Record<NotificationTerminal, (n: Notifiers, kind: AttentionKind, message: string) => void> = {
  ghostty: (n, _kind, message) => n.notifyGhostty({ message, title: 'Hermes' }),
  iterm2: (n, _kind, message) => n.notifyITerm2({ message, title: 'Hermes' }),
  kitty: (n, _kind, message) => n.notifyKitty({ id: 1, message, title: 'Hermes' }),
  // Warp's OSC 777 shares Ghostty's `777;notify;<title>;<body>` wire shape.
  warp: (n, kind, message) =>
    n.notifyGhostty({
      message: warpAgentPayload(kind, message, {
        cwd: process.cwd(),
        protocolVersion: process.env.WARP_CLI_AGENT_PROTOCOL_VERSION,
        sessionId: $uiState.get().sid ?? ''
      }),
      title: 'warp://cli-agent'
    })
}

/**
 * display.bell_on_complete / bell_on_prompt: the two flags (synced from config) and the notifier
 * they gate. BEL always (tmux bell-action, SSH), plus the detected terminal's desktop notification
 * (OSC 9 / 99 / 777, Warp's OSC 777 agent event), written through Ink so it never splices a frame.
 */
export function useAttentionNotifier() {
  const [bellOnComplete, setBellOnComplete] = useState(false)
  const [bellOnPrompt, setBellOnPrompt] = useState(false)
  const { notifyBell, notifyGhostty, notifyITerm2, notifyKitty } = useTerminalNotification()

  const notifyAttention = useCallback(
    (kind: AttentionKind) => {
      const terminal = notificationTerminalForEnv(process.env)

      notifyBell()

      if (terminal) {
        SENDERS[terminal]({ notifyGhostty, notifyITerm2, notifyKitty }, kind, messages().notify[kind])
      }
    },
    [notifyBell, notifyGhostty, notifyITerm2, notifyKitty]
  )

  return { bellOnComplete, bellOnPrompt, notifyAttention, setBellOnComplete, setBellOnPrompt }
}
