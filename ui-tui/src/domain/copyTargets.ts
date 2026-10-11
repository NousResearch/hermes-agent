/**
 * What `/copy code` and `/copy cmd` copy — mirrors hermes_cli/copy_targets.py.
 * Scope grammar follows Muse Code's `/copy` scope picker (full response / one
 * code block / one command), typed as an argument so CLI and TUI agree.
 */

import type { TranscriptMessage } from '@hermes/shared/gateway-events'

import { parseCodeFences } from './codeFence.js'

export interface CopyItem {
  label: string
  text: string
}

/** Closed fences of the newest assistant message that has at least one. */
export function latestCodeBlocks(messages: readonly { role: string; text: string }[]): CopyItem[] {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]!

    if (msg.role !== 'assistant') {
      continue
    }

    const fences = parseCodeFences(msg.text).filter(f => f.closed)

    if (fences.length) {
      return fences.map(f => ({ label: f.language || 'text', text: f.rawContent }))
    }
  }

  return []
}

/** Shell commands of the newest turn that ran any, from the gateway's
 *  `session.history` projection (tool rows carry the call's `args`). */
export function latestCommands(messages: readonly TranscriptMessage[]): CopyItem[] {
  const turn: CopyItem[] = []

  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]!

    if (msg.role === 'user') {
      if (turn.length) {
        break
      }

      continue
    }

    const command = msg.role === 'tool' && msg.name === 'terminal' ? msg.args?.command : undefined

    if (typeof command === 'string' && command.trim()) {
      turn.push({ label: '$', text: command.trim() })
    }
  }

  return turn.reverse()
}
