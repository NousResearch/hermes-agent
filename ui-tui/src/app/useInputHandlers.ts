import { useInput } from '@hermes/ink'
import { useStore } from '@nanostores/react'

import type {
  ApprovalRespondResponse,
  ConfigSetResponse,
  SecretRespondResponse,
  SudoRespondResponse,
  VoiceRecordResponse
} from '../gatewayTypes.js'
import { isAction, isMac, isVoiceToggleKey } from '../lib/platform.js'

import { getInputSelection } from './inputSelectionStore.js'
import type { InputHandlerContext, InputHandlerResult } from './interfaces.js'
import { $isBlocked, $overlayState, patchOverlayState } from './overlayStore.js'
import { turnController } from './turnController.js'
import { patchTurnState } from './turnStore.js'
import { getUiState, patchUiState } from './uiStore.js'

const isCtrl = (key: { ctrl: boolean }, ch: string, target: string) => key.ctrl && ch.toLowerCase() === target

export function useInputHandlers(ctx: InputHandlerContext): InputHandlerResult {
  const { actions, composer, gateway, terminal, voice, wheelStep } = ctx
  const { actions: cActions, refs: cRefs, state: cState } = composer

  const overlay = useStore($overlayState)
  const isBlocked = useStore($isBlocked)
  const pagerPageSize = Math.max(5, (terminal.stdout?.rows ?? 24) - 6)

  const copySelection = () => {
    // ink's copySelection() already calls setClipboard() which handles
    // pbcopy (macOS), wl-copy/xclip (Linux), tmux, and OSC 52 fallback.
    const text = terminal.selection.copySelection()

    if (text) {
      actions.sys(`copied ${text.length} chars`)
    }
  }

  const clearSelection = () => {
    terminal.selection.clearSelection()
  }

  const cancelOverlayFromCtrlC = () => {
    if (overlay.clarify) {
      return actions.answerClarify('')
    }

    if (overlay.approval) {
      return gateway
        .rpc<ApprovalRespondResponse>('approval.respond', { choice: 'deny', session_id: getUiState().sid })
        .then(r => r && (patchOverlayState({ approval: null }), patchTurnState({ outcome: 'denied' })))
    }

    if (overlay.sudo) {
      return gateway
        .rpc<SudoRespondResponse>('sudo.respond', { password: '', request_id: overlay.sudo.requestId })
        .then(r => r && (patchOverlayState({ sudo: null }), actions.sys('sudo cancelled')))
    }

    if (overlay.secret) {
      return gateway
        .rpc<SecretRespondResponse>('secret.respond', { request_id: overlay.secret.requestId, value: '' })
        .then(r => r && (patchOverlayState({ secret: null }), actions.sys('secret entry cancelled')))
    }

    if (overlay.modelPicker) {
      return patchOverlayState({ modelPicker: false })
    }

    if (overlay.skillsHub) {
      return patchOverlayState({ skillsHub: false })
    }

    if (overlay.picker) {
      return patchOverlayState({ picker: false })
    }

    if (overlay.agents) {
      return patchOverlayState({ agents: false })
    }
  }

  const cycleQueue = (dir: 1 | -1) => {
    const len = cRefs.queueRef.current.length

    if (!len) {
      return false
    }

    const index = cState.queueEditIdx === null ? (dir > 0 ? 0 : len - 1) : (cState.queueEditIdx + dir + len) % len

    cActions.setQueueEdit(index)
    cActions.setHistoryIdx(null)
    cActions.setInput(cRefs.queueRef.current[index] ?? '')

    return true
  }

  const cycleHistory = (dir: 1 | -1) => {
    const h = cRefs.historyRef.current
    const cur = cState.historyIdx

    if (dir < 0) {
      if (!h.length) {
        return
      }

      if (cur === null) {
        cRefs.historyDraftRef.current = cState.input
      }

      const index = cur === null ? h.length - 1 : Math.max(0, cur - 1)

      cActions.setHistoryIdx(index)
      cActions.setQueueEdit(null)
      cActions.setInput(h[index] ?? '')

      return
    }

    if (cur === null) {
      return
    }

    const next = cur + 1

    if (next >= h.length) {
      cActions.setHistoryIdx(null)
      cActions.setInput(cRefs.historyDraftRef.current)
    } else {
      cActions.setHistoryIdx(next)
      cActions.setInput(h[next] ?? '')
    }
  }

  // CLI parity: Ctrl+B toggles the VAD-driven continuous recording loop
  // (NOT the voice-mode umbrella bit). The mode is enabled via /voice on;
  // Ctrl+B while the mode is off sys-nudges the user. While the mode is
  // on, the first press starts a continuous loop (gateway → start_continuous,
  // VAD auto-stop → transcribe → auto-restart), a subsequent press stops it.
  // The gateway publishes voice.status + voice.transcript events that
  // createGatewayEventHandler turns into UI badges and composer injection.
  const voiceRecordToggle = () => {
    if (!voice.enabled) {
      return actions.sys('voice: mode is off — enable with /voice on')
    }

    const starting = !voice.recording
    const action = starting ? 'start' : 'stop'

    // Optimistic UI — flip the REC badge immediately so the user gets
    // feedback while the RPC round-trips; the voice.status event is the
    // authoritative source and may correct us.
    if (starting) {
      voice.setRecording(true)
    } else {
      voice.setRecording(false)
      voice.setProcessing(false)
    }

    gateway
      .rpc<VoiceRecordResponse>('voice.record', { action })
      .catch((e: Error) => {
        // Revert optimistic UI on failure.
        if (starting) {
          voice.setRecording(false)
        }

        actions.sys(`voice error: ${e.message}`)
      })
  }

  useInput((ch, key) => {
    const live = getUiState()

    if (isBlocked) {
      // When approval/clarify/confirm overlays are active, their own useInput
      // handlers must receive keystrokes (arrow keys, numbers, Enter).  Only
      // intercept Ctrl+C here so the user can deny/dismiss — all other keys
      // fall through to the component-level handlers.
      if (overlay.approval || overlay.clarify || overlay.confirm) {
        if (isCtrl(key, ch, 'c')) {
          cancelOverlayFromCtrlC()
        }

        return
      }

      if (overlay.pager) {
        if (key.escape || isCtrl(key, ch, 'c') || ch === 'q') {
          return patchOverlayState({ pager: null })
        }

        const move = (delta: number | 'top' | 'bottom') =>
          patchOverlayState(prev => {
            if (!prev.pager) {
              return prev
            }

            const { lines, offset } = prev.pager
            const max = Math.max(0, lines.length - pagerPageSize)
            const step = delta === 'top' ? -lines.length : delta === 'bottom' ? lines.length : delta
            const next = Math.max(0, Math.min(offset + step, max))

            return next === offset ? prev : { ...prev, pager: { ...prev.pager, offset: next } }
          })

        if (key.upArrow || ch === 'k') {
          return move(-1)
        }

        if (key.downArrow || ch === 'j') {
          return move(1)
        }

        if (key.pageUp || ch === 'b') {
          return move(-pagerPageSize)
        }

        if (ch === 'g') {
          return move('top')
        }

        if (ch === 'G') {
          return move('bottom')
        }

        if (key.return || ch === ' ' || key.pageDown) {
          patchOverlayState(prev => {
            if (!prev.pager) {
              return prev
            }

            const { lines, offset } = prev.pager
            const max = Math.max(0, lines.length - pagerPageSize)

            // Auto-close only when already at the last page — otherwise clamp
            // to `max` so the offset matches what the line/page-back handlers
            // can reach (prevents a snap-back jump on the next ↑/↓/PgUp).
            return offset >= max
              ? { ...prev, pager: null }
              : { ...prev, pager: { ...prev.pager, offset: Math.min(offset + pagerPageSize, max) } }
          })
        }

        return
      }

      if (isCtrl(key, ch, 'c')) {
        cancelOverlayFromCtrlC()
      } else if (key.escape && overlay.picker) {
        patchOverlayState({ picker: false })
      }

      return
    }

    if (cState.completions.length && cState.input && cState.historyIdx === null && (key.upArrow || key.downArrow)) {
      const len = cState.completions.length

      cActions.setCompIdx(i => (key.upArrow ? (i - 1 + len) % len : (i + 1) % len))

      return
    }

    if (key.wheelUp) {
      return terminal.scrollWithSelection(-wheelStep)
    }

    if (key.wheelDown) {
      return terminal.scrollWithSelection(wheelStep)
    }

    if (key.shift && key.upArrow) {
      return terminal.scrollWithSelection(-1)
    }

    if (key.shift && key.downArrow) {
      return terminal.scrollWithSelection(1)
    }

    if (key.pageUp || key.pageDown) {
      const viewport = terminal.scrollRef.current?.getViewportHeight() ?? Math.max(6, (terminal.stdout?.rows ?? 24) - 8)
      const step = Math.max(4, viewport - 2)

      return terminal.scrollWithSelection(key.pageUp ? -step : step)
    }

    if (key.escape && terminal.hasSelection) {
      return clearSelection()
    }

    if (key.upArrow && !cState.inputBuf.length) {
      const inputSel = getInputSelection()
      const cursor = inputSel && inputSel.start === inputSel.end ? inputSel.start : null

      const noLineAbove =
        !cState.input || (cursor !== null && cState.input.lastIndexOf('\n', Math.max(0, cursor - 1)) < 0)

      if (noLineAbove) {
        cycleQueue(1) || cycleHistory(-1)

        return
      }
    }

    if (key.downArrow && !cState.inputBuf.length) {
      const inputSel = getInputSelection()
      const cursor = inputSel && inputSel.start === inputSel.end ? inputSel.start : null
      const noLineBelow = !cState.input || (cursor !== null && cState.input.indexOf('\n', cursor) < 0)

      if (noLineBelow || cState.historyIdx !== null) {
        cycleQueue(-1) || cycleHistory(1)

        return
      }
    }

    if (isAction(key, ch, 'c')) {
      if (terminal.hasSelection) {
        return copySelection()
      }

      const inputSel = getInputSelection()

      if (inputSel && inputSel.end > inputSel.start) {
        inputSel.clear()

        return
      }

      // On macOS, Cmd+C with no selection is a no-op (Ctrl+C below handles interrupt).
      // On non-macOS, isAction uses Ctrl, so fall through to interrupt/clear/exit.
      if (isMac) {
        return
      }
    }

    if (key.ctrl && ch.toLowerCase() === 'c') {
      if (live.busy && live.sid) {
        return turnController.interruptTurn({
          appendMessage: actions.appendMessage,
          gw: gateway.gw,
          sid: live.sid,
          sys: actions.sys
        })
      }

      if (cState.input || cState.inputBuf.length) {
        return cActions.clearIn()
      }

      return actions.die()
    }

    if (isAction(key, ch, 'd')) {
      return actions.die()
    }

    if (isAction(key, ch, 'l')) {
      if (actions.guardBusySessionSwitch()) {
        return
      }

      patchUiState({ status: 'forging session…' })

      return actions.newSession()
    }

    if (isVoiceToggleKey(key, ch)) {
      return voiceRecordToggle()
    }

    if (isAction(key, ch, 'g')) {
      return cActions.openEditor()
    }

    // shift-tab flips yolo without spending a turn (claude-code parity)
    if (key.shift && key.tab && !cState.completions.length) {
      if (!live.sid) {
        return void actions.sys('yolo needs an active session')
      }

      // gateway.rpc swallows errors with its own sys() message and resolves to null,
      // so we only speak when it came back with a real shape. null = rpc already spoke.
      return void gateway.rpc<ConfigSetResponse>('config.set', { key: 'yolo', session_id: live.sid }).then(r => {
        if (r?.value === '1') {
          return actions.sys('yolo on')
        }

        if (r?.value === '0') {
          return actions.sys('yolo off')
        }

        if (r) {
          actions.sys('failed to toggle yolo')
        }
      })
    }

    if (key.tab && cState.completions.length) {
      const row = cState.completions[cState.compIdx]

      if (row?.text) {
        const text =
          cState.input.startsWith('/') && row.text.startsWith('/') && cState.compReplace > 0
            ? row.text.slice(1)
            : row.text

        cActions.setInput(cState.input.slice(0, cState.compReplace) + text)
      }

      return
    }

    if (isAction(key, ch, 'k') && cRefs.queueRef.current.length && live.sid) {
      const next = cActions.dequeue()

      if (next) {
        cActions.setQueueEdit(null)
        actions.dispatchSubmission(next)
      }
    }
  })

  return { pagerPageSize }
}
