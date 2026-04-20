import { useInput } from '@hermes/ink'
import { useStore } from '@nanostores/react'

import type {
  ApprovalRespondResponse,
  SecretRespondResponse,
  SudoRespondResponse,
  VoiceRecordResponse
} from '../gatewayTypes.js'
import { writeOsc52Clipboard } from '../lib/osc52.js'

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

  const voiceStop = () => {
    voice.setRecording(false)
    voice.setProcessing(true)

    gateway
      .rpc<VoiceRecordResponse>('voice.record', { action: 'stop' })
      .then(r => {
        if (!r) {
          return
        }

        const transcript = String(r.text || '').trim()

        if (!transcript) {
          return actions.sys('voice: no speech detected')
        }

        cActions.setInput(prev => (prev ? `${prev}${/\s$/.test(prev) ? '' : ' '}${transcript}` : transcript))
      })
      .catch((e: Error) => actions.sys(`voice error: ${e.message}`))
      .finally(() => {
        voice.setProcessing(false)
        patchUiState({ status: 'ready' })
      })
  }

  const voiceStart = () =>
    gateway
      .rpc<VoiceRecordResponse>('voice.record', { action: 'start' })
      .then(r => {
        if (!r) {
          return
        }

        voice.setRecording(true)
        patchUiState({ status: 'recording…' })
      })
      .catch((e: Error) => actions.sys(`voice error: ${e.message}`))

  useInput((ch, key) => {
    const live = getUiState()

    if (isBlocked) {
      if (overlay.pager) {
        if (key.return || ch === ' ') {
          const nextOffset = overlay.pager.offset + pagerPageSize

          patchOverlayState({
            pager: nextOffset >= overlay.pager.lines.length ? null : { ...overlay.pager, offset: nextOffset }
          })
        } else if (key.escape || isCtrl(key, ch, 'c') || ch === 'q') {
          patchOverlayState({ pager: null })
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

    if (key.ctrl && key.shift && ch.toLowerCase() === 'c') {
      return copySelection()
    }

    if (key.escape && terminal.hasSelection) {
      return clearSelection()
    }

    if (key.upArrow && !cState.inputBuf.length) {
      cycleQueue(1) || cycleHistory(-1)

      return
    }

    if (key.downArrow && !cState.inputBuf.length) {
      cycleQueue(-1) || cycleHistory(1)

      return
    }

    if (isCtrl(key, ch, 'c')) {
      if (terminal.hasSelection) {
        return copySelection()
      }

      const inputSel = getInputSelection()

      if (inputSel && inputSel.end > inputSel.start) {
        writeOsc52Clipboard(inputSel.value.slice(inputSel.start, inputSel.end))
        inputSel.clear()

        return
      }

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

    if (isCtrl(key, ch, 'd')) {
      return actions.die()
    }

    if (isCtrl(key, ch, 'l')) {
      if (actions.guardBusySessionSwitch()) {
        return
      }

      patchUiState({ status: 'forging session…' })

      return actions.newSession()
    }

    if (isCtrl(key, ch, 'b')) {
      return voice.recording ? voiceStop() : voiceStart()
    }

    if (isCtrl(key, ch, 'g')) {
      return cActions.openEditor()
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

    if (isCtrl(key, ch, 'k') && cRefs.queueRef.current.length && live.sid) {
      const next = cActions.dequeue()

      if (next) {
        cActions.setQueueEdit(null)
        actions.dispatchSubmission(next)
      }
    }
  })

  return { pagerPageSize }
}
