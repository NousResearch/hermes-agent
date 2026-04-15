import { type ScrollBoxHandle, useInput } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import type { Dispatch, RefObject, SetStateAction } from 'react'

import type { Msg } from '../types.js'

import type { GatewayServices } from './interfaces.js'
import { $isBlocked, $overlayState, patchOverlayState } from './overlayStore.js'
import { getUiState, patchUiState } from './uiStore.js'
import type { ComposerActions, ComposerRefs, ComposerState } from './useComposerState.js'
import type { TurnActions, TurnRefs } from './useTurnState.js'

export interface InputHandlerActions {
  answerClarify: (answer: string) => void
  appendMessage: (msg: Msg) => void
  die: () => void
  dispatchSubmission: (full: string) => void
  guardBusySessionSwitch: (what?: string) => boolean
  newSession: (msg?: string) => void
  sys: (text: string) => void
}

export interface InputHandlerContext {
  actions: InputHandlerActions
  composer: {
    actions: ComposerActions
    refs: ComposerRefs
    state: ComposerState
  }
  gateway: GatewayServices
  terminal: {
    hasSelection: boolean
    scrollRef: RefObject<ScrollBoxHandle | null>
    scrollWithSelection: (delta: number) => void
    selection: {
      copySelection: () => string
    }
    stdout?: NodeJS.WriteStream
  }
  turn: {
    actions: TurnActions
    refs: TurnRefs
  }
  voice: {
    recording: boolean
    setProcessing: Dispatch<SetStateAction<boolean>>
    setRecording: Dispatch<SetStateAction<boolean>>
  }
  wheelStep: number
}

export interface InputHandlerResult {
  pagerPageSize: number
}

export function useInputHandlers(ctx: InputHandlerContext): InputHandlerResult {
  const { actions, composer, gateway, terminal, turn, voice, wheelStep } = ctx
  const overlay = useStore($overlayState)
  const isBlocked = useStore($isBlocked)
  const pagerPageSize = Math.max(5, (terminal.stdout?.rows ?? 24) - 6)

  const ctrl = (key: { ctrl: boolean }, ch: string, target: string) => key.ctrl && ch.toLowerCase() === target

  const copySelection = () => {
    if (terminal.selection.copySelection()) {
      actions.sys('copied selection')
    }
  }

  useInput((ch, key) => {
    const live = getUiState()

    if (isBlocked) {
      if (overlay.pager) {
        if (key.return || ch === ' ') {
          const next = overlay.pager.offset + pagerPageSize

          patchOverlayState({
            pager: next >= overlay.pager.lines.length ? null : { ...overlay.pager, offset: next }
          })
        } else if (key.escape || ctrl(key, ch, 'c') || ch === 'q') {
          patchOverlayState({ pager: null })
        }

        return
      }

      if (ctrl(key, ch, 'c')) {
        if (overlay.clarify) {
          actions.answerClarify('')
        } else if (overlay.approval) {
          gateway.rpc('approval.respond', { choice: 'deny', session_id: live.sid }).then(r => {
            if (!r) {
              return
            }

            patchOverlayState({ approval: null })
            actions.sys('denied')
          })
        } else if (overlay.sudo) {
          gateway.rpc('sudo.respond', { password: '', request_id: overlay.sudo.requestId }).then(r => {
            if (!r) {
              return
            }

            patchOverlayState({ sudo: null })
            actions.sys('sudo cancelled')
          })
        } else if (overlay.secret) {
          gateway.rpc('secret.respond', { request_id: overlay.secret.requestId, value: '' }).then(r => {
            if (!r) {
              return
            }

            patchOverlayState({ secret: null })
            actions.sys('secret entry cancelled')
          })
        } else if (overlay.modelPicker) {
          patchOverlayState({ modelPicker: false })
        } else if (overlay.picker) {
          patchOverlayState({ picker: false })
        }
      } else if (key.escape && overlay.picker) {
        patchOverlayState({ picker: false })
      }

      return
    }

    if (
      composer.state.completions.length &&
      composer.state.input &&
      composer.state.historyIdx === null &&
      (key.upArrow || key.downArrow)
    ) {
      composer.actions.setCompIdx(index =>
        key.upArrow
          ? (index - 1 + composer.state.completions.length) % composer.state.completions.length
          : (index + 1) % composer.state.completions.length
      )

      return
    }

    if (key.wheelUp) {
      terminal.scrollWithSelection(-wheelStep)

      return
    }

    if (key.wheelDown) {
      terminal.scrollWithSelection(wheelStep)

      return
    }

    if (key.shift && key.upArrow) {
      terminal.scrollWithSelection(-1)

      return
    }

    if (key.shift && key.downArrow) {
      terminal.scrollWithSelection(1)

      return
    }

    if (key.pageUp || key.pageDown) {
      const viewport = terminal.scrollRef.current?.getViewportHeight() ?? Math.max(6, (terminal.stdout?.rows ?? 24) - 8)
      const step = Math.max(4, viewport - 2)

      terminal.scrollWithSelection(key.pageUp ? -step : step)

      return
    }

    if (key.ctrl && key.shift && ch.toLowerCase() === 'c') {
      copySelection()

      return
    }

    if (key.upArrow && !composer.state.inputBuf.length) {
      if (composer.refs.queueRef.current.length) {
        const index =
          composer.state.queueEditIdx === null
            ? 0
            : (composer.state.queueEditIdx + 1) % composer.refs.queueRef.current.length

        composer.actions.setQueueEdit(index)
        composer.actions.setHistoryIdx(null)
        composer.actions.setInput(composer.refs.queueRef.current[index] ?? '')
      } else if (composer.refs.historyRef.current.length) {
        const index =
          composer.state.historyIdx === null
            ? composer.refs.historyRef.current.length - 1
            : Math.max(0, composer.state.historyIdx - 1)

        if (composer.state.historyIdx === null) {
          composer.refs.historyDraftRef.current = composer.state.input
        }

        composer.actions.setHistoryIdx(index)
        composer.actions.setQueueEdit(null)
        composer.actions.setInput(composer.refs.historyRef.current[index] ?? '')
      }

      return
    }

    if (key.downArrow && !composer.state.inputBuf.length) {
      if (composer.refs.queueRef.current.length) {
        const index =
          composer.state.queueEditIdx === null
            ? composer.refs.queueRef.current.length - 1
            : (composer.state.queueEditIdx - 1 + composer.refs.queueRef.current.length) %
              composer.refs.queueRef.current.length

        composer.actions.setQueueEdit(index)
        composer.actions.setHistoryIdx(null)
        composer.actions.setInput(composer.refs.queueRef.current[index] ?? '')
      } else if (composer.state.historyIdx !== null) {
        const next = composer.state.historyIdx + 1

        if (next >= composer.refs.historyRef.current.length) {
          composer.actions.setHistoryIdx(null)
          composer.actions.setInput(composer.refs.historyDraftRef.current)
        } else {
          composer.actions.setHistoryIdx(next)
          composer.actions.setInput(composer.refs.historyRef.current[next] ?? '')
        }
      }

      return
    }

    if (ctrl(key, ch, 'c')) {
      if (terminal.hasSelection) {
        copySelection()
      } else if (live.busy && live.sid) {
        turn.actions.interruptTurn({
          appendMessage: actions.appendMessage,
          gw: gateway.gw,
          sid: live.sid,
          sys: actions.sys
        })
      } else if (composer.state.input || composer.state.inputBuf.length) {
        composer.actions.clearIn()
      } else {
        return actions.die()
      }

      return
    }

    if (ctrl(key, ch, 'd')) {
      return actions.die()
    }

    if (ctrl(key, ch, 'l')) {
      if (actions.guardBusySessionSwitch()) {
        return
      }

      patchUiState({ status: 'forging session…' })
      actions.newSession()

      return
    }

    if (ctrl(key, ch, 'b')) {
      if (voice.recording) {
        voice.setRecording(false)
        voice.setProcessing(true)
        gateway
          .rpc('voice.record', { action: 'stop' })
          .then((r: any) => {
            if (!r) {
              return
            }

            const transcript = String(r?.text || '').trim()

            if (transcript) {
              composer.actions.setInput(prev =>
                prev ? `${prev}${/\s$/.test(prev) ? '' : ' '}${transcript}` : transcript
              )
            } else {
              actions.sys('voice: no speech detected')
            }
          })
          .catch((e: Error) => actions.sys(`voice error: ${e.message}`))
          .finally(() => {
            voice.setProcessing(false)
            patchUiState({ status: 'ready' })
          })
      } else {
        gateway
          .rpc('voice.record', { action: 'start' })
          .then((r: any) => {
            if (!r) {
              return
            }

            voice.setRecording(true)
            patchUiState({ status: 'recording…' })
          })
          .catch((e: Error) => actions.sys(`voice error: ${e.message}`))
      }

      return
    }

    if (ctrl(key, ch, 'g')) {
      return composer.actions.openEditor()
    }

    if (key.tab && composer.state.completions.length) {
      const row = composer.state.completions[composer.state.compIdx]

      if (row?.text) {
        const text =
          composer.state.input.startsWith('/') && row.text.startsWith('/') && composer.state.compReplace > 0
            ? row.text.slice(1)
            : row.text

        composer.actions.setInput(composer.state.input.slice(0, composer.state.compReplace) + text)
      }

      return
    }

    if (ctrl(key, ch, 'k') && composer.refs.queueRef.current.length && live.sid) {
      const next = composer.actions.dequeue()

      if (next) {
        composer.actions.setQueueEdit(null)
        actions.dispatchSubmission(next)
      }
    }
  })

  return { pagerPageSize }
}
