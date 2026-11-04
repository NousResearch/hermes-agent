import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef } from 'react'

import { t } from '../i18n/runtime.js'

import type { UseComposerStateResult } from './interfaces.js'
import { $isBlocked, $overlayState, hasSensitivePrompt } from './overlayStore.js'
import {
  $sessionTabs,
  emptyTabDraft,
  findSessionTab,
  hasTabDraft,
  openSessionTab,
  removeSessionTab,
  saveSessionTabDraft,
  type SessionTabAction,
  tabsInDirectory
} from './sessionTabsStore.js'
import { $uiState, getUiState, sessionCwd } from './uiStore.js'
import type { useSessionLifecycle } from './useSessionLifecycle.js'

export function useSessionTabDrafts(composer: UseComposerStateResult) {
  const latest = useRef(composer)
  latest.current = composer

  const capture = useCallback(() => {
    const { refs, state } = latest.current
    saveSessionTabDraft(getUiState().sid, {
      input: refs.inputRef?.current ?? state.input,
      inputBuf: refs.inputBufRef?.current ?? state.inputBuf,
      tokens: refs.tokensRef.current
    })
  }, [])

  const activate = useCallback((view: Parameters<typeof openSessionTab>[0]) => {
    const tab = openSessionTab(view)
    const draft = tab.draft || emptyTabDraft()
    const { actions, refs } = latest.current
    actions.setInput(draft.input)
    actions.setInputBuf([...draft.inputBuf])
    actions.setComposerTokens(draft.tokens.map(token => ({ ...token })))
    actions.setQueueEdit(null)
    actions.setHistoryIdx(null)
    refs.historyDraftRef.current = draft.input
  }, [])

  useEffect(() => capture(), [capture, composer.state.input, composer.state.inputBuf, composer.state.tokens])

  return { capture, activate }
}

export function useSessionTabs(
  session: Pick<ReturnType<typeof useSessionLifecycle>, 'activateLiveSession' | 'newLiveSession'>,
  capture: () => void,
  sys: (message: string) => void
) {
  const ui = useStore($uiState)
  const overlay = useStore($overlayState)
  const pending = useRef(false)
  const cwd = ui.info?.cwd || process.env.HERMES_CWD || process.env.HERMES_TUI_CWD || ''

  useEffect(() => {
    if (!ui.sid || !cwd) {
      return
    }

    openSessionTab({
      id: ui.sid,
      storedId: ui.storedSid || ui.sid,
      cwd,
      title: ui.sessionTitle || ui.info?.title || '',
      running: ui.busy,
      blocked: Boolean(overlay.approval || overlay.clarify || hasSensitivePrompt(overlay))
    })
  }, [cwd, ui.sid, ui.storedSid, ui.sessionTitle, ui.info?.title, ui.busy, overlay])

  return useCallback(
    (action: SessionTabAction) => {
      if ($isBlocked.get() || getUiState().gatewayConnected === false || pending.current) {
        return
      }

      const current = getUiState()

      if (!current.sid) {
        return
      }

      const visible = tabsInDirectory($sessionTabs.get(), sessionCwd())
      const active = findSessionTab(current.sid)

      const index = Math.max(
        0,
        visible.findIndex(tab => tab.id === active?.id)
      )

      const run = async () => {
        pending.current = true

        try {
          if (action.kind === 'new') {
            await session.newLiveSession()

            return
          }

          if (action.kind === 'close') {
            const target = action.id ? findSessionTab(action.id) : active

            if (!target || !visible.some(tab => tab.id === target.id)) {
              return
            }

            capture()

            if (hasTabDraft(findSessionTab(target.id)?.draft || emptyTabDraft())) {
              sys(t('pickers.session.tabs.unsentDraft'))

              return
            }

            if (target.id === active?.id) {
              const remaining = visible.filter(tab => tab.id !== target.id)
              const next = remaining[Math.min(index, remaining.length - 1)]

              if (next) {
                await session.activateLiveSession(next.id)
              } else {
                await session.newLiveSession()
              }

              // Do not remove a view if its replacement failed, or if the user
              // typed a new draft while the replacement was opening.
              if (getUiState().sid === target.id || hasTabDraft(findSessionTab(target.id)?.draft || emptyTabDraft())) {
                return
              }
            }

            removeSessionTab(target.id)

            return
          }

          const target =
            action.kind === 'select'
              ? visible.find(tab => tab.id === action.id)
              : action.kind === 'index'
                ? visible[action.index]
                : visible[(index + action.direction + visible.length) % visible.length]

          if (target && target.id !== active?.id) {
            await session.activateLiveSession(target.id)
          }
        } catch (error) {
          sys(`error: ${error instanceof Error ? error.message : String(error)}`)
        } finally {
          pending.current = false
        }
      }

      void run()
    },
    [capture, session, sys]
  )
}
