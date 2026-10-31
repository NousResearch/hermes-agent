import { useTerminalTitle } from '@hermes/ink'
import { useStore } from '@nanostores/react'

import { composeTabTitle, shortCwd } from '../domain/paths.js'

import { $overlayState, hasSensitivePrompt } from './overlayStore.js'
import { $sessionTabs } from './sessionTabsStore.js'
import { $uiState } from './uiStore.js'
import { sessionTabActivity, useHerdrActivity } from './useHerdrActivity.js'

export function useSessionTabTerminalTitle() {
  const ui = useStore($uiState)
  const overlay = useStore($overlayState)
  const tabs = useStore($sessionTabs)
  const model = ui.info?.model?.replace(/^.*\//, '') ?? ''

  const activity = sessionTabActivity(
    tabs,
    ui.info?.cwd || process.env.HERMES_TUI_CWD || process.env.HERMES_CWD || '',
    ui.sid,
    ui.busy,
    Boolean(overlay.approval || overlay.clarify || hasSensitivePrompt(overlay)),
    ui.gatewayConnected !== false
  )

  useHerdrActivity()
  const marker = activity === 'blocked' ? '⚠' : activity === 'working' ? '⏳' : activity === 'unknown' ? '?' : '✓'
  const cwd = ui.info?.cwd

  useTerminalTitle(
    model
      ? {
          tab: composeTabTitle(marker, ui.sessionTitle, '', ''),
          window: composeTabTitle(marker, ui.sessionTitle, model, cwd ? shortCwd(cwd, 24) : '')
        }
      : `${marker} Hermes`
  )
}
