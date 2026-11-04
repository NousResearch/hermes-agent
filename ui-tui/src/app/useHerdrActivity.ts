import { useEffect } from 'react'

import { type AgentActivity, createHerdrReporter } from '../lib/herdrAgentState.js'

import { $overlayState, hasSensitivePrompt } from './overlayStore.js'
import { $sessionTabs, type SessionTab, tabsInDirectory } from './sessionTabsStore.js'
import { $uiState, getUiState } from './uiStore.js'

export function sessionTabActivity(
  tabs: readonly SessionTab[],
  cwd: string,
  currentId: string | null,
  busy: boolean,
  blocked: boolean,
  connected = true
): AgentActivity {
  if (!connected) {
    return 'unknown'
  }

  const current = tabs.find(tab => [tab.id, tab.storedId, ...tab.aliases].includes(currentId || ''))
  const background = tabsInDirectory(tabs, cwd).filter(tab => tab !== current)

  if (blocked || background.some(tab => tab.blocked)) {
    return 'blocked'
  }

  if (busy || background.some(tab => tab.running)) {
    return 'working'
  }

  return 'idle'
}

const reporters = new Set<ReturnType<typeof createHerdrReporter>>()

export async function releaseHerdrReporters() {
  const active = [...reporters]
  reporters.clear()
  await Promise.all(active.map(reporter => reporter.close()))
}

export function startHerdrActivityBridge(env: NodeJS.ProcessEnv = process.env) {
  const instance = createHerdrReporter(env)
  reporters.add(instance)

  const report = () => {
    const ui = getUiState()
    const overlay = $overlayState.get()

    const state = sessionTabActivity(
      $sessionTabs.get(),
      ui.info?.cwd || env.HERMES_TUI_CWD || env.HERMES_CWD || '',
      ui.sid,
      ui.busy,
      Boolean(overlay.approval || overlay.clarify || hasSensitivePrompt(overlay)),
      ui.gatewayConnected !== false
    )

    instance.update({ state, sessionId: ui.storedSid || ui.sid || undefined })
  }

  // Subscribe outside React's paint/focus scheduling: a background tab's
  // owner-roster update must reach Herdr even when no terminal frame paints.
  const offTabs = $sessionTabs.listen(report)
  const offUi = $uiState.listen(report)
  const offOverlay = $overlayState.listen(report)
  report()
  const retry = setInterval(report, 15_000)
  retry.unref?.()

  return async () => {
    clearInterval(retry)
    offTabs()
    offUi()
    offOverlay()
    reporters.delete(instance)
    await instance.close()
  }
}

export function useHerdrActivity() {
  useEffect(() => {
    const stop = startHerdrActivityBridge()

    return () => {
      void stop()
    }
  }, [])
}
