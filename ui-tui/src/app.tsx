import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { GatewayProvider } from './app/gatewayContext.js'
import { $skinReady, $uiState, markSkinReady } from './app/uiStore.js'
import { useMainApp } from './app/useMainApp.js'
import { AppLayout } from './components/appLayout.js'
import type { GatewayClient } from './gatewayClient.js'

// Hard ceiling on how long the first paint is held waiting for the gateway's
// skin. The skin normally arrives within a few ms (entry.py emits skin.changed
// before MCP discovery), but a wedged/old gateway must not strand the UI on a
// blank screen — fall back to painting in DEFAULT_THEME after this.
const SKIN_GATE_TIMEOUT_MS = 1500

export function App({ gw }: { gw: GatewayClient }) {
  const { appActions, appComposer, appProgress, appStatus, appTranscript, gateway } = useMainApp(gw)
  const { mouseTracking } = useStore($uiState)
  const skinReady = useStore($skinReady)

  useEffect(() => {
    const timer = setTimeout(markSkinReady, SKIN_GATE_TIMEOUT_MS)

    return () => clearTimeout(timer)
  }, [])

  // Hold the first paint until the skin lands so we render once in the user's
  // theme. The screen was already cleared in entry.tsx, so this is just blank,
  // not a flash. useMainApp above still runs, so gateway events (incl. the skin)
  // keep flowing while we wait.
  if (!skinReady) {
    return null
  }

  return (
    <GatewayProvider value={gateway}>
      <AppLayout
        actions={appActions}
        composer={appComposer}
        mouseTracking={mouseTracking}
        progress={appProgress}
        status={appStatus}
        transcript={appTranscript}
      />
    </GatewayProvider>
  )
}
