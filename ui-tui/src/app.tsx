import { useStore } from '@nanostores/react'

import { GatewayProvider } from './app/gatewayContext.js'
import { $uiState } from './app/uiStore.js'
import { useMainApp } from './app/useMainApp.js'
import { AppLayout } from './components/appLayout.js'
import type { GatewayClient } from './gatewayClient.js'
import { I18nProvider } from './i18n/index.js'

export function App({ gw }: { gw: GatewayClient }) {
  const { locale } = useStore($uiState)

  return (
    <I18nProvider locale={locale}>
      <AppContent gw={gw} />
    </I18nProvider>
  )
}

/**
 * Keep application hooks below the locale provider. Several hooks emit
 * user-facing messages directly, so mounting the provider only around
 * AppLayout would silently pin those paths to the default English context.
 */
function AppContent({ gw }: { gw: GatewayClient }) {
  const { appActions, appComposer, appProgress, appStatus, appTranscript, gateway } = useMainApp(gw)
  const { mouseTracking } = useStore($uiState)

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
