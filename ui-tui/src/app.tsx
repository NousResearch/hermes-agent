import { useStore } from '@nanostores/react'

import { GatewayProvider } from './app/gatewayContext.js'
import { $uiState } from './app/uiStore.js'
import { useMainApp } from './app/useMainApp.js'
import { AppLayout } from './components/appLayout.js'
import type { GatewayClient } from './gatewayClient.js'
import { I18nProvider } from './i18n/index.js'

export function App({ gw }: { gw: GatewayClient }) {
  const { appActions, appComposer, appProgress, appStatus, appTranscript, gateway } = useMainApp(gw)
  const { locale, mouseTracking } = useStore($uiState)

  return (
    <GatewayProvider value={gateway}>
      <I18nProvider locale={locale}>
        <AppLayout
          actions={appActions}
          composer={appComposer}
          mouseTracking={mouseTracking}
          progress={appProgress}
          status={appStatus}
          transcript={appTranscript}
        />
      </I18nProvider>
    </GatewayProvider>
  )
}
