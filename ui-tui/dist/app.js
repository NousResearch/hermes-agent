import { jsx as _jsx } from "react/jsx-runtime";
import { GatewayProvider } from './app/gatewayContext.js';
import { useMainApp } from './app/useMainApp.js';
import { AppLayout } from './components/appLayout.js';
import { MOUSE_TRACKING } from './config/env.js';
export function App({ gw }) {
    const { appActions, appComposer, appProgress, appStatus, appTranscript, gateway } = useMainApp(gw);
    return (_jsx(GatewayProvider, { value: gateway, children: _jsx(AppLayout, { actions: appActions, composer: appComposer, mouseTracking: MOUSE_TRACKING, progress: appProgress, status: appStatus, transcript: appTranscript }) }));
}
