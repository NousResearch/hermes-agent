import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router";
import "./index.css";
import App from "./App";
import { ProfileProvider } from "./contexts/ProfileProvider";
import { SystemActionsProvider } from "./contexts/SystemActions";
import { I18nProvider } from "./i18n";
import { exposePluginSDK } from "./plugins";
import { ThemeProvider } from "./themes";
import { HERMES_BASE_PATH } from "./lib/api";
import { startChunkLoadRecovery } from "./lib/chunk-load-recovery";

// Before the first render, so the first lazy route is covered too. The dev
// server has no hashed build to go stale.
if (import.meta.env.PROD) startChunkLoadRecovery();

// Expose the plugin SDK before rendering so plugins loaded via <script>
// can access React, components, etc. immediately.
exposePluginSDK();

createRoot(document.getElementById("root")!).render(
  <BrowserRouter basename={HERMES_BASE_PATH || undefined}>
    <I18nProvider>
      <ThemeProvider>
        <SystemActionsProvider>
          {/* Above App, so its plugin loader follows the selected profile. */}
          <ProfileProvider>
            <App />
          </ProfileProvider>
        </SystemActionsProvider>
      </ThemeProvider>
    </I18nProvider>
  </BrowserRouter>,
);
