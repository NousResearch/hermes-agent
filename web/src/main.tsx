import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import "./index.css";
import App from "./App";
import LandingPage from "./pages/LandingPage";
import PortalPage from "./pages/PortalPage";
import { SystemActionsProvider } from "./contexts/SystemActions";
import { I18nProvider } from "./i18n";
import { exposePluginSDK } from "./plugins";
import { ThemeProvider } from "./themes";
import { HERMES_BASE_PATH } from "./lib/api";

// Expose the plugin SDK before rendering so plugins loaded via <script>
// can access React, components, etc. immediately.
exposePluginSDK();

createRoot(document.getElementById("root")!).render(
  <BrowserRouter basename={HERMES_BASE_PATH || undefined}>
    <I18nProvider>
      <ThemeProvider>
        <SystemActionsProvider>
          <Routes>
            {/* Public pages — no admin auth, no sidebar */}
            <Route path="/subscribe" element={<LandingPage />} />
            <Route path="/portal" element={<PortalPage />} />
            {/* Everything else → authenticated admin dashboard */}
            <Route path="/*" element={<App />} />
          </Routes>
        </SystemActionsProvider>
      </ThemeProvider>
    </I18nProvider>
  </BrowserRouter>,
);
