import { createRoot } from "react-dom/client";
import { BrowserRouter, useLocation } from "react-router-dom";
import "./index.css";
import App from "./App";
import CopilotPage from "./pages/CopilotPage";
import { SystemActionsProvider } from "./contexts/SystemActions";
import { I18nProvider } from "./i18n";
import { exposePluginSDK } from "./plugins";
import { ThemeProvider } from "./themes";
import { HERMES_BASE_PATH } from "./lib/api";

// Expose the plugin SDK before rendering so plugins loaded via <script>
// can access React, components, etc. immediately.
exposePluginSDK();

// `/copilot` renders the chat-bubble copilot CHROME-LESS — no dashboard
// sidebar/shell. It's what the desktop workflow view docks beside the
// langflow canvas in a narrow webview, so the full app chrome would just
// get in the way. Every other path renders the normal dashboard <App />.
function Root() {
  const { pathname } = useLocation();

  if (pathname.replace(/\/$/, "") === "/copilot") {
    return <CopilotPage />;
  }

  return <App />;
}

createRoot(document.getElementById("root")!).render(
  <BrowserRouter basename={HERMES_BASE_PATH || undefined}>
    <I18nProvider>
      <ThemeProvider>
        <SystemActionsProvider>
          <Root />
        </SystemActionsProvider>
      </ThemeProvider>
    </I18nProvider>
  </BrowserRouter>,
);
