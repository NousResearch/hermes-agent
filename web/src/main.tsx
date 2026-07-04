import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import "./index.css";
import App from "./App";
import MiniApp from "./miniapp/MiniApp";
import { SystemActionsProvider } from "./contexts/SystemActions";
import { I18nProvider } from "./i18n";
import { exposePluginSDK } from "./plugins";
import { ThemeProvider } from "./themes";
import { HERMES_BASE_PATH } from "./lib/api";

// Expose the plugin SDK before rendering so plugins loaded via <script>
// can access React, components, etc. immediately.
exposePluginSDK();

// The Telegram Mini App is a deliberately separate root, not a route nested
// inside <App>: <App> unconditionally renders the full desktop shell
// (sidebar, ThemeProvider-driven CSS vars, i18n chrome) around every one of
// its <Routes>, none of which belongs in a phone-sized Telegram WebView --
// and the Mini App's own 5-palette system (miniapp/palettes.css) is
// deliberately independent of the desktop ThemeProvider's CSS variables, so
// mounting it inside that provider risks the two token sets colliding.
// Branching here, before any of that mounts, keeps the two surfaces fully
// isolated. hermes_cli/web_server.py's mount_spa() SPA-fallback catch-all
// already serves this same index.html/bundle for /miniapp with no backend
// changes needed -- this client-side check is the only routing decision.
const isMiniApp = window.location.pathname.replace(HERMES_BASE_PATH || "", "").startsWith("/miniapp");

createRoot(document.getElementById("root")!).render(
  isMiniApp ? (
    <MiniApp />
  ) : (
    <BrowserRouter basename={HERMES_BASE_PATH || undefined}>
      <I18nProvider>
        <ThemeProvider>
          <SystemActionsProvider>
            <App />
          </SystemActionsProvider>
        </ThemeProvider>
      </I18nProvider>
    </BrowserRouter>
  ),
);
