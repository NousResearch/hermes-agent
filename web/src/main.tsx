import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router";
import "./index.css";
import { InsetDebugSentinel } from "./components/InsetDebugSentinel";
import App from "./App";
import { SystemActionsProvider } from "./contexts/SystemActions";
import { I18nProvider } from "./i18n";
import { exposePluginSDK } from "./plugins";
import { ThemeProvider } from "./themes";
import { HERMES_BASE_PATH } from "./lib/api";
import { trackMobileViewportInset } from "./lib/mobile-viewport-inset";

// Expose the plugin SDK before rendering so plugins loaded via <script>
// can access React, components, etc. immediately.
exposePluginSDK();

// NS-434 (mobile): Firefox Mobile (and iOS) keep the layout viewport put
// while the soft keyboard / bottom URL bar overlay the content, so the
// dashboard's mobile layout would render the prompt row underneath them.
// The tracker measures the obscured region via visualViewport and publishes
// it as `--hermes-mobile-bottom-inset`; index.css consumes that variable
// inside its ≤768px media block only — desktop layouts never read it.
// Gated on the mobile media query so no timer/listeners exist at all on
// desktop (byte-for-byte desktop behaviour preserved).
if (typeof window !== "undefined") {
  const mobileMql = window.matchMedia("(max-width: 768px)");
  let stopMobileInset = mobileMql.matches ? trackMobileViewportInset() : null;
  const onMobileMql = (e: MediaQueryListEvent) => {
    if (e.matches) {
      stopMobileInset = trackMobileViewportInset();
    } else if (stopMobileInset) {
      stopMobileInset();
      stopMobileInset = null;
    }
  };
  mobileMql.addEventListener?.("change", onMobileMql);
}

createRoot(document.getElementById("root")!).render(
  <BrowserRouter basename={HERMES_BASE_PATH || undefined}>
    {new URLSearchParams(window.location.search).get("debug") === "inset" ? (
      <InsetDebugSentinel />
    ) : null}
    <I18nProvider>
      <ThemeProvider>
        <SystemActionsProvider>
          <App />
        </SystemActionsProvider>
      </ThemeProvider>
    </I18nProvider>
  </BrowserRouter>,
);
