import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router";
import "./index.css";
import App from "./App";
import { SystemActionsProvider } from "./contexts/SystemActions";
import { I18nProvider } from "./i18n";
import { exposePluginSDK } from "./plugins";
import { ThemeProvider } from "./themes";
import { HERMES_BASE_PATH } from "./lib/api";

// Expose the plugin SDK before rendering so plugins loaded via <script>
// can access React, components, etc. immediately.
exposePluginSDK();

// Register the PWA shell service worker. Best-effort: browsers without SW
// support (or a non-secure context — plain HTTP on a LAN/Tailscale IP)
// simply skip this; the SPA works identically either way, it just won't
// be installable as a home-screen app without HTTPS.
//
// Installed PWAs are long-lived: closing the app switcher / backgrounding
// it does NOT tear down the page the way closing a browser tab would, so
// a rebuilt+redeployed dashboard was previously invisible to an already-
// open instance until the user force-quit and relaunched it.
//
// PITFALL (found 2026-08-22): checking for updates on every
// `visibilitychange` seemed safe but isn't — picking files via the native
// photo picker (the paperclip button) backgrounds the page while the OS
// picker is shown, then fires `visibilitychange` back to "visible" the
// instant the picker returns control, WITH the file selection already in
// flight. An update reload firing at that exact moment silently destroys
// the in-progress upload before its `change` handler can run — this is
// why file uploads appeared to "do nothing". Fix: only check for updates
// on a plain timer, never on visibility changes, and never reload while
// hidden (queue it for the next time the page is genuinely idle+visible).
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker
      .register(`${HERMES_BASE_PATH}/sw.js`)
      .then((registration) => {
        // Installed PWAs may sit open for hours/days without ever
        // navigating, which is normally the only thing that triggers the
        // browser's own update check. Force one periodically — timer
        // only, deliberately NOT tied to visibilitychange (see pitfall
        // above).
        setInterval(() => {
          registration.update().catch(() => {
            // Offline or server unreachable — try again next tick.
          });
        }, 5 * 60_000);
      })
      .catch(() => {
        // Non-fatal — dashboard functions fully without the SW.
      });
  });

  // A new service worker taking control means a fresh build just
  // activated (see sw.js: skipWaiting + clients.claim). Reload so this
  // tab loads the new JS/CSS instead of continuing to run stale code
  // under the new SW — but never while the page is hidden (mid photo
  // picker, app backgrounded, etc.), and not immediately on becoming
  // visible either: the instant after a photo picker closes IS a
  // visibilitychange-to-visible event, with the file `change` handler
  // about to fire. Wait a few seconds of continuous visibility before
  // reloading so an in-flight upload always gets to start first.
  let updateReady = false;
  let visibleSinceTimer: ReturnType<typeof setTimeout> | null = null;
  const armVisibleReloadCheck = () => {
    if (visibleSinceTimer) clearTimeout(visibleSinceTimer);
    visibleSinceTimer = setTimeout(() => {
      if (updateReady && document.visibilityState === "visible") {
        window.location.reload();
      }
    }, 5_000);
  };
  navigator.serviceWorker.addEventListener("controllerchange", () => {
    if (updateReady) return;
    updateReady = true;
    if (document.visibilityState === "visible") armVisibleReloadCheck();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") armVisibleReloadCheck();
  });
}

createRoot(document.getElementById("root")!).render(
  <BrowserRouter basename={HERMES_BASE_PATH || undefined}>
    <I18nProvider>
      <ThemeProvider>
        <SystemActionsProvider>
          <App />
        </SystemActionsProvider>
      </ThemeProvider>
    </I18nProvider>
  </BrowserRouter>,
);
