declare global {
  interface Window {
    /**
     * Set true by the server only for `hermes dashboard --tui`
     * (or HERMES_DASHBOARD_TUI=1).
     */
    __HERMES_DASHBOARD_EMBEDDED_CHAT__?: boolean;
  }
}

/**
 * True only when the dashboard was started with embedded TUI Chat
 * (`hermes dashboard --tui`).  This is the server-side gate — even when
 * true, the Chat tab is only shown if the user has not disabled it via
 * the `dashboard.chat_ui` config flag.
 */
export function isDashboardEmbeddedChatEnabled(): boolean {
  return window.__HERMES_DASHBOARD_EMBEDDED_CHAT__ === true;
}
