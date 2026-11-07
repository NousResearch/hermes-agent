/**
 * Collapse state for the desktop chat side panel (model + sessions),
 * persisted in localStorage so the choice survives reloads.
 *
 * Pure helpers so ChatPage stays under its FILE_LINES cap and the
 * persistence stays testable without React.
 */

const CHAT_PANEL_COLLAPSED_KEY = "hermes-chat-panel-collapsed";

/** The panel's collapsed flag remembered from the previous visit. */
export function readChatPanelCollapsed(): boolean {
  return localStorage.getItem(CHAT_PANEL_COLLAPSED_KEY) === "1";
}

/** Flip the collapsed flag and persist the new choice. */
export function toggleChatPanelCollapsed(previous: boolean): boolean {
  const next = !previous;
  localStorage.setItem(CHAT_PANEL_COLLAPSED_KEY, next ? "1" : "0");
  return next;
}
