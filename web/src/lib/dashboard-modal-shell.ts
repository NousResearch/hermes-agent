/**
 * Shared dashboard dialog shell classes.
 *
 * Page `<Card>` defaults to `bg-background-base/80` (glass). That looks fine
 * on the page canvas, but as a modal panel it lets the Models page bleed
 * through and kills readability — especially on Cyberpunk / mobile.
 *
 * Modal panels must use opaque `bg-card`; backdrops use the same z-index
 * band as Auxiliary / Confirm dialogs so they sit above page chrome.
 *
 * Callers must `createPortal(..., document.body)` — `z-[100]` alone cannot
 * escape the dashboard column's `relative z-2` stacking context (see
 * ModelPickerDialog / ToolsetConfigDrawer).
 */
export const DASHBOARD_MODAL_BACKDROP =
  "hermes-modal-backdrop fixed inset-0 z-[100] flex items-center justify-center bg-background/85 p-4";

export const DASHBOARD_MODAL_PANEL =
  "hermes-modal-panel-viewport relative w-full border border-border bg-card shadow-2xl";

interface DashboardDrawerA11yProps {
  "aria-hidden"?: boolean;
  "aria-modal"?: true;
  inert?: true;
  role?: "dialog";
}

/** Keep drawer semantics synchronized while allowing the desktop sidebar to remain navigation. */
export function dashboardDrawerA11yProps(
  open: boolean,
  enabled = true,
): DashboardDrawerA11yProps {
  if (!enabled) return {};
  return {
    "aria-hidden": !open,
    "aria-modal": open ? true : undefined,
    inert: open ? undefined : true,
    role: "dialog",
  };
}

/**
 * Outer modals that host a nested picker (e.g. MoA → ModelPickerDialog)
 * must ignore Escape while the picker is open; the picker owns that key.
 */
export function shouldCloseOuterModalOnEscape(
  nestedPickerOpen: boolean,
): boolean {
  return !nestedPickerOpen;
}
