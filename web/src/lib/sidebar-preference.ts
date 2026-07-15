export const SIDEBAR_OPEN_KEY = "hermes-sidebar-open";
/** @deprecated Replaced by SIDEBAR_OPEN_KEY — read once for migration. */
export const LEGACY_SIDEBAR_COLLAPSED_KEY = "hermes-sidebar-collapsed";

type StorageLike = Pick<Storage, "getItem" | "setItem" | "removeItem">;

/**
 * Read the persisted desktop sidebar preference from storage.
 * Defaults to open when no preference is stored (desktop-first layout).
 */
export function readDesktopSidebarOpenFromStorage(
  storage: StorageLike | null | undefined = localStorage,
): boolean {
  if (!storage) return true;
  try {
    const storedOpen = storage.getItem(SIDEBAR_OPEN_KEY);
    if (storedOpen !== null) return storedOpen !== "false";

    const legacyCollapsed = storage.getItem(LEGACY_SIDEBAR_COLLAPSED_KEY);
    if (legacyCollapsed !== null) {
      // Old "collapsed" was a narrow icon rail; the new model is open (w-64)
      // or fully closed — never a rail. Treat legacy collapsed=true as closed.
      const open = legacyCollapsed !== "true";
      storage.setItem(SIDEBAR_OPEN_KEY, open ? "true" : "false");
      storage.removeItem(LEGACY_SIDEBAR_COLLAPSED_KEY);
      return open;
    }
  } catch {
    /* storage may be unavailable in private browsing */
  }
  return true;
}

export function writeDesktopSidebarOpenToStorage(
  open: boolean,
  storage: StorageLike | null | undefined = localStorage,
): void {
  if (!storage) return;
  try {
    storage.setItem(SIDEBAR_OPEN_KEY, open ? "true" : "false");
  } catch {
    /* storage may be unavailable in private browsing */
  }
}
