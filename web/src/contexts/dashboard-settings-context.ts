/**
 * Dashboard Settings Context — per-user visibility preferences for
 * Kanban columns, side-menu tabs, and sidebar item ordering.
 *
 * Persisted in localStorage under a single key.  All toggles default
 * to visible (true) so existing behaviour is unchanged for users who
 * never open the settings panel.
 */

import { createContext, useContext } from "react";

/* ── localStorage key ─────────────────────────────────────────────── */

const STORAGE_KEY = "hermes-dashboard-settings";

/* ── Types ────────────────────────────────────────────────────────── */

export interface KanbanColumnVisibility {
  triage: boolean;
  todo: boolean;
  scheduled: boolean;
  ready: boolean;
  running: boolean;
  blocked: boolean;
  review: boolean;
  done: boolean;
  archived: boolean;
}

export interface SideMenuTabVisibility {
  chat: boolean;
  sessions: boolean;
  analytics: boolean;
  models: boolean;
  logs: boolean;
  cron: boolean;
  skills: boolean;
  plugins: boolean;
  mcp: boolean;
  channels: boolean;
  webhooks: boolean;
  pairing: boolean;
  profiles: boolean;
  config: boolean;
  env: boolean;
  system: boolean;
  docs: boolean;
}

/** Entry in an ordered sidebar list. */
export interface SidebarOrderItem {
  id: string;
  label?: string;   // Custom display label (overrides default)
  icon?: string;    // Custom icon name (lucide-react icon name)
}

/**
 * Sidebar item ordering configuration.
 *
 * When `pluginsFoldedIntoSidebar` is false (default):
 *   - core items are ordered by `coreOrder`
 *   - plugin items are ordered by `pluginOrder`
 *
 * When `pluginsFoldedIntoSidebar` is true:
 *   - all items are ordered by `unifiedOrder`
 */
export interface SidebarItemOrder {
  pluginsFoldedIntoSidebar: boolean;
  coreOrder: SidebarOrderItem[];
  pluginOrder: SidebarOrderItem[];
  unifiedOrder: SidebarOrderItem[];
}

export interface DashboardSettings {
  kanbanColumns: KanbanColumnVisibility;
  sideMenuTabs: SideMenuTabVisibility;
  sidebarItemOrder: SidebarItemOrder;
}

export interface DashboardSettingsContextValue {
  settings: DashboardSettings;
  availablePlugins: { id: string; label: string }[];
  toggleKanbanColumn: (col: keyof KanbanColumnVisibility) => void;
  toggleSideMenuTab: (tab: keyof SideMenuTabVisibility) => void;
  isKanbanColumnVisible: (col: keyof KanbanColumnVisibility) => boolean;
  isSideMenuTabVisible: (tab: keyof SideMenuTabVisibility) => boolean;
  reorderSidebarItem: (
    group: "core" | "plugin" | "unified",
    fromIndex: number,
    toIndex: number,
  ) => void;
  setSidebarOrderAndFold: (
    newOrder: SidebarOrderItem[],
    group: "core" | "plugin" | "unified",
    folded: boolean,
  ) => void;
  setPluginsFoldedIntoSidebar: (folded: boolean) => void;
  isPluginsFolded: () => boolean;
  setSidebarItemLabel: (
    id: string,
    label: string,
    group: "core" | "plugin" | "unified",
  ) => void;
  setSidebarItemIcon: (
    id: string,
    icon: string,
    group: "core" | "plugin" | "unified",
  ) => void;
  getSidebarItemCustomization: (
    id: string,
  ) => { label?: string; icon?: string } | undefined;
}

/* ── Defaults ─────────────────────────────────────────────────────── */

export const DEFAULT_KANBAN_COLUMNS: KanbanColumnVisibility = {
  triage: true,
  todo: true,
  scheduled: true,
  ready: true,
  running: true,
  blocked: true,
  review: true,
  done: true,
  archived: true,
};

export const DEFAULT_SIDE_MENU_TABS: SideMenuTabVisibility = {
  chat: true,
  sessions: true,
  analytics: true,
  models: true,
  logs: true,
  cron: true,
  skills: true,
  plugins: true,
  mcp: true,
  channels: true,
  webhooks: true,
  pairing: true,
  profiles: true,
  config: true,
  env: true,
  system: true,
  docs: true,
};

export const DEFAULT_SIDEBAR_ITEM_ORDER: SidebarItemOrder = {
  pluginsFoldedIntoSidebar: false,
  coreOrder: [],
  pluginOrder: [],
  unifiedOrder: [],
};

export const DEFAULT_SETTINGS: DashboardSettings = {
  kanbanColumns: { ...DEFAULT_KANBAN_COLUMNS },
  sideMenuTabs: { ...DEFAULT_SIDE_MENU_TABS },
  sidebarItemOrder: { ...DEFAULT_SIDEBAR_ITEM_ORDER },
};

/* ── Storage helpers ──────────────────────────────────────────────── */

/**
 * Load settings from server (config.yaml via /api/dashboard/settings).
 * Falls back to localStorage if the server call fails, then to defaults.
 */
export async function loadSettingsAsync(): Promise<DashboardSettings> {
  try {
    const { api } = await import("@/lib/api");
    const { settings } = await api.getDashboardSettings();
    if (settings && Object.keys(settings).length > 0) {
      return {
        kanbanColumns: {
          ...DEFAULT_KANBAN_COLUMNS,
          ...(settings.kanbanColumns ?? {}),
        },
        sideMenuTabs: {
          ...DEFAULT_SIDE_MENU_TABS,
          ...(settings.sideMenuTabs ?? {}),
        },
        sidebarItemOrder: {
          ...DEFAULT_SIDEBAR_ITEM_ORDER,
          ...(settings.sidebarItemOrder ?? {}),
        },
      };
    }
  } catch {
    /* server unavailable — fall through to localStorage */
  }
  // Fallback: localStorage
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<DashboardSettings>;
      return {
        kanbanColumns: {
          ...DEFAULT_KANBAN_COLUMNS,
          ...(parsed.kanbanColumns ?? {}),
        },
        sideMenuTabs: {
          ...DEFAULT_SIDE_MENU_TABS,
          ...(parsed.sideMenuTabs ?? {}),
        },
        sidebarItemOrder: {
          ...DEFAULT_SIDEBAR_ITEM_ORDER,
          ...(parsed.sidebarItemOrder ?? {}),
        },
      };
    }
  } catch {
    /* ignore */
  }
  return DEFAULT_SETTINGS;
}

/** Synchronous load — used only for initial React state hydration. */
export function loadSettings(): DashboardSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_SETTINGS;
    const parsed = JSON.parse(raw) as Partial<DashboardSettings>;
    return {
      kanbanColumns: {
        ...DEFAULT_KANBAN_COLUMNS,
        ...(parsed.kanbanColumns ?? {}),
      },
      sideMenuTabs: {
        ...DEFAULT_SIDE_MENU_TABS,
        ...(parsed.sideMenuTabs ?? {}),
      },
      sidebarItemOrder: {
        ...DEFAULT_SIDEBAR_ITEM_ORDER,
        ...(parsed.sidebarItemOrder ?? {}),
      },
    };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

/**
 * Persist settings to both server (config.yaml) and localStorage (fallback).
 * Fire-and-forget: errors are silently caught so the UI never blocks.
 */
export function saveSettings(settings: DashboardSettings): void {
  // Always write to localStorage for immediate offline access
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    /* localStorage may be unavailable in private browsing */
  }
  // Also persist to server — fire and forget
  import("@/lib/api")
    .then(({ api }) => api.saveDashboardSettings(settings))
    .catch(() => {
      /* server unavailable — localStorage copy is still valid */
    });
}

/* ── Order application helpers (exported for use in App.tsx) ──────── */

/**
 * Given a list of current item IDs and a saved order, return the IDs
 * reordered according to saved preferences.  Items not present in the
 * saved order are appended at the end in their natural order.
 */
export function applySavedOrder(
  currentIds: string[],
  savedOrder: SidebarOrderItem[],
): SidebarOrderItem[] {
  if (!savedOrder || savedOrder.length === 0)
    return currentIds.map((id) => ({ id }));
  const savedMap = new Map<string, number>();
  savedOrder.forEach((item, idx) => savedMap.set(item.id, idx));

  const inBoth = currentIds.filter((id) => savedMap.has(id));
  const newItems = currentIds.filter((id) => !savedMap.has(id));

  inBoth.sort((a, b) => savedMap.get(a)! - savedMap.get(b)!);

  return [...inBoth, ...newItems].map((id) => ({ id }));
}

/**
 * Apply a saved ordering to a list of nav items.  Items not in the saved
 * order retain their relative position at the end.
 */
export function applySidebarOrder<T extends { id?: string; path: string }>(
  items: T[],
  savedOrder: SidebarOrderItem[],
): T[] {
  if (!savedOrder || savedOrder.length === 0) return items;
  const orderMap = new Map<string, number>();
  savedOrder.forEach((item, idx) => orderMap.set(item.id, idx));

  const sorted = [...items];
  sorted.sort((a, b) => {
    // Match by id, or by path with/without leading slash
    const aKey = a.id || a.path.replace(/^\//, "");
    const bKey = b.id || b.path.replace(/^\//, "");
    const aIdx = orderMap.get(aKey) ?? orderMap.get(a.path);
    const bIdx = orderMap.get(bKey) ?? orderMap.get(b.path);
    if (aIdx === undefined && bIdx === undefined) return 0;
    if (aIdx === undefined) return 1;
    if (bIdx === undefined) return -1;
    return aIdx - bIdx;
  });
  return sorted;
}

/* ── Context ──────────────────────────────────────────────────────── */

export const DashboardSettingsContext =
  createContext<DashboardSettingsContextValue | null>(null);

export function useDashboardSettings(): DashboardSettingsContextValue {
  const ctx = useContext(DashboardSettingsContext);
  if (!ctx) {
    throw new Error(
      "useDashboardSettings must be used within a DashboardSettingsProvider",
    );
  }
  return ctx;
}
