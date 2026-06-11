/**
 * DashboardSettingsProvider — wraps the app and supplies visibility
 * settings via context.  Children read/write toggles through
 * useDashboardSettings().
 */

import { useState, useCallback, useMemo, useEffect, useRef } from "react";
import type { ReactNode } from "react";
import {
  DashboardSettingsContext,
  DEFAULT_SETTINGS,
  loadSettings,
  loadSettingsAsync,
  saveSettings,
} from "./dashboard-settings-context";
import type {
  KanbanColumnVisibility,
  SideMenuTabVisibility,
  SidebarOrderItem,
} from "./dashboard-settings-context";

export function DashboardSettingsProvider({
  children,
  availablePlugins = [],
}: {
  children: ReactNode;
  availablePlugins?: { id: string; label: string }[];
}) {
  const [settings, setSettings] = useState(loadSettings);

  // On mount, fetch authoritative settings from the server (config.yaml).
  // Only apply if the server has data that differs from what localStorage gave us.
  // Use a ref to prevent overwriting user changes made before this resolves.
  const serverSynced = useRef(false);
  useEffect(() => {
    loadSettingsAsync().then((serverSettings) => {
      serverSynced.current = true;
      setSettings((prev) => {
        const serverStr = JSON.stringify(serverSettings);
        const prevStr = JSON.stringify(prev);
        // Only overwrite if server has strictly MORE data than prev (e.g. server
        // has saved orders while prev is still default/empty).
        if (serverStr !== prevStr && serverStr !== JSON.stringify(DEFAULT_SETTINGS)) {
          return serverSettings;
        }
        return prev;
      });
    });
  }, []);

  const toggleKanbanColumn = useCallback(
    (col: keyof KanbanColumnVisibility) => {
      setSettings((prev) => {
        const next = {
          ...prev,
          kanbanColumns: {
            ...prev.kanbanColumns,
            [col]: !prev.kanbanColumns[col],
          },
        };
        saveSettings(next);
        return next;
      });
    },
    [],
  );

  const toggleSideMenuTab = useCallback(
    (tab: keyof SideMenuTabVisibility) => {
      setSettings((prev) => {
        const next = {
          ...prev,
          sideMenuTabs: {
            ...prev.sideMenuTabs,
            [tab]: !prev.sideMenuTabs[tab],
          },
        };
        saveSettings(next);
        return next;
      });
    },
    [],
  );

  const isKanbanColumnVisible = useCallback(
    (col: keyof KanbanColumnVisibility) => settings.kanbanColumns[col],
    [settings],
  );

  const isSideMenuTabVisible = useCallback(
    (tab: keyof SideMenuTabVisibility) => settings.sideMenuTabs[tab],
    [settings],
  );

  const reorderSidebarItem = useCallback(
    (
      group: "core" | "plugin" | "unified",
      fromIndex: number,
      toIndex: number,
    ) => {
      setSettings((prev) => {
        const key =
          group === "core"
            ? "coreOrder"
            : group === "plugin"
              ? "pluginOrder"
              : "unifiedOrder";
        const prevOrder = [...prev.sidebarItemOrder[key]];
        if (
          fromIndex < 0 ||
          fromIndex >= prevOrder.length ||
          toIndex < 0 ||
          toIndex > prevOrder.length
        ) {
          return prev;
        }
        const [moved] = prevOrder.splice(fromIndex, 1);
        prevOrder.splice(toIndex, 0, moved);
        const next = {
          ...prev,
          sidebarItemOrder: {
            ...prev.sidebarItemOrder,
            [key]: prevOrder,
          },
        };
        saveSettings(next);
        return next;
      });
    },
    [],
  );

  const setSidebarOrderAndFold = useCallback(
    (
      newOrder: SidebarOrderItem[],
      group: "core" | "plugin" | "unified",
      folded: boolean,
    ) => {
      setSettings((prev) => {
        const key =
          group === "core"
            ? "coreOrder"
            : group === "plugin"
              ? "pluginOrder"
              : "unifiedOrder";
        const next = {
          ...prev,
          sidebarItemOrder: {
            ...prev.sidebarItemOrder,
            pluginsFoldedIntoSidebar: folded,
            [key]: newOrder,
          },
        };
        saveSettings(next);
        return next;
      });
    },
    [],
  );

  const setPluginsFoldedIntoSidebar = useCallback(
    (folded: boolean) => {
      setSettings((prev) => {
        const next = {
          ...prev,
          sidebarItemOrder: {
            ...prev.sidebarItemOrder,
            pluginsFoldedIntoSidebar: folded,
          },
        };
        saveSettings(next);
        return next;
      });
    },
    [],
  );

  const isPluginsFolded = useCallback(
    () => settings.sidebarItemOrder.pluginsFoldedIntoSidebar,
    [settings.sidebarItemOrder.pluginsFoldedIntoSidebar],
  );

  const value = useMemo(
    () => ({
      settings,
      availablePlugins,
      toggleKanbanColumn,
      toggleSideMenuTab,
      isKanbanColumnVisible,
      isSideMenuTabVisible,
      reorderSidebarItem,
      setSidebarOrderAndFold,
      setPluginsFoldedIntoSidebar,
      isPluginsFolded,
    }),
    [
      settings,
      availablePlugins,
      toggleKanbanColumn,
      toggleSideMenuTab,
      isKanbanColumnVisible,
      isSideMenuTabVisible,
      reorderSidebarItem,
      setSidebarOrderAndFold,
      setPluginsFoldedIntoSidebar,
      isPluginsFolded,
    ],
  );

  return (
    <DashboardSettingsContext.Provider value={value}>
      {children}
    </DashboardSettingsContext.Provider>
  );
}
