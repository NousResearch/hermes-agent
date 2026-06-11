/**
 * DashboardSettingsPage — per-user visibility toggles for Kanban columns,
 * side-menu tabs, and sidebar item ordering.
 *
 * Persisted in localStorage.  All toggles default to visible.
 */

import { useEffect, useMemo, useCallback, useState } from "react";
import { LayoutDashboard, PanelLeft, FoldVertical, RefreshCw } from "lucide-react";
import { useI18n } from "@/i18n";
import type { Translations } from "@/i18n/types";
import { useDashboardSettings } from "@/contexts/dashboard-settings-context";
import { usePlugins } from "@/plugins";
import { api } from "@/lib/api";
import type {
  KanbanColumnVisibility,
  SideMenuTabVisibility,
  SidebarOrderItem,
} from "@/contexts/dashboard-settings-context";
import { usePageHeader } from "@/contexts/usePageHeader";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { Typography } from "@nous-research/ui/ui/components/typography/index";
import { cn } from "@/lib/utils";
import type { ComponentType } from "react";
import { SidebarReorder } from "@/components/SidebarReorder";

/* ── Toggle row component ─────────────────────────────────────────── */

function ToggleRow({
  label,
  description,
  checked,
  onToggle,
}: {
  label: string;
  description?: string;
  checked: boolean;
  onToggle: () => void;
}) {
  return (
    <label
      className={cn(
        "flex cursor-pointer items-center justify-between gap-4",
        "px-4 py-3 transition-colors hover:bg-secondary/30",
        "border-b border-border last:border-b-0",
      )}
    >
      <div className="flex min-w-0 flex-col gap-0.5">
        <span className="text-sm font-medium">{label}</span>
        {description && (
          <span className="text-xs text-text-tertiary">{description}</span>
        )}
      </div>
      <div
        role="switch"
        aria-checked={checked}
        tabIndex={0}
        onClick={onToggle}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onToggle();
          }
        }}
        className={cn(
          "relative inline-flex h-6 w-11 shrink-0 cursor-pointer items-center",
          "rounded-full border border-current/20 transition-colors",
          checked ? "bg-midground/30" : "bg-transparent",
        )}
      >
        <span
          className={cn(
            "inline-block h-4 w-4 rounded-full transition-transform",
            "bg-midground shadow-sm",
            checked ? "translate-x-[1.375rem]" : "translate-x-1",
          )}
        />
      </div>
    </label>
  );
}

/* ── Section component ────────────────────────────────────────────── */

function SettingsSection({
  title,
  description,
  icon: Icon,
  children,
}: {
  title: string;
  description?: string;
  icon: ComponentType<{ className?: string }>;
  children: React.ReactNode;
}) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center gap-2">
          <Icon className="h-4 w-4 text-text-tertiary" />
          <CardTitle className="text-base">{title}</CardTitle>
        </div>
        {description && (
          <Typography className="text-xs text-text-tertiary">
            {description}
          </Typography>
        )}
      </CardHeader>
      <CardContent className="p-0">
        <div className="flex flex-col">{children}</div>
      </CardContent>
    </Card>
  );
}

/* ── Kanban column config ─────────────────────────────────────────── */

const KANBAN_COLUMN_KEYS: Array<{
  key: keyof KanbanColumnVisibility;
  labelKey: keyof Translations["kanban"]["columnLabels"];
  helpKey: keyof Translations["kanban"]["columnHelp"];
}> = [
  { key: "triage", labelKey: "triage", helpKey: "triage" },
  { key: "todo", labelKey: "todo", helpKey: "todo" },
  { key: "scheduled", labelKey: "scheduled", helpKey: "scheduled" },
  { key: "ready", labelKey: "ready", helpKey: "ready" },
  { key: "running", labelKey: "running", helpKey: "running" },
  { key: "blocked", labelKey: "blocked", helpKey: "blocked" },
  { key: "review", labelKey: "review", helpKey: "review" },
  { key: "done", labelKey: "done", helpKey: "done" },
  { key: "archived", labelKey: "archived", helpKey: "archived" },
];

/* ── Side-menu tab config ─────────────────────────────────────────── */

const TAB_CONFIG: Array<{
  key: keyof SideMenuTabVisibility;
}> = [
  { key: "chat" },
  { key: "sessions" },
  { key: "analytics" },
  { key: "models" },
  { key: "logs" },
  { key: "cron" },
  { key: "skills" },
  { key: "plugins" },
  { key: "channels" },
  { key: "webhooks" },
  { key: "pairing" },
  { key: "profiles" },
  { key: "config" },
  { key: "env" },
  { key: "system" },
  { key: "docs" },
];

/* ── Page ─────────────────────────────────────────────────────────── */

export default function DashboardSettingsPage() {
  const { t } = useI18n();
  const { setTitle } = usePageHeader();
  const { manifests, loading: pluginsLoading } = usePlugins();
  const {
    toggleKanbanColumn,
    toggleSideMenuTab,
    isKanbanColumnVisible,
    isSideMenuTabVisible,
    setSidebarOrderAndFold,
    setPluginsFoldedIntoSidebar,
    setSidebarItemLabel,
    setSidebarItemIcon,
    settings,
  } = useDashboardSettings();

  const [scanning, setScanning] = useState(false);

  // Set page header
  useEffect(() => {
    setTitle(t.dashboardSettings.title);
  }, [setTitle, t.dashboardSettings.title]);

  // Build settings-dependent values via useMemo
  const kanbanSectionTitle = t.dashboardSettings.kanbanColumns.title;
  const kanbanSectionDesc = t.dashboardSettings.kanbanColumns.description;
  const sideMenuSectionTitle = t.dashboardSettings.sideMenuTabs.title;
  const sideMenuSectionDesc = t.dashboardSettings.sideMenuTabs.description;

  const navLabels: Record<string, string> = useMemo(
    () => ({
      chat: t.app.nav.chat,
      sessions: t.app.nav.sessions,
      analytics: t.app.nav.analytics,
      models: t.app.nav.models,
      logs: t.app.nav.logs,
      cron: t.app.nav.cron,
      skills: t.app.nav.skills,
      plugins: t.app.nav.plugins,
      mcp: "MCP",
      channels: t.app.nav.channels,
      webhooks: t.app.nav.webhooks,
      pairing: t.app.nav.pairing,
      profiles: t.app.nav.profiles,
      config: t.app.nav.config,
      env: t.app.nav.keys,
      system: t.app.nav.system,
      docs: t.app.nav.documentation,
      // Plugin manifest labels (by id derived from tab.path)
      ...Object.fromEntries(
        manifests.map((m) => [
          m.tab.path.replace(/^\//, ""),
          m.label,
        ]),
      ),
    }),
    [t.app.nav, manifests],
  );

  // Build reorder items from saved order, falling back to defaults when empty
  const savedCoreOrder = settings.sidebarItemOrder.coreOrder;
  const savedPluginOrder = settings.sidebarItemOrder.pluginOrder;
  const savedUnifiedOrder = settings.sidebarItemOrder.unifiedOrder;
  const folded = settings.sidebarItemOrder.pluginsFoldedIntoSidebar;

  const defaultCoreItems = useMemo(() => {
    return TAB_CONFIG.filter((t) => isSideMenuTabVisible(t.key)).map((t) => ({
      id: t.key,
      label: navLabels[t.key] ?? t.key,
    }));
  }, [navLabels, isSideMenuTabVisible]);

  const coreItems = useMemo(() => {
    // When folded, the unified list owns the order — skip core-specific logic
    if (folded) return [];
    const visible = isSideMenuTabVisible;
    if (savedCoreOrder.length > 0) {
      return savedCoreOrder
        .filter((o) => visible(o.id as keyof SideMenuTabVisibility))
        .map((o) => ({
          id: o.id,
          label: navLabels[o.id] ?? o.id,
        }));
    }
    return defaultCoreItems;
  }, [folded, savedCoreOrder, navLabels, defaultCoreItems, isSideMenuTabVisible]);

  const pluginItems = useMemo(() => {
    // When folded, the unified list owns the order — skip plugin-specific logic
    if (folded) return [];
    if (savedPluginOrder.length > 0) {
      return savedPluginOrder.map((o) => ({
        id: o.id,
        label: navLabels[o.id] ?? o.id,
      }));
    }
    // Populate from actual plugins when no saved order exists
    return manifests
      .filter((m) => !m.tab.hidden && !m.tab.override)
      .map((m) => ({
        id: m.tab.path.replace(/^\//, ""),
        label: m.label,
      }));
  }, [folded, savedPluginOrder, navLabels, manifests]);

  const unifiedItems = useMemo(() => {
    if (!folded) return [];
    if (savedUnifiedOrder.length > 0) {
      return savedUnifiedOrder.map((o) => ({
        id: o.id,
        label: navLabels[o.id] ?? o.id,
      }));
    }
    // No saved unified order — merge core + plugin defaults
    return [
      ...defaultCoreItems,
      ...manifests
        .filter((m) => !m.tab.hidden && !m.tab.override)
        .map((m) => ({
          id: m.tab.path.replace(/^\//, ""),
          label: m.label,
        })),
    ];
  }, [folded, savedUnifiedOrder, navLabels, defaultCoreItems, manifests]);

  const handleCoreReorder = useCallback(
    (items: { id: string; label: string }[]) => {
      const order: SidebarOrderItem[] = items.map((i) => ({ id: i.id }));
      setSidebarOrderAndFold(order, "core", false);
    },
    [setSidebarOrderAndFold],
  );

  const handlePluginReorder = useCallback(
    (items: { id: string; label: string }[]) => {
      const order: SidebarOrderItem[] = items.map((i) => ({ id: i.id }));
      setSidebarOrderAndFold(order, "plugin", false);
    },
    [setSidebarOrderAndFold],
  );

  const handleUnifiedReorder = useCallback(
    (items: { id: string; label: string }[]) => {
      const order: SidebarOrderItem[] = items.map((i) => ({ id: i.id }));
      setSidebarOrderAndFold(order, "unified", true);
    },
    [setSidebarOrderAndFold],
  );

  const handleFoldToggle = useCallback(
    (folded: boolean) => {
      setPluginsFoldedIntoSidebar(folded);
      if (folded) {
        const merged = [
          ...coreItems.map((i) => ({ id: i.id })),
          ...pluginItems.map((i) => ({ id: i.id })),
        ];
        setSidebarOrderAndFold(merged, "unified", true);
      } else {
        const coreSaved = settings.sidebarItemOrder.coreOrder;
        const pluginSaved = settings.sidebarItemOrder.pluginOrder;
        setSidebarOrderAndFold(coreSaved, "core", false);
        setSidebarOrderAndFold(pluginSaved, "plugin", false);
      }
    },
    [setPluginsFoldedIntoSidebar, setSidebarOrderAndFold, coreItems, pluginItems, settings.sidebarItemOrder],
  );

  const handleRescanPlugins = useCallback(async () => {
    setScanning(true);
    try {
      await api.rescanPlugins();
      // Force a page reload to pick up new manifests
      window.location.reload();
    } catch {
      setScanning(false);
    }
  }, []);

  // Build customizations map for all sidebar items
  const customizationsMap = useMemo(() => {
    const map = new Map<string, { label?: string; icon?: string }>();
    const allItems = [
      ...settings.sidebarItemOrder.coreOrder,
      ...settings.sidebarItemOrder.pluginOrder,
      ...settings.sidebarItemOrder.unifiedOrder,
    ];
    for (const item of allItems) {
      if (item.label || item.icon) {
        map.set(item.id, { label: item.label, icon: item.icon });
      }
    }
    return map;
  }, [settings.sidebarItemOrder]);

  const handleLabelChange = useCallback(
    (id: string, label: string, group: "core" | "plugin" | "unified") => {
      setSidebarItemLabel(id, label, group);
    },
    [setSidebarItemLabel],
  );

  const handleIconChange = useCallback(
    (id: string, icon: string, group: "core" | "plugin" | "unified") => {
      setSidebarItemIcon(id, icon, group);
    },
    [setSidebarItemIcon],
  );

  return (
    <div className="mx-auto grid max-w-5xl grid-cols-1 gap-6 p-4 md:grid-cols-2">
      <SettingsSection
        title={kanbanSectionTitle}
        description={kanbanSectionDesc}
        icon={LayoutDashboard}
      >
        {KANBAN_COLUMN_KEYS.map(({ key, labelKey, helpKey }) => (
          <ToggleRow
            key={key}
            label={t.kanban.columnLabels[labelKey]}
            description={t.kanban.columnHelp[helpKey]}
            checked={isKanbanColumnVisible(key)}
            onToggle={() => toggleKanbanColumn(key)}
          />
        ))}
      </SettingsSection>

      <SettingsSection
        title={sideMenuSectionTitle}
        description={sideMenuSectionDesc}
        icon={PanelLeft}
      >
        <ToggleRow
          label="MCP"
          checked={isSideMenuTabVisible("mcp")}
          onToggle={() => toggleSideMenuTab("mcp")}
        />
        {TAB_CONFIG.map(({ key }) => (
          <ToggleRow
            key={key}
            label={navLabels[key]}
            checked={isSideMenuTabVisible(key)}
            onToggle={() => toggleSideMenuTab(key)}
          />
        ))}
      </SettingsSection>

      <div className="md:col-span-2">
        <SettingsSection
          title={t.dashboardSettings.sidebarOrder?.title ?? "Sidebar Order"}
          description={t.dashboardSettings.sidebarOrder?.description ?? "Drag and drop to reorder sidebar navigation items."}
          icon={FoldVertical}
        >
          <div className="flex items-center justify-between border-b border-border px-4 py-2">
            <span className="text-xs text-text-tertiary">
              {manifests.length} plugin{manifests.length !== 1 ? "s" : ""} detected
            </span>
            <button
              onClick={handleRescanPlugins}
              disabled={scanning || pluginsLoading}
              className={cn(
                "flex items-center gap-1.5 rounded-md px-2 py-1 text-xs",
                "transition-colors hover:bg-secondary/30",
                "disabled:opacity-50 disabled:cursor-not-allowed",
              )}
            >
              <RefreshCw className={cn("h-3 w-3", scanning && "animate-spin")} />
              {scanning ? "Scanning…" : "Scan for plugins"}
            </button>
          </div>
          <SidebarReorder
            coreItems={coreItems}
            pluginItems={pluginItems}
            unifiedItems={unifiedItems}
            folded={folded}
            onCoreReorder={handleCoreReorder}
            onPluginReorder={handlePluginReorder}
            onUnifiedReorder={handleUnifiedReorder}
            onFoldToggle={handleFoldToggle}
            mainItemsLabel={t.dashboardSettings.sidebarOrder?.mainItems ?? "Main Items"}
            pluginItemsLabel={t.dashboardSettings.sidebarOrder?.pluginItems ?? "Plugin Items"}
            unifiedItemsLabel={t.dashboardSettings.sidebarOrder?.unifiedItems ?? "All Items"}
            customizations={customizationsMap}
            onLabelChange={handleLabelChange}
            onIconChange={handleIconChange}
          />
        </SettingsSection>
      </div>
    </div>
  );
}
