import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ComponentType,
  type ReactNode,
} from "react";
import {
  Routes,
  Route,
  NavLink,
  Navigate,
  useLocation,
  useNavigate,
} from "react-router-dom";
import {
  Activity,
  BarChart3,
  BookOpen,
  Clock,
  Code,
  Database,
  Download,
  Eye,
  FileText,
  Globe,
  Heart,
  KeyRound,
  Loader2,
  Menu,
  MessageSquare,
  Package,
  PanelLeftClose,
  PanelLeftOpen,
  Puzzle,
  RotateCw,
  Settings,
  Shield,
  Sparkles,
  Star,
  Terminal,
  Wrench,
  X,
  Zap,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { TooltipProvider } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { SidebarFooter } from "@/components/SidebarFooter";
import { SidebarStatusStrip } from "@/components/SidebarStatusStrip";
import { ThemeSwitcher } from "@/components/ThemeSwitcher";
import { PageHeaderProvider } from "@/contexts/PageHeaderProvider";
import { useSystemActions } from "@/contexts/useSystemActions";
import type { SystemAction } from "@/contexts/system-actions-context";
import ConfigPage from "@/pages/ConfigPage";
import DocsPage from "@/pages/DocsPage";
import EnvPage from "@/pages/EnvPage";
import SessionsPage from "@/pages/SessionsPage";
import LogsPage from "@/pages/LogsPage";
import AnalyticsPage from "@/pages/AnalyticsPage";
import CronPage from "@/pages/CronPage";
import SkillsPage from "@/pages/SkillsPage";
import ChatPage from "@/pages/ChatPage";
import { LanguageSwitcher } from "@/components/LanguageSwitcher";
import { useI18n } from "@/i18n";
import { PluginPage, PluginSlot, usePlugins } from "@/plugins";
import type { PluginManifest } from "@/plugins";
import { useTheme } from "@/themes";
import { isDashboardEmbeddedChatEnabled } from "@/lib/dashboard-flags";

function RootRedirect() {
  return <Navigate to={isDashboardEmbeddedChatEnabled() ? "/chat" : "/sessions"} replace />;
}

const CHAT_NAV_ITEM: NavItem = {
  path: "/chat",
  labelKey: "chat",
  label: "Chat",
  icon: Terminal,
};

const BUILTIN_ROUTES_CORE: Record<string, ComponentType> = {
  "/": RootRedirect,
  "/sessions": SessionsPage,
  "/analytics": AnalyticsPage,
  "/logs": LogsPage,
  "/cron": CronPage,
  "/skills": SkillsPage,
  "/config": ConfigPage,
  "/env": EnvPage,
  "/documentation": DocsPage,
};

const BUILTIN_NAV_REST: NavItem[] = [
  { path: "/sessions", labelKey: "sessions", label: "Sessions", icon: MessageSquare },
  { path: "/analytics", labelKey: "analytics", label: "Analytics", icon: BarChart3 },
  { path: "/logs", labelKey: "logs", label: "Logs", icon: FileText },
  { path: "/cron", labelKey: "cron", label: "Cron", icon: Clock },
  { path: "/skills", labelKey: "skills", label: "Skills", icon: Package },
  { path: "/config", labelKey: "config", label: "Config", icon: Settings },
  { path: "/env", labelKey: "keys", label: "Keys", icon: KeyRound },
  { path: "/documentation", labelKey: "documentation", label: "Documentation", icon: BookOpen },
];

const ICON_MAP: Record<string, ComponentType<{ className?: string }>> = {
  Activity,
  BarChart3,
  Clock,
  FileText,
  KeyRound,
  MessageSquare,
  Package,
  Settings,
  Puzzle,
  Sparkles,
  Terminal,
  Globe,
  Database,
  Shield,
  Wrench,
  Zap,
  Heart,
  Star,
  Code,
  Eye,
};

function resolveIcon(name: string): ComponentType<{ className?: string }> {
  return ICON_MAP[name] ?? Puzzle;
}

function buildNavItems(builtIn: NavItem[], manifests: PluginManifest[]): NavItem[] {
  const items = [...builtIn];

  for (const manifest of manifests) {
    if (manifest.tab.override || manifest.tab.hidden) continue;

    const pluginItem: NavItem = {
      path: manifest.tab.path,
      label: manifest.label,
      icon: resolveIcon(manifest.icon),
    };

    const pos = manifest.tab.position ?? "end";
    if (pos === "end") {
      items.push(pluginItem);
    } else if (pos.startsWith("after:")) {
      const target = "/" + pos.slice(6);
      const idx = items.findIndex((i) => i.path === target);
      items.splice(idx >= 0 ? idx + 1 : items.length, 0, pluginItem);
    } else if (pos.startsWith("before:")) {
      const target = "/" + pos.slice(7);
      const idx = items.findIndex((i) => i.path === target);
      items.splice(idx >= 0 ? idx : items.length, 0, pluginItem);
    } else {
      items.push(pluginItem);
    }
  }

  return items;
}

function buildRoutes(
  builtinRoutes: Record<string, ComponentType>,
  manifests: PluginManifest[],
): Array<{ key: string; path: string; element: ReactNode }> {
  const byOverride = new Map<string, PluginManifest>();
  const addons: PluginManifest[] = [];

  for (const m of manifests) {
    if (m.tab.override) byOverride.set(m.tab.override, m);
    else addons.push(m);
  }

  const routes: Array<{ key: string; path: string; element: ReactNode }> = [];

  for (const [path, Component] of Object.entries(builtinRoutes)) {
    const om = byOverride.get(path);
    routes.push(
      om
        ? { key: `override:${om.name}`, path, element: <PluginPage name={om.name} /> }
        : { key: `builtin:${path}`, path, element: <Component /> },
    );
  }

  for (const m of addons) {
    if (m.tab.hidden || builtinRoutes[m.tab.path]) continue;
    routes.push({ key: `plugin:${m.name}`, path: m.tab.path, element: <PluginPage name={m.name} /> });
  }

  for (const m of manifests) {
    if (!m.tab.hidden || builtinRoutes[m.tab.path] || m.tab.override) continue;
    routes.push({ key: `plugin:hidden:${m.name}`, path: m.tab.path, element: <PluginPage name={m.name} /> });
  }

  return routes;
}

export default function App() {
  const { t } = useI18n();
  const { pathname } = useLocation();
  const { manifests } = usePlugins();
  const { theme } = useTheme();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const closeMobile = useCallback(() => setMobileOpen(false), []);
  const normalizedPath = pathname.replace(/\/$/, "") || "/";
  const isChatRoute = normalizedPath === "/chat";
  const isDocsRoute = normalizedPath === "/documentation";
  const embeddedChat = isDashboardEmbeddedChatEnabled();

  const builtinRoutes = useMemo(
    () => ({ ...BUILTIN_ROUTES_CORE, ...(embeddedChat ? { "/chat": ChatPage } : {}) }),
    [embeddedChat],
  );

  const builtinNav = useMemo(
    () => (embeddedChat ? [CHAT_NAV_ITEM, ...BUILTIN_NAV_REST] : BUILTIN_NAV_REST),
    [embeddedChat],
  );

  const navItems = useMemo(() => buildNavItems(builtinNav, manifests), [builtinNav, manifests]);
  const routes = useMemo(() => buildRoutes(builtinRoutes, manifests), [builtinRoutes, manifests]);
  const pluginTabMeta = useMemo(
    () => manifests.filter((m) => !m.tab.hidden).map((m) => ({ path: m.tab.override ?? m.tab.path, label: m.label })),
    [manifests],
  );

  const layoutVariant = theme.layoutVariant ?? "standard";

  useEffect(() => {
    if (!mobileOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMobileOpen(false);
    };
    document.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [mobileOpen]);

  useEffect(() => {
    const mql = window.matchMedia("(min-width: 1024px)");
    const onChange = (e: MediaQueryListEvent) => {
      if (e.matches) setMobileOpen(false);
    };
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, []);

  return (
    <TooltipProvider delayDuration={150}>
      <div
        data-layout-variant={layoutVariant}
        className="mission-shell flex h-dvh max-h-dvh min-h-0 overflow-hidden bg-background text-foreground antialiased"
      >
        <PluginSlot name="backdrop" />

        {mobileOpen && (
          <button
            type="button"
            aria-label={t.app.closeNavigation}
            onClick={closeMobile}
            className="fixed inset-0 z-40 bg-black/35 backdrop-blur-sm lg:hidden"
          />
        )}

        <aside
          id="app-sidebar"
          aria-label={t.app.navigation}
          className={cn(
            "fixed inset-y-0 left-0 z-50 flex min-h-0 w-[18rem] flex-col border-r border-border bg-sidebar text-sidebar-foreground shadow-xl transition-transform duration-200 lg:relative lg:z-0 lg:translate-x-0 lg:shadow-none",
            mobileOpen ? "translate-x-0" : "-translate-x-full",
            sidebarCollapsed ? "lg:w-[4.75rem]" : "lg:w-[18rem]",
          )}
        >
          <div className="flex h-16 shrink-0 items-center gap-3 border-b border-sidebar-border px-4">
            <div className="flex size-9 shrink-0 items-center justify-center rounded-xl border border-sidebar-border bg-sidebar-primary text-sidebar-primary-foreground shadow-sm">
              <Terminal className="size-4" />
            </div>
            {!sidebarCollapsed && (
              <div className="min-w-0 leading-tight">
                <div className="truncate text-sm font-semibold tracking-tight">Mission Control</div>
                <div className="truncate text-xs text-muted-foreground">Hermes Agent</div>
              </div>
            )}
            <button
              type="button"
              onClick={closeMobile}
              aria-label={t.app.closeNavigation}
              className="ml-auto inline-flex size-8 items-center justify-center rounded-md text-muted-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground lg:hidden"
            >
              <X className="size-4" />
            </button>
          </div>

          <PluginSlot name="header-left" />

          <nav className="min-h-0 flex-1 overflow-y-auto px-2 py-3" aria-label={t.app.navigation}>
            <ul className="space-y-1">
              {navItems.map(({ path, label, labelKey, icon: Icon }) => {
                const navLabel = labelKey ? ((t.app.nav as Record<string, string>)[labelKey] ?? label) : label;
                return (
                  <li key={path}>
                    <NavLink
                      to={path}
                      end={path === "/sessions"}
                      onClick={closeMobile}
                      title={sidebarCollapsed ? navLabel : undefined}
                      className={({ isActive }) =>
                        cn(
                          "group flex h-10 items-center gap-3 rounded-lg px-3 text-sm font-medium transition-colors",
                          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                          isActive
                            ? "bg-sidebar-primary text-sidebar-primary-foreground shadow-sm"
                            : "text-muted-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                          sidebarCollapsed && "lg:justify-center lg:px-0",
                        )
                      }
                    >
                      <Icon className="size-4 shrink-0" />
                      {!sidebarCollapsed && <span className="truncate">{navLabel}</span>}
                    </NavLink>
                  </li>
                );
              })}
            </ul>
          </nav>

          <SidebarSystemActions collapsed={sidebarCollapsed} onNavigate={closeMobile} />

          <div className="flex shrink-0 items-center gap-2 border-t border-sidebar-border p-3">
            <PluginSlot name="header-right" />
            {!sidebarCollapsed && <ThemeSwitcher dropUp />}
            {sidebarCollapsed && <ThemeSwitcher dropUp iconOnly />}
            {!sidebarCollapsed && <LanguageSwitcher />}
          </div>

          {!sidebarCollapsed && <SidebarFooter />}
        </aside>

        <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
          <PluginSlot name="header-banner" />
          <header className="flex h-14 shrink-0 items-center gap-2 border-b border-border bg-background/80 px-3 backdrop-blur supports-[backdrop-filter]:bg-background/70 sm:px-4">
            <Button
              variant="ghost"
              size="icon"
              className="lg:hidden"
              onClick={() => setMobileOpen(true)}
              aria-label={t.app.openNavigation}
              aria-expanded={mobileOpen}
              aria-controls="app-sidebar"
            >
              <Menu className="size-5" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="hidden lg:inline-flex"
              onClick={() => setSidebarCollapsed((v) => !v)}
              aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            >
              {sidebarCollapsed ? <PanelLeftOpen className="size-5" /> : <PanelLeftClose className="size-5" />}
            </Button>
            <div className="min-w-0 flex-1">
              <div className="truncate text-sm font-semibold tracking-tight">Mission Control</div>
              <div className="hidden truncate text-xs text-muted-foreground sm:block">
                Remote Hermes cockpit · responsive chat, tools, sessions, logs
              </div>
            </div>
            <Badge variant="outline" className="hidden border-emerald-500/30 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 sm:inline-flex">
              Remote live
            </Badge>
          </header>

          <PageHeaderProvider pluginTabs={pluginTabMeta}>
            <div
              className={cn(
                "min-h-0 min-w-0 flex-1",
                isChatRoute
                  ? "flex flex-col overflow-hidden p-2 sm:p-3"
                  : isDocsRoute
                    ? "flex flex-col overflow-hidden p-0"
                    : "overflow-y-auto overflow-x-hidden p-4 sm:p-6",
              )}
            >
              <PluginSlot name="pre-main" />
              <Routes>
                {routes.map(({ key, path, element }) => (
                  <Route key={key} path={path} element={element} />
                ))}
                <Route path="*" element={<Navigate to={embeddedChat ? "/chat" : "/sessions"} replace />} />
              </Routes>
              <PluginSlot name="post-main" />
            </div>
          </PageHeaderProvider>
        </div>

        <PluginSlot name="overlay" />
      </div>
    </TooltipProvider>
  );
}

function SidebarSystemActions({ collapsed, onNavigate }: { collapsed: boolean; onNavigate: () => void }) {
  const { t } = useI18n();
  const navigate = useNavigate();
  const { activeAction, isBusy, isRunning, pendingAction, runAction } = useSystemActions();

  const items: SystemActionItem[] = [
    {
      action: "restart",
      icon: RotateCw,
      label: t.status.restartGateway,
      runningLabel: t.status.restartingGateway,
      spin: true,
    },
    {
      action: "update",
      icon: Download,
      label: t.status.updateHermes,
      runningLabel: t.status.updatingHermes,
      spin: false,
    },
  ];

  const handleClick = (action: SystemAction) => {
    if (isBusy) return;
    void runAction(action);
    navigate("/sessions");
    onNavigate();
  };

  return (
    <div className="shrink-0 border-t border-sidebar-border px-2 py-3">
      {!collapsed && <div className="px-3 pb-2 text-xs font-medium text-muted-foreground">System</div>}
      {!collapsed && <SidebarStatusStrip />}
      <ul className="mt-1 space-y-1">
        {items.map(({ action, icon: Icon, label, runningLabel, spin }) => {
          const isPending = pendingAction === action;
          const isActionRunning = activeAction === action && isRunning && !isPending;
          const busy = isPending || isActionRunning;
          const displayLabel = isActionRunning ? runningLabel : label;
          const disabled = isBusy && !busy;

          return (
            <li key={action}>
              <button
                type="button"
                onClick={() => handleClick(action)}
                disabled={disabled}
                aria-busy={busy}
                title={collapsed ? displayLabel : undefined}
                className={cn(
                  "flex h-9 w-full items-center gap-3 rounded-lg px-3 text-sm font-medium text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-40",
                  busy && "bg-sidebar-accent text-sidebar-accent-foreground",
                  collapsed && "justify-center px-0",
                )}
              >
                {isPending ? (
                  <Loader2 className="size-4 shrink-0 animate-spin" />
                ) : (
                  <Icon className={cn("size-4 shrink-0", isActionRunning && spin && "animate-spin", isActionRunning && !spin && "animate-pulse")} />
                )}
                {!collapsed && <span className="truncate">{displayLabel}</span>}
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

interface NavItem {
  icon: ComponentType<{ className?: string }>;
  label: string;
  labelKey?: string;
  path: string;
}

interface SystemActionItem {
  action: SystemAction;
  icon: ComponentType<{ className?: string }>;
  label: string;
  runningLabel: string;
  spin: boolean;
}
