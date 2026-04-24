import { useMemo } from "react";
import { Routes, Route, NavLink, Navigate } from "react-router-dom";
import {
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
} from "lucide-react";
import { Cell, Grid, SelectionSwitcher, Typography } from "@nous-research/ui";
import { cn } from "@/lib/utils";
import { Backdrop } from "@/components/Backdrop";
import StatusPage from "@/pages/StatusPage";
import ConfigPage from "@/pages/ConfigPage";
import EnvPage from "@/pages/EnvPage";
import SessionsPage from "@/pages/SessionsPage";
import LogsPage from "@/pages/LogsPage";
import AnalyticsPage from "@/pages/AnalyticsPage";
import CronPage from "@/pages/CronPage";
import SkillsPage from "@/pages/SkillsPage";
import { LanguageSwitcher } from "@/components/LanguageSwitcher";
import { ThemeSwitcher } from "@/components/ThemeSwitcher";
import { useI18n } from "@/i18n";
import { PluginSlot, usePlugins } from "@/plugins";
import type { RegisteredPlugin } from "@/plugins";
import { useTheme } from "@/themes";

/** Built-in route → default page component. Used both for standard routing
 *  and for resolving plugin `tab.override` values. Keys must match the
 *  `path` in `BUILTIN_NAV` so `/path` lookups stay consistent. */
const BUILTIN_ROUTES: Record<string, React.ComponentType> = {
  "/": StatusPage,
  "/sessions": SessionsPage,
  "/analytics": AnalyticsPage,
  "/logs": LogsPage,
  "/cron": CronPage,
  "/skills": SkillsPage,
  "/config": ConfigPage,
  "/env": EnvPage,
};

const BUILTIN_NAV: NavItem[] = [
  { path: "/", labelKey: "status", label: "Status", icon: Activity },
  {
    path: "/sessions",
    labelKey: "sessions",
    label: "Sessions",
    icon: MessageSquare,
  },
  {
    path: "/analytics",
    labelKey: "analytics",
    label: "Analytics",
    icon: BarChart3,
  },
  { path: "/logs", labelKey: "logs", label: "Logs", icon: FileText },
  { path: "/cron", labelKey: "cron", label: "Cron", icon: Clock },
  { path: "/skills", labelKey: "skills", label: "Skills", icon: Package },
  { path: "/config", labelKey: "config", label: "Config", icon: Settings },
  { path: "/env", labelKey: "keys", label: "Keys", icon: KeyRound },
];

// Plugins can reference any of these by name in their manifest — keeps bundle
// size sane vs. importing the full lucide-react set.
const ICON_MAP: Record<string, React.ComponentType<{ className?: string }>> = {
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

function resolveIcon(
  name: string,
): React.ComponentType<{ className?: string }> {
  return ICON_MAP[name] ?? Puzzle;
}

function buildNavItems(
  builtIn: NavItem[],
  plugins: RegisteredPlugin[],
): NavItem[] {
  const items = [...builtIn];

  for (const { manifest } of plugins) {
    // Plugins that replace a built-in route don't add a new tab entry —
    // they reuse the existing tab. The nav just lights up the original
    // built-in entry when the user visits `/`.
    if (manifest.tab.override) continue;
    // Hidden plugins register their component + slots but skip the nav.
    if (manifest.tab.hidden) continue;

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

/** Build the final route table, letting plugins override built-in pages.
 *
 *  Returns (path, Component, key) tuples. Plugins with `tab.override`
 *  win over both built-ins and other plugins (last registration wins if
 *  two plugins claim the same override, but we warn in dev). Plugins with
 *  a regular `tab.path` register alongside built-ins as standalone
 *  routes. */
function buildRoutes(
  plugins: RegisteredPlugin[],
): Array<{ key: string; path: string; Component: React.ComponentType }> {
  const overrides = new Map<string, RegisteredPlugin>();
  const addons: RegisteredPlugin[] = [];

  for (const p of plugins) {
    if (p.manifest.tab.override) {
      overrides.set(p.manifest.tab.override, p);
    } else {
      addons.push(p);
    }
  }

  const routes: Array<{
    key: string;
    path: string;
    Component: React.ComponentType;
  }> = [];

  for (const [path, Component] of Object.entries(BUILTIN_ROUTES)) {
    const override = overrides.get(path);
    if (override) {
      routes.push({
        key: `override:${override.manifest.name}`,
        path,
        Component: override.component,
      });
    } else {
      routes.push({ key: `builtin:${path}`, path, Component });
    }
  }

  for (const addon of addons) {
    // Don't double-register a plugin that shadows a built-in path via
    // `tab.path` — `override` is the supported mechanism for that.
    if (BUILTIN_ROUTES[addon.manifest.tab.path]) continue;
    routes.push({
      key: `plugin:${addon.manifest.name}`,
      path: addon.manifest.tab.path,
      Component: addon.component,
    });
  }

  return routes;
}

export default function App() {
  const { t } = useI18n();
  const { plugins } = usePlugins();
  const { theme } = useTheme();

  const navItems = useMemo(
    () => buildNavItems(BUILTIN_NAV, plugins),
    [plugins],
  );
  const routes = useMemo(() => buildRoutes(plugins), [plugins]);

  const layoutVariant = theme.layoutVariant ?? "standard";
  const showSidebar = layoutVariant === "cockpit";
  // Tiled layout drops the 1600px clamp so pages can use the full viewport;
  // standard + cockpit keep the centered reading width.
  const mainMaxWidth = layoutVariant === "tiled" ? "max-w-none" : "max-w-[1600px]";

  return (
    <div
      data-layout-variant={layoutVariant}
      className="text-midground font-mondwest bg-black min-h-screen flex flex-col uppercase antialiased overflow-x-hidden"
    >
      <SelectionSwitcher />
      <Backdrop />
      {/* Themes can style backdrop chrome via `componentStyles.backdrop.*`
          CSS vars read by <Backdrop />. Plugins can also inject full
          components into the backdrop layer via the `backdrop` slot —
          useful for scanlines, parallax stars, hero artwork, etc. */}
      <PluginSlot name="backdrop" />

      <header
        className={cn(
          "fixed top-0 left-0 right-0 z-40",
          "border-b border-current/20",
          "bg-background-base/90 backdrop-blur-sm",
        )}
        style={{
          // Themes can tweak header chrome (background, border-image,
          // clip-path) via these CSS vars. Unset vars compute to the
          // property's initial value, so themes opt in per-property.
          background: "var(--component-header-background)",
          borderImage: "var(--component-header-border-image)",
          clipPath: "var(--component-header-clip-path)",
        }}
      >
        <div className={cn("mx-auto flex h-12", mainMaxWidth)}>
          <PluginSlot name="header-left" />
          <div className="min-w-0 flex-1 overflow-x-auto scrollbar-none">
            <Grid
              className="h-full !border-t-0 !border-b-0"
              style={{
                gridTemplateColumns: `auto repeat(${navItems.length}, auto)`,
              }}
            >
              <Cell className="flex items-center !p-0 !px-3 sm:!px-5">
                <Typography
                  className="font-bold text-[1.0625rem] sm:text-[1.125rem] leading-[0.95] tracking-[0.0525rem] text-midground"
                  style={{ mixBlendMode: "plus-lighter" }}
                >
                  Hermes
                  <br />
                  Agent
                </Typography>
              </Cell>

              {navItems.map(({ path, label, labelKey, icon: Icon }) => (
                <Cell key={path} className="relative !p-0">
                  <NavLink
                    to={path}
                    end={path === "/"}
                    className={({ isActive }) =>
                      cn(
                        "group relative flex h-full w-full items-center gap-1.5",
                        "px-2.5 sm:px-4 py-2",
                        "font-mondwest text-[0.65rem] sm:text-[0.8rem] tracking-[0.12em]",
                        "whitespace-nowrap transition-colors cursor-pointer",
                        "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-midground",
                        isActive
                          ? "text-midground"
                          : "opacity-60 hover:opacity-100",
                      )
                    }
                    style={{
                      clipPath: "var(--component-tab-clip-path)",
                    }}
                  >
                    {({ isActive }) => (
                      <>
                        <Icon className="h-3.5 w-3.5 shrink-0" />
                        <span className="hidden sm:inline">
                          {labelKey
                            ? ((t.app.nav as Record<string, string>)[
                                labelKey
                              ] ?? label)
                            : label}
                        </span>

                        <span
                          aria-hidden
                          className="absolute inset-1 bg-midground opacity-0 pointer-events-none transition-opacity duration-200 group-hover:opacity-5"
                        />

                        {isActive && (
                          <span
                            aria-hidden
                            className="absolute bottom-0 left-0 right-0 h-px bg-midground"
                            style={{ mixBlendMode: "plus-lighter" }}
                          />
                        )}
                      </>
                    )}
                  </NavLink>
                </Cell>
              ))}
            </Grid>
          </div>

          <Grid className="h-full shrink-0 !border-t-0 !border-b-0">
            <Cell className="flex items-center gap-2 !p-0 !px-2 sm:!px-4">
              <PluginSlot name="header-right" />
              <ThemeSwitcher />
              <LanguageSwitcher />
              <Typography
                mondwest
                className="hidden sm:inline text-[0.7rem] tracking-[0.15em] opacity-50"
              >
                {t.app.webUi}
              </Typography>
            </Cell>
          </Grid>
        </div>
      </header>

      {/* Full-width banner slot under the nav, outside the main clamp —
          useful for marquee/alert/status strips themes want to show
          above page content. */}
      <PluginSlot name="header-banner" />

      <div
        className={cn(
          "relative z-2 mx-auto w-full flex-1 px-3 sm:px-6 pt-16 sm:pt-20 pb-4 sm:pb-8",
          mainMaxWidth,
          showSidebar && "flex gap-4 sm:gap-6",
        )}
      >
        {showSidebar && (
          <aside
            className={cn(
              "w-[260px] shrink-0 border-r border-current/20 pr-3 sm:pr-4",
              "hidden lg:block",
            )}
            style={{
              background: "var(--component-sidebar-background)",
              clipPath: "var(--component-sidebar-clip-path)",
              borderImage: "var(--component-sidebar-border-image)",
            }}
          >
            <PluginSlot
              name="sidebar"
              fallback={
                <div className="p-4 text-xs opacity-60 font-mondwest tracking-wide">
                  {/* Cockpit layout with no sidebar plugin — rare but valid;
                      the space still exists so the grid doesn't shift when
                      a plugin loads asynchronously. */}
                  sidebar slot empty
                </div>
              }
            />
          </aside>
        )}

        <main className="min-w-0 flex-1">
          <PluginSlot name="pre-main" />
          <Routes>
            {routes.map(({ key, path, Component }) => (
              <Route key={key} path={path} element={<Component />} />
            ))}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
          <PluginSlot name="post-main" />
        </main>
      </div>

      <footer className="relative z-2 border-t border-current/20">
        <Grid className={cn("mx-auto !border-t-0 !border-b-0", mainMaxWidth)}>
          <Cell className="flex items-center !px-3 sm:!px-6 !py-3">
            <PluginSlot
              name="footer-left"
              fallback={
                <Typography
                  mondwest
                  className="text-[0.7rem] sm:text-[0.8rem] tracking-[0.12em] opacity-60"
                >
                  {t.app.footer.name}
                </Typography>
              }
            />
          </Cell>
          <Cell className="flex items-center justify-end !px-3 sm:!px-6 !py-3">
            <PluginSlot
              name="footer-right"
              fallback={
                <Typography
                  mondwest
                  className="text-[0.6rem] sm:text-[0.7rem] tracking-[0.15em] text-midground"
                  style={{ mixBlendMode: "plus-lighter" }}
                >
                  {t.app.footer.org}
                </Typography>
              }
            />
          </Cell>
        </Grid>
      </footer>

      {/* Fixed-position overlay plugins (scanlines, vignettes, etc.) render
          above everything else. Each plugin is responsible for its own
          pointer-events and z-index. */}
      <PluginSlot name="overlay" />
    </div>
  );
}

interface NavItem {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  labelKey?: string;
  path: string;
}
