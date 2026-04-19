import { useMemo } from "react";
import { Routes, Route, NavLink, Navigate } from "react-router-dom";
import {
  Activity, BarChart3, Clock, FileText, KeyRound,
  MessageSquare, Package, Settings, Puzzle,
  Sparkles, Terminal, Globe, Database, Shield,
  Wrench, Zap, Heart, Star, Code, Eye,
} from "lucide-react";
import { SelectionSwitcher } from "@nous-research/ui/ui/components/selection-switcher";
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
import { usePlugins } from "@/plugins";
import type { RegisteredPlugin } from "@/plugins";

const BUILTIN_NAV: NavItem[] = [
  { path: "/", labelKey: "status", label: "Status", icon: Activity },
  { path: "/sessions", labelKey: "sessions", label: "Sessions", icon: MessageSquare },
  { path: "/analytics", labelKey: "analytics", label: "Analytics", icon: BarChart3 },
  { path: "/logs", labelKey: "logs", label: "Logs", icon: FileText },
  { path: "/cron", labelKey: "cron", label: "Cron", icon: Clock },
  { path: "/skills", labelKey: "skills", label: "Skills", icon: Package },
  { path: "/config", labelKey: "config", label: "Config", icon: Settings },
  { path: "/env", labelKey: "keys", label: "Keys", icon: KeyRound },
];

// Plugins can reference any of these by name in their manifest — keeps bundle
// size sane vs. importing the full lucide-react set.
const ICON_MAP: Record<string, React.ComponentType<{ className?: string }>> = {
  Activity, BarChart3, Clock, FileText, KeyRound,
  MessageSquare, Package, Settings, Puzzle,
  Sparkles, Terminal, Globe, Database, Shield,
  Wrench, Zap, Heart, Star, Code, Eye,
};

function resolveIcon(name: string): React.ComponentType<{ className?: string }> {
  return ICON_MAP[name] ?? Puzzle;
}

function buildNavItems(builtIn: NavItem[], plugins: RegisteredPlugin[]): NavItem[] {
  const items = [...builtIn];

  for (const { manifest } of plugins) {
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

export default function App() {
  const { t } = useI18n();
  const { plugins } = usePlugins();

  const navItems = useMemo(
    () => buildNavItems(BUILTIN_NAV, plugins),
    [plugins],
  );

  return (
    <div className="text-midground font-mondwest bg-black min-h-screen flex flex-col uppercase antialiased overflow-x-hidden">
      <SelectionSwitcher />
      <Backdrop />

      <header
        className={cn(
          "fixed top-0 left-0 right-0 z-40",
          "border-b border-current/20",
          "bg-background-base/90 backdrop-blur-sm",
        )}
      >
        <div className="mx-auto flex h-12 max-w-[1600px] items-stretch">
          <div className="flex items-center border-r border-current/20 px-3 sm:px-5 shrink-0">
            <span
              className="font-sans font-bold text-[1.0625rem] sm:text-[1.125rem] leading-[0.95] tracking-[0.0525rem] text-midground"
              style={{ mixBlendMode: "plus-lighter" }}
            >
              Hermes
              <br />
              Agent
            </span>
          </div>

          <nav className="flex items-stretch overflow-x-auto scrollbar-none">
            {navItems.map(({ path, label, labelKey, icon: Icon }) => (
              <NavLink
                key={path}
                to={path}
                end={path === "/"}
                className={({ isActive }) =>
                  cn(
                    "group relative inline-flex items-center gap-1.5 shrink-0",
                    "border-r border-current/20 px-2.5 sm:px-4 py-2",
                    "font-mondwest text-[0.65rem] sm:text-[0.8rem] tracking-[0.12em]",
                    "whitespace-nowrap transition-colors cursor-pointer",
                    "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-midground",
                    isActive
                      ? "text-midground"
                      : "opacity-60 hover:opacity-100",
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <Icon className="h-3.5 w-3.5 shrink-0" />
                    <span className="hidden sm:inline">
                      {labelKey ? (t.app.nav as Record<string, string>)[labelKey] ?? label : label}
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
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-2 border-l border-current/20 px-2 sm:px-4">
            <ThemeSwitcher />
            <LanguageSwitcher />
            <span className="hidden sm:inline font-mondwest text-[0.7rem] tracking-[0.15em] opacity-50">
              {t.app.webUi}
            </span>
          </div>
        </div>
      </header>

      <main className="relative z-2 mx-auto w-full max-w-[1600px] flex-1 px-3 sm:px-6 pt-16 sm:pt-20 pb-4 sm:pb-8">
        <Routes>
          <Route path="/" element={<StatusPage />} />
          <Route path="/sessions" element={<SessionsPage />} />
          <Route path="/analytics" element={<AnalyticsPage />} />
          <Route path="/logs" element={<LogsPage />} />
          <Route path="/cron" element={<CronPage />} />
          <Route path="/skills" element={<SkillsPage />} />
          <Route path="/config" element={<ConfigPage />} />
          <Route path="/env" element={<EnvPage />} />

          {plugins.map(({ manifest, component: PluginComponent }) => (
            <Route
              key={manifest.name}
              path={manifest.tab.path}
              element={<PluginComponent />}
            />
          ))}

          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>

      <footer className="relative z-2 border-t border-current/20">
        <div className="mx-auto flex max-w-[1600px] items-center justify-between px-3 sm:px-6 py-3">
          <span className="font-mondwest text-[0.7rem] sm:text-[0.8rem] tracking-[0.12em] opacity-60">
            {t.app.footer.name}
          </span>
          <span
            className="font-mondwest text-[0.6rem] sm:text-[0.7rem] tracking-[0.15em] text-midground"
            style={{ mixBlendMode: "plus-lighter" }}
          >
            {t.app.footer.org}
          </span>
        </div>
      </footer>
    </div>
  );
}

interface NavItem {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  labelKey?: string;
  path: string;
}
