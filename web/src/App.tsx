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
import { SelectionSwitcher } from "@nous-research/ui";
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
import { ProfileSwitcher } from "@/components/ProfileSwitcher";
import { useI18n } from "@/i18n";
import { usePlugins } from "@/plugins";
import type { RegisteredPlugin } from "@/plugins";

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
    <div className="min-h-screen overflow-x-hidden bg-background text-foreground antialiased">
      <SelectionSwitcher />
      <Backdrop />

      <header className="fixed inset-x-0 top-0 z-40 border-b border-border/80 bg-background/92 backdrop-blur-xl">
        <div className="mx-auto flex h-14 max-w-[1600px] items-center gap-3 px-3 sm:h-16 sm:px-6">
          <div className="flex min-w-0 items-center gap-3 pr-1 sm:pr-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl border border-border bg-card shadow-sm">
              <Sparkles className="h-4 w-4 text-primary" />
            </div>
            <div className="min-w-0">
              <div className="truncate text-sm font-semibold tracking-tight sm:text-base">Hermes Agent</div>
              <div className="hidden text-xs text-muted-foreground sm:block">{t.app.webUi}</div>
            </div>
          </div>

          <div className="min-w-0 flex-1 overflow-x-auto scrollbar-none">
            <nav className="flex min-w-max items-center gap-1 py-2">
              {navItems.map(({ path, label, labelKey, icon: Icon }) => (
                <NavLink
                  key={path}
                  to={path}
                  end={path === "/"}
                  className={({ isActive }) =>
                    cn(
                      "flex h-10 items-center gap-2 rounded-xl px-3 text-sm font-medium transition-colors",
                      "whitespace-nowrap border border-transparent",
                      "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
                      isActive
                        ? "border-border bg-card text-foreground shadow-sm"
                        : "text-muted-foreground hover:bg-secondary hover:text-foreground",
                    )
                  }
                >
                  <Icon className="h-4 w-4 shrink-0" />
                  <span>{labelKey ? ((t.app.nav as Record<string, string>)[labelKey] ?? label) : label}</span>
                </NavLink>
              ))}
            </nav>
          </div>

          <div className="flex shrink-0 items-center gap-1.5 sm:gap-2">
            <ProfileSwitcher />
            <ThemeSwitcher />
            <LanguageSwitcher />
          </div>
        </div>
      </header>

      <main className="relative z-10 mx-auto w-full max-w-[1600px] px-3 pb-6 pt-20 sm:px-6 sm:pb-8 sm:pt-24">
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

      <footer className="relative z-10 border-t border-border/80 bg-background/70">
        <div className="mx-auto flex max-w-[1600px] items-center justify-between gap-3 px-3 py-4 text-xs text-muted-foreground sm:px-6">
          <span>{t.app.footer.name} · v0.6</span>
          <span className="hidden sm:inline">{t.app.footer.org}</span>
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
