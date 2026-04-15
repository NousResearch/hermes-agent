import { Routes, Route, NavLink, Navigate } from "react-router-dom";
import { Activity, BarChart3, Clock, FileText, KeyRound, MessageSquare, Package, Settings } from "lucide-react";
import StatusPage from "@/pages/StatusPage";
import ConfigPage from "@/pages/ConfigPage";
import EnvPage from "@/pages/EnvPage";
import SessionsPage from "@/pages/SessionsPage";
import LogsPage from "@/pages/LogsPage";
import AnalyticsPage from "@/pages/AnalyticsPage";
import CronPage from "@/pages/CronPage";
import SkillsPage from "@/pages/SkillsPage";
import { LanguageSwitcher } from "@/components/LanguageSwitcher";
import { useI18n } from "@/i18n";

const NAV_ITEMS = [
  { path: "/", labelKey: "status" as const, icon: Activity },
  { path: "/sessions", labelKey: "sessions" as const, icon: MessageSquare },
  { path: "/analytics", labelKey: "analytics" as const, icon: BarChart3 },
  { path: "/logs", labelKey: "logs" as const, icon: FileText },
  { path: "/cron", labelKey: "cron" as const, icon: Clock },
  { path: "/skills", labelKey: "skills" as const, icon: Package },
  { path: "/config", labelKey: "config" as const, icon: Settings },
  { path: "/env", labelKey: "keys" as const, icon: KeyRound },
] as const;

export default function App() {
  const { t } = useI18n();

  return (
    <div className="flex min-h-screen flex-col bg-background text-foreground overflow-x-hidden">
      <div className="noise-overlay" />
      <div className="warm-glow" />

      <header className="fixed top-0 left-0 right-0 z-40 border-b border-border bg-background/90 backdrop-blur-sm">
        <div className="mx-auto flex h-12 max-w-[1400px] items-stretch">
          <div className="flex items-center border-r border-border px-3 sm:px-5 shrink-0">
            <span className="font-collapse text-lg sm:text-xl font-bold tracking-wider uppercase blend-lighter">
              H<span className="hidden sm:inline">ermes </span>A<span className="hidden sm:inline">gent</span>
            </span>
          </div>

          <nav className="flex items-stretch overflow-x-auto scrollbar-none">
            {NAV_ITEMS.map(({ path, labelKey, icon: Icon }) => (
              <NavLink
                key={path}
                to={path}
                end={path === "/"}
                className={({ isActive }) =>
                  `group relative inline-flex items-center gap-1 sm:gap-1.5 border-r border-border px-2.5 sm:px-4 py-2 font-display text-[0.65rem] sm:text-[0.8rem] tracking-[0.12em] uppercase whitespace-nowrap transition-colors cursor-pointer shrink-0 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring ${
                    isActive
                      ? "text-foreground"
                      : "text-muted-foreground hover:text-foreground"
                  }`
                }
              >
                {({ isActive }) => (
                  <>
                    <Icon className="h-4 w-4 sm:h-3.5 sm:w-3.5 shrink-0" />
                    <span className="hidden sm:inline">{t.app.nav[labelKey]}</span>
                    <span className="absolute inset-0 bg-foreground pointer-events-none transition-opacity duration-150 group-hover:opacity-5 opacity-0" />
                    {isActive && (
                      <span className="absolute bottom-0 left-0 right-0 h-px bg-foreground" />
                    )}
                  </>
                )}
              </NavLink>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-2 px-2 sm:px-4">
            <LanguageSwitcher />
            <span className="hidden sm:inline font-display text-[0.7rem] tracking-[0.15em] uppercase opacity-50">
              {t.app.webUi}
            </span>
          </div>
        </div>
      </header>

      <main className="relative z-2 mx-auto w-full max-w-[1400px] flex-1 px-3 sm:px-6 pt-16 sm:pt-20 pb-4 sm:pb-8">
        <Routes>
          <Route path="/" element={<StatusPage />} />
          <Route path="/sessions" element={<SessionsPage />} />
          <Route path="/analytics" element={<AnalyticsPage />} />
          <Route path="/logs" element={<LogsPage />} />
          <Route path="/cron" element={<CronPage />} />
          <Route path="/skills" element={<SkillsPage />} />
          <Route path="/config" element={<ConfigPage />} />
          <Route path="/env" element={<EnvPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>

      <footer className="relative z-2 border-t border-border">
        <div className="mx-auto flex max-w-[1400px] items-center justify-between px-3 sm:px-6 py-3">
          <span className="font-display text-[0.7rem] sm:text-[0.8rem] tracking-[0.12em] uppercase opacity-50">
            {t.app.footer.name}
          </span>
          <span className="font-display text-[0.6rem] sm:text-[0.7rem] tracking-[0.15em] uppercase text-foreground/40">
            {t.app.footer.org}
          </span>
        </div>
      </footer>
    </div>
  );
}
