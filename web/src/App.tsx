import { useState, useEffect, useRef } from "react";
import { Activity, BarChart3, Clock, FileText, KeyRound, MessageSquare, Package, Settings } from "lucide-react";
import StatusPage from "@/pages/StatusPage";
import ConfigPage from "@/pages/ConfigPage";
import EnvPage from "@/pages/EnvPage";
import SessionsPage from "@/pages/SessionsPage";
import LogsPage from "@/pages/LogsPage";
import AnalyticsPage from "@/pages/AnalyticsPage";
import CronPage from "@/pages/CronPage";
import SkillsPage from "@/pages/SkillsPage";
import { useI18n } from "@/lib/i18n";

const NAV_KEYS = [
  { id: "status", labelKey: "nav.status", icon: Activity },
  { id: "sessions", labelKey: "nav.sessions", icon: MessageSquare },
  { id: "analytics", labelKey: "nav.analytics", icon: BarChart3 },
  { id: "logs", labelKey: "nav.logs", icon: FileText },
  { id: "cron", labelKey: "nav.cron", icon: Clock },
  { id: "skills", labelKey: "nav.skills", icon: Package },
  { id: "config", labelKey: "nav.config", icon: Settings },
  { id: "env", labelKey: "nav.keys", icon: KeyRound },
] as const;

type PageId = (typeof NAV_KEYS)[number]["id"];

const PAGE_COMPONENTS: Record<PageId, React.FC> = {
  status: StatusPage,
  sessions: SessionsPage,
  analytics: AnalyticsPage,
  logs: LogsPage,
  cron: CronPage,
  skills: SkillsPage,
  config: ConfigPage,
  env: EnvPage,
};

export default function App() {
  const [page, setPage] = useState<PageId>("status");
  const [animKey, setAnimKey] = useState(0);
  const initialRef = useRef(true);
  const { t, locale, setLocale } = useI18n();

  useEffect(() => {
    // Skip the animation key bump on initial mount to avoid re-mounting
    // the default page component (which causes duplicate API requests).
    if (initialRef.current) {
      initialRef.current = false;
      return;
    }
    setAnimKey((k) => k + 1);
  }, [page]);

  const PageComponent = PAGE_COMPONENTS[page];

  return (
    <div className="flex min-h-screen flex-col bg-background text-foreground overflow-x-hidden">
      {/* Global grain + warm glow (matches landing page) */}
      <div className="noise-overlay" />
      <div className="warm-glow" />

      {/* ---- Header with grid-border nav ---- */}
      <header className="sticky top-0 z-40 border-b border-border bg-background/90 backdrop-blur-sm">
        <div className="mx-auto flex h-12 max-w-[1400px] items-stretch">
          {/* Brand — abbreviated on mobile */}
          <div className="flex items-center border-r border-border px-3 sm:px-5 shrink-0">
            <span className="font-collapse text-lg sm:text-xl font-bold tracking-wider uppercase blend-lighter">
              H<span className="hidden sm:inline">ermes </span>A<span className="hidden sm:inline">gent</span>
            </span>
          </div>

          {/* Nav — icons only on mobile, icon+label on sm+ */}
          <nav className="flex items-stretch overflow-x-auto scrollbar-none">
            {NAV_KEYS.map(({ id, labelKey, icon: Icon }) => (
              <button
                key={id}
                type="button"
                onClick={() => setPage(id)}
                className={`group relative inline-flex items-center gap-1 sm:gap-1.5 border-r border-border px-2.5 sm:px-4 py-2 font-display text-[0.65rem] sm:text-[0.8rem] tracking-[0.12em] uppercase whitespace-nowrap transition-colors cursor-pointer shrink-0 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring ${
                  page === id
                    ? "text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                <Icon className="h-4 w-4 sm:h-3.5 sm:w-3.5 shrink-0" />
                <span className="hidden sm:inline">{t(labelKey)}</span>
                {/* Hover highlight */}
                <span className="absolute inset-0 bg-foreground pointer-events-none transition-opacity duration-150 group-hover:opacity-5 opacity-0" />
                {/* Active indicator */}
                {page === id && (
                  <span className="absolute bottom-0 left-0 right-0 h-px bg-foreground" />
                )}
              </button>
            ))}
          </nav>

          {/* Version badge + language switcher */}
          <div className="ml-auto flex items-center gap-2 px-3 sm:px-4 text-muted-foreground">
            <span className="font-display text-[0.7rem] tracking-[0.15em] uppercase opacity-50 hidden sm:inline">
              {t("app.webUI")}
            </span>
            <button
              type="button"
              onClick={() => setLocale(locale === "en" ? "zh" : "en")}
              className="font-display text-[0.65rem] sm:text-[0.7rem] tracking-[0.1em] uppercase border border-border px-2 py-0.5 hover:bg-foreground/5 transition-colors cursor-pointer"
            >
              {locale === "en" ? "中" : "EN"}
            </button>
          </div>
        </div>
      </header>

      <main
        key={animKey}
        className="relative z-2 mx-auto w-full max-w-[1400px] flex-1 px-3 sm:px-6 py-4 sm:py-8"
        style={{ animation: "fade-in 150ms ease-out" }}
      >
        <PageComponent />
      </main>

      {/* ---- Footer ---- */}
      <footer className="relative z-2 border-t border-border">
        <div className="mx-auto flex max-w-[1400px] items-center justify-between px-3 sm:px-6 py-3">
          <span className="font-display text-[0.7rem] sm:text-[0.8rem] tracking-[0.12em] uppercase opacity-50">
            {t("app.title")}
          </span>
          <span className="font-display text-[0.6rem] sm:text-[0.7rem] tracking-[0.15em] uppercase text-foreground/40">
            {t("app.footer")}
          </span>
        </div>
      </footer>
    </div>
  );
}
