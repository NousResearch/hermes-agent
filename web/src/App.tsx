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

const NAV_ITEMS = [
  { id: "status", label: "Status", icon: Activity },
  { id: "sessions", label: "Sessions", icon: MessageSquare },
  { id: "analytics", label: "Analytics", icon: BarChart3 },
  { id: "logs", label: "Logs", icon: FileText },
  { id: "cron", label: "Cron", icon: Clock },
  { id: "skills", label: "Skills", icon: Package },
  { id: "config", label: "Config", icon: Settings },
  { id: "env", label: "Keys", icon: KeyRound },
] as const;

type PageId = (typeof NAV_ITEMS)[number]["id"];

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
      <div className="noise-overlay" />
      <div className="warm-glow" />

      <header className="sticky top-0 z-40 border-b border-black/[0.06] bg-white/72 backdrop-blur-2xl supports-[backdrop-filter]:bg-white/62">
        <div className="mx-auto flex min-h-14 max-w-[1180px] flex-col gap-2 px-4 py-2 sm:min-h-[64px] sm:flex-row sm:items-center sm:gap-5 sm:px-6">
          <button
            type="button"
            onClick={() => setPage("status")}
            className="flex shrink-0 items-center gap-2 rounded-full px-1 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/30"
          >
            <span className="flex h-7 w-7 items-center justify-center rounded-full bg-foreground text-[0.72rem] font-semibold text-background shadow-sm">
              H
            </span>
            <span className="text-sm font-semibold tracking-[-0.02em] sm:text-base">
              Hermes Agent
            </span>
          </button>

          <nav className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto rounded-full bg-black/[0.035] p-1 scrollbar-none">
            {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                type="button"
                onClick={() => setPage(id)}
                className={`inline-flex shrink-0 items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium tracking-[-0.01em] transition-all cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/30 sm:text-sm ${
                  page === id
                    ? "bg-white text-foreground shadow-[0_1px_8px_rgba(0,0,0,0.08)]"
                    : "text-muted-foreground hover:bg-white/60 hover:text-foreground"
                }`}
              >
                <Icon className="h-3.5 w-3.5 shrink-0" />
                <span>{label}</span>
              </button>
            ))}
          </nav>

          <div className="hidden shrink-0 items-center gap-2 rounded-full bg-white px-3 py-1.5 text-xs font-medium text-muted-foreground shadow-sm ring-1 ring-black/[0.04] lg:flex">
            <span className="h-1.5 w-1.5 rounded-full bg-success" />
            Web UI
          </div>
        </div>
      </header>

      <main
        key={animKey}
        className="relative z-10 mx-auto w-full max-w-[1180px] flex-1 px-4 py-6 sm:px-6 sm:py-10"
        style={{ animation: "fade-in 150ms ease-out" }}
      >
        <PageComponent />
      </main>

      <footer className="relative z-10 border-t border-black/[0.06] bg-white/50">
        <div className="mx-auto flex max-w-[1180px] items-center justify-between px-4 py-5 text-xs text-muted-foreground sm:px-6">
          <span>Hermes Agent</span>
          <span>Nous Research</span>
        </div>
      </footer>
    </div>
  );
}
