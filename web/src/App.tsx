import { useState } from "react";
import { Activity, KeyRound, Settings } from "lucide-react";
import StatusPage from "@/pages/StatusPage";
import ConfigPage from "@/pages/ConfigPage";
import EnvPage from "@/pages/EnvPage";

const NAV_ITEMS = [
  { id: "status", label: "Status", icon: Activity },
  { id: "config", label: "Config", icon: Settings },
  { id: "env", label: "API Keys", icon: KeyRound },
] as const;

type PageId = (typeof NAV_ITEMS)[number]["id"];

export default function App() {
  const [page, setPage] = useState<PageId>("status");

  return (
    <div className="flex min-h-screen flex-col bg-background text-foreground">
      <header className="sticky top-0 z-40 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="mx-auto flex h-14 max-w-5xl items-center gap-6 px-6">
          <span className="text-lg font-bold tracking-tight">Hermes Agent</span>

          <nav className="flex items-center gap-1">
            {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                type="button"
                onClick={() => setPage(id)}
                className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors cursor-pointer ${
                  page === id
                    ? "bg-secondary text-secondary-foreground"
                    : "text-muted-foreground hover:bg-secondary/50 hover:text-foreground"
                }`}
              >
                <Icon className="h-4 w-4" />
                {label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto w-full max-w-5xl flex-1 px-6 py-8">
        {page === "status" && <StatusPage />}
        {page === "config" && <ConfigPage />}
        {page === "env" && <EnvPage />}
      </main>
    </div>
  );
}
