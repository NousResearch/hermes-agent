import { useEffect, useState, useCallback, useRef } from "react";
import { FileText, RefreshCw, ChevronRight } from "lucide-react";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { useI18n } from "@/i18n";

const FILES = ["agent", "errors", "gateway"] as const;
const LEVELS = ["ALL", "DEBUG", "INFO", "WARNING", "ERROR"] as const;
const COMPONENTS = ["all", "gateway", "agent", "tools", "cli", "cron"] as const;
const LINE_COUNTS = [50, 100, 200, 500] as const;

function classifyLine(line: string): "error" | "warning" | "info" | "debug" {
  const upper = line.toUpperCase();
  if (upper.includes("ERROR") || upper.includes("CRITICAL") || upper.includes("FATAL")) return "error";
  if (upper.includes("WARNING") || upper.includes("WARN")) return "warning";
  if (upper.includes("DEBUG")) return "debug";
  return "info";
}

const LINE_COLORS: Record<string, string> = {
  error: "text-destructive",
  warning: "text-warning",
  info: "text-foreground",
  debug: "text-muted-foreground/60",
};

function SidebarHeading({ children }: { children: React.ReactNode }) {
  return (
    <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/60 px-2.5 pt-3 pb-1">
      {children}
    </span>
  );
}

function SidebarItem<T extends string>({
  label,
  value,
  current,
  onChange,
}: SidebarItemProps<T>) {
  const isActive = current === value;
  return (
    <button
      type="button"
      onClick={() => onChange(value)}
      className={`group flex items-center gap-2 px-2.5 py-1 text-left text-xs transition-colors cursor-pointer ${
        isActive
          ? "bg-primary/10 text-primary font-medium"
          : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
      }`}
    >
      <span className="flex-1 truncate">{label}</span>
      {isActive && <ChevronRight className="h-3 w-3 text-primary/50 shrink-0" />}
    </button>
  );
}

export default function LogsPage() {
  const [file, setFile] = useState<(typeof FILES)[number]>("agent");
  const [level, setLevel] = useState<(typeof LEVELS)[number]>("ALL");
  const [component, setComponent] = useState<(typeof COMPONENTS)[number]>("all");
  const [lineCount, setLineCount] = useState<(typeof LINE_COUNTS)[number]>(100);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [lines, setLines] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const { t } = useI18n();

  const fetchLogs = useCallback(() => {
    setLoading(true);
    setError(null);
    api
      .getLogs({ file, lines: lineCount, level, component })
      .then((resp) => {
        setLines(resp.lines);
        setTimeout(() => {
          if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
          }
        }, 50);
      })
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, [file, lineCount, level, component]);

  useEffect(() => {
    fetchLogs();
  }, [fetchLogs]);

  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(fetchLogs, 5000);
    return () => clearInterval(interval);
  }, [autoRefresh, fetchLogs]);

  return (
    <div className="flex flex-col gap-4">
      {/* ═══════════════ Header ═══════════════ */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <FileText className="h-5 w-5 text-muted-foreground" />
          <h1 className="text-base font-semibold">{t.logs.title}</h1>
          {loading && (
            <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
          )}
          <Badge variant="secondary" className="text-[10px]">
            {file} · {level} · {component}
          </Badge>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Switch checked={autoRefresh} onCheckedChange={setAutoRefresh} />
            <Label className="text-xs">{t.logs.autoRefresh}</Label>
            {autoRefresh && (
              <Badge variant="success" className="text-[10px]">
                <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
                {t.common.live}
              </Badge>
            )}
          </div>
          <Button variant="outline" size="sm" onClick={fetchLogs} className="text-xs h-7">
            <RefreshCw className="h-3 w-3 mr-1" />
            {t.common.refresh}
          </Button>
        </div>
      </div>

      {/* ═══════════════ Sidebar + Content ═══════════════ */}
      <div className="flex flex-col sm:flex-row gap-4" style={{ minHeight: "calc(100vh - 180px)" }}>
        {/* ---- Sidebar ---- */}
        <div className="sm:w-44 sm:shrink-0">
          <div className="sm:sticky sm:top-[72px] flex flex-col gap-0.5">
            <SidebarHeading>{t.logs.file}</SidebarHeading>
            {FILES.map((f) => (
              <SidebarItem key={f} label={f} value={f} current={file} onChange={setFile} />
            ))}

            <SidebarHeading>{t.logs.level}</SidebarHeading>
            {LEVELS.map((l) => (
              <SidebarItem key={l} label={l} value={l} current={level} onChange={setLevel} />
            ))}

            <SidebarHeading>{t.logs.component}</SidebarHeading>
            {COMPONENTS.map((c) => (
              <SidebarItem key={c} label={c} value={c} current={component} onChange={setComponent} />
            ))}

            <SidebarHeading>{t.logs.lines}</SidebarHeading>
            {LINE_COUNTS.map((n) => (
              <SidebarItem
                key={n}
                label={String(n)}
                value={String(n)}
                current={String(lineCount)}
                onChange={(v) => setLineCount(Number(v) as (typeof LINE_COUNTS)[number])}
              />
            ))}
          </div>
        </div>

        {/* ---- Content ---- */}
        <div className="flex-1 min-w-0">
          <Card>
            <CardHeader className="py-3 px-4">
              <CardTitle className="text-sm flex items-center gap-2">
                <FileText className="h-4 w-4" />
                {file}.log
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              {error && (
                <div className="bg-destructive/10 border-b border-destructive/20 p-3">
                  <p className="text-sm text-destructive">{error}</p>
                </div>
              )}

              <div
                ref={scrollRef}
                className="p-4 font-mono-ui text-xs leading-5 overflow-auto max-h-[600px] min-h-[200px]"
              >
                {lines.length === 0 && !loading && (
                  <p className="text-muted-foreground text-center py-8">{t.logs.noLogLines}</p>
                )}
                {lines.map((line, i) => {
                  const cls = classifyLine(line);
                  return (
                    <div key={i} className={`${LINE_COLORS[cls]} hover:bg-secondary/20 px-1 -mx-1`}>
                      {line}
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

interface SidebarItemProps<T extends string> {
  label: string;
  value: T;
  current: T;
  onChange: (v: T) => void;
}
