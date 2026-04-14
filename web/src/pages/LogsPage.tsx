import { useEffect, useState, useCallback, useRef } from "react";
import {
  AlertTriangle,
  Bug,
  ChevronRight,
  FileText,
  Hash,
  Layers,
  RefreshCw,
} from "lucide-react";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

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
          <FileText className="h-4 w-4 text-muted-foreground" />
          <span className="text-xs text-muted-foreground">
            {file} / {level.toLowerCase()} / {component}
          </span>
          {loading && (
            <div className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-primary border-t-transparent" />
          )}
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Switch checked={autoRefresh} onCheckedChange={setAutoRefresh} />
            <Label className="text-xs">Auto-refresh</Label>
            {autoRefresh && (
              <Badge variant="success" className="text-[10px]">
                <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
                Live
              </Badge>
            )}
          </div>
          <Button variant="outline" size="sm" onClick={fetchLogs} className="text-xs h-7">
            <RefreshCw className="h-3 w-3 mr-1" />
            Refresh
          </Button>
        </div>
      </div>

      {/* ═══════════════ Sidebar + Content ═══════════════ */}
      <div className="flex flex-col sm:flex-row gap-4" style={{ minHeight: "calc(100vh - 180px)" }}>
        {/* ---- Sidebar ---- */}
        <div className="sm:w-52 sm:shrink-0">
          <div className="sm:sticky sm:top-[72px] flex flex-col gap-1">
            {/* File section */}
            <div className="flex sm:flex-col gap-1 overflow-x-auto sm:overflow-x-visible scrollbar-none pb-1 sm:pb-0">
              <SidebarHeading icon={FileText} label="File" />
              {FILES.map((f) => (
                <SidebarItem
                  key={f}
                  label={f}
                  active={file === f}
                  indented
                  onClick={() => setFile(f)}
                />
              ))}

              <div className="hidden sm:block border-t border-border my-1" />

              <SidebarHeading icon={AlertTriangle} label="Level" />
              {LEVELS.map((l) => (
                <SidebarItem
                  key={l}
                  label={l.toLowerCase()}
                  active={level === l}
                  indented
                  onClick={() => setLevel(l)}
                />
              ))}

              <div className="hidden sm:block border-t border-border my-1" />

              <SidebarHeading icon={Layers} label="Component" />
              {COMPONENTS.map((c) => (
                <SidebarItem
                  key={c}
                  label={c}
                  active={component === c}
                  indented
                  onClick={() => setComponent(c)}
                />
              ))}

              <div className="hidden sm:block border-t border-border my-1" />

              <SidebarHeading icon={Hash} label="Lines" />
              {LINE_COUNTS.map((n) => (
                <SidebarItem
                  key={n}
                  label={String(n)}
                  active={lineCount === n}
                  indented
                  onClick={() => setLineCount(n)}
                />
              ))}
            </div>
          </div>
        </div>

        {/* ---- Content ---- */}
        <div className="flex-1 min-w-0">
          <Card>
            <CardHeader className="py-3 px-4">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Bug className="h-4 w-4" />
                  {file} logs
                </CardTitle>
                <Badge variant="secondary" className="text-[10px]">
                  {lines.length} line{lines.length !== 1 ? "s" : ""}
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="px-4 pb-4">
              {error && (
                <div className="bg-destructive/10 border border-destructive/20 p-3 mb-4">
                  <p className="text-sm text-destructive">{error}</p>
                </div>
              )}

              <div
                ref={scrollRef}
                className="border border-border bg-background p-4 font-mono-ui text-xs leading-5 overflow-auto max-h-[600px] min-h-[200px]"
              >
                {lines.length === 0 && !loading && (
                  <p className="text-muted-foreground text-center py-8">No log lines found</p>
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

function SidebarHeading({ icon: Icon, label }: SidebarHeadingProps) {
  return (
    <div className="flex items-center gap-2 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/60">
      <Icon className="h-3.5 w-3.5" />
      {label}
    </div>
  );
}

function SidebarItem({ label, active, indented, onClick }: SidebarItemProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`group flex items-center gap-2 ${indented ? "sm:pl-6" : ""} px-2.5 py-1.5 text-left text-xs transition-colors cursor-pointer ${
        active
          ? "bg-primary/10 text-primary font-medium"
          : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
      }`}
    >
      <span className="flex-1 truncate">{label}</span>
      {active && <ChevronRight className="h-3 w-3 text-primary/50 shrink-0" />}
    </button>
  );
}

interface SidebarHeadingProps {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
}

interface SidebarItemProps {
  label: string;
  active: boolean;
  indented?: boolean;
  onClick: () => void;
}
