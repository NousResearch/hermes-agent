import { useEffect, useLayoutEffect, useState, useCallback, useRef } from "react";
import { FileText, RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { FilterGroup, Segmented } from "@/components/ui/segmented";
import { useI18n } from "@/i18n";
import { usePageHeader } from "@/contexts/usePageHeader";
import { PluginSlot } from "@/plugins";

const FILES = ["agent", "errors", "gateway"] as const;
const LEVELS = ["ALL", "DEBUG", "INFO", "WARNING", "ERROR"] as const;
const COMPONENTS = ["all", "gateway", "agent", "tools", "cli", "cron"] as const;
const LINE_COUNTS = [50, 100, 200, 500] as const;

function classifyLine(line: string): "error" | "warning" | "info" | "debug" {
  const upper = line.toUpperCase();
  if (
    upper.includes("ERROR") ||
    upper.includes("CRITICAL") ||
    upper.includes("FATAL")
  )
    return "error";
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

const toOptions = <T extends string>(values: readonly T[]) =>
  values.map((v) => ({ value: v, label: v }));

export default function LogsPage() {
  const [file, setFile] = useState<(typeof FILES)[number]>("agent");
  const [level, setLevel] = useState<(typeof LEVELS)[number]>("ALL");
  const [component, setComponent] =
    useState<(typeof COMPONENTS)[number]>("all");
  const [lineCount, setLineCount] = useState<(typeof LINE_COUNTS)[number]>(100);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [lines, setLines] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const { t } = useI18n();
  const { setAfterTitle, setEnd } = usePageHeader();

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

  useLayoutEffect(() => {
    setAfterTitle(
      <span className="flex items-center gap-2">
        {loading && (
          <div className="h-4 w-4 shrink-0 animate-spin rounded-full border-2 border-primary border-t-transparent" />
        )}
        <Badge variant="secondary" className="text-[10px]">
          {file} · {level} · {component}
        </Badge>
      </span>,
    );
    setEnd(
      <div className="flex w-full min-w-0 flex-wrap items-center justify-end gap-2 sm:gap-3">
        <div className="flex items-center gap-2">
          <Switch
            checked={autoRefresh}
            onCheckedChange={setAutoRefresh}
            id="logs-auto-refresh"
          />
          <Label htmlFor="logs-auto-refresh" className="text-xs cursor-pointer">
            {t.logs.autoRefresh}
          </Label>
          {autoRefresh && (
            <Badge variant="success" className="text-[10px]">
              <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
              {t.common.live}
            </Badge>
          )}
        </div>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={fetchLogs}
          disabled={loading}
          className="h-7 text-xs"
        >
          <RefreshCw className="mr-1 h-3 w-3" />
          {t.common.refresh}
        </Button>
      </div>,
    );
    return () => {
      setAfterTitle(null);
      setEnd(null);
    };
  }, [
    autoRefresh,
    component,
    file,
    level,
    loading,
    setAfterTitle,
    setEnd,
    t.common.live,
    t.common.refresh,
    t.logs.autoRefresh,
    fetchLogs,
  ]);

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
      <PluginSlot name="logs:top" />
      {/* ═══════════════ Filter toolbar ═══════════════ */}
      <div
        role="toolbar"
        aria-label={t.logs.title}
        className="flex flex-wrap items-center gap-x-6 gap-y-2"
      >
        <FilterGroup label={t.logs.file}>
          <Segmented value={file} onChange={setFile} options={toOptions(FILES)} />
        </FilterGroup>

        <FilterGroup label={t.logs.level}>
          <Segmented value={level} onChange={setLevel} options={toOptions(LEVELS)} />
        </FilterGroup>

        <FilterGroup label={t.logs.component}>
          <Segmented
            value={component}
            onChange={setComponent}
            options={toOptions(COMPONENTS)}
          />
        </FilterGroup>

        <FilterGroup label={t.logs.lines}>
          <Segmented
            value={String(lineCount)}
            onChange={(v) =>
              setLineCount(Number(v) as (typeof LINE_COUNTS)[number])
            }
            options={LINE_COUNTS.map((n) => ({
              value: String(n),
              label: String(n),
            }))}
          />
        </FilterGroup>
      </div>

      {/* ═══════════════ Log viewer ═══════════════ */}
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
            className="p-4 font-mono-ui text-xs leading-5 overflow-auto min-h-[400px] max-h-[calc(100vh-220px)]"
          >
            {lines.length === 0 && !loading && (
              <p className="text-muted-foreground text-center py-8">
                {t.logs.noLogLines}
              </p>
            )}
            {lines.map((line, i) => {
              const cls = classifyLine(line);
              return (
                <div
                  key={i}
                  className={`${LINE_COLORS[cls]} hover:bg-secondary/20 px-1 -mx-1`}
                >
                  {line}
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>
      <PluginSlot name="logs:bottom" />
    </div>
  );
}
