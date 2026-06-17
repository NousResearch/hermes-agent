import {
  useEffect,
  useLayoutEffect,
  useState,
  useCallback,
  useRef,
} from "react";
import { FileText, RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { FilterGroup, Segmented } from "@nous-research/ui/ui/components/segmented";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Switch } from "@nous-research/ui/ui/components/switch";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { Label } from "@nous-research/ui/ui/components/label";
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
  debug: "text-text-tertiary",
};

const formatFilterLabel = (value: string) => value.toUpperCase();

const toSegmentOptions = <T extends string>(values: readonly T[]) =>
  values.map((v) => ({ value: v, label: formatFilterLabel(v) }));

const filterGroupClass =
  "flex min-w-0 w-full flex-col items-start gap-1.5 sm:w-auto sm:max-w-full sm:flex-row sm:items-center";

const segmentedClass =
  "w-fit max-w-full flex-wrap justify-start self-start";

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
      <span className="flex items-center gap-1.5">
        <Badge tone="secondary" className="text-xs">
          {formatFilterLabel(file)} · {formatFilterLabel(level)} ·{" "}
          {formatFilterLabel(component)}
        </Badge>
        <Button
          type="button"
          ghost
          size="icon"
          className="text-muted-foreground hover:text-foreground"
          onClick={fetchLogs}
          disabled={loading}
          aria-label={t.common.refresh}
        >
          {loading ? <Spinner /> : <RefreshCw />}
        </Button>
      </span>,
    );
    setEnd(
      <div className="flex w-full min-w-0 flex-wrap items-center justify-start gap-2 sm:justify-end sm:gap-3">
        <div className="flex items-center gap-2">
          <Label htmlFor="logs-auto-refresh" className="text-xs cursor-pointer">
            {t.logs.autoRefresh}
          </Label>
          <Switch
            checked={autoRefresh}
            onCheckedChange={setAutoRefresh}
            id="logs-auto-refresh"
          />
          {autoRefresh && (
            <Badge tone="success" className="text-xs">
              <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
              {t.common.live}
            </Badge>
          )}
        </div>
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
    const id = window.setTimeout(fetchLogs, 0);
    return () => window.clearTimeout(id);
  }, [fetchLogs]);

  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(fetchLogs, 5000);
    return () => clearInterval(interval);
  }, [autoRefresh, fetchLogs]);

  return (
    <div className="flex min-w-0 max-w-full flex-col gap-4">
      <PluginSlot name="logs:top" />
      <div
        role="toolbar"
        aria-label={t.logs.title}
        className="flex min-w-0 max-w-full flex-col items-start gap-3 sm:flex-row sm:flex-wrap sm:items-start sm:gap-x-6 sm:gap-y-3"
      >
        <FilterGroup label={t.logs.file} className={filterGroupClass}>
          <Segmented
            className={segmentedClass}
            value={file}
            onChange={setFile}
            options={toSegmentOptions(FILES)}
          />
        </FilterGroup>

        <FilterGroup label={t.logs.level} className={filterGroupClass}>
          <Segmented
            className={segmentedClass}
            value={level}
            onChange={setLevel}
            options={toSegmentOptions(LEVELS)}
          />
        </FilterGroup>

        <FilterGroup label={t.logs.component} className={filterGroupClass}>
          <Segmented
            className={segmentedClass}
            value={component}
            onChange={setComponent}
            options={toSegmentOptions(COMPONENTS)}
          />
        </FilterGroup>

        <FilterGroup label={t.logs.lines} className={filterGroupClass}>
          <Segmented
            className={segmentedClass}
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

      <Card className="min-w-0 max-w-full overflow-hidden">
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
            className="max-w-full min-h-[400px] max-h-[calc(100vh-220px)] overflow-auto p-4 font-mono-ui text-xs leading-5 break-words"
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
