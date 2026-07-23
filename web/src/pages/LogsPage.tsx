import {
  useEffect,
  useLayoutEffect,
  useState,
  useCallback,
  useRef,
  type RefObject,
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
  "w-full max-w-full flex-wrap justify-start self-start [&_button]:min-h-11 [&_button]:flex-1 sm:w-fit sm:[&_button]:min-h-0 sm:[&_button]:flex-none";

type LogFile = (typeof FILES)[number];
type LogLevel = (typeof LEVELS)[number];
type LogComponent = (typeof COMPONENTS)[number];
type LogLineCount = (typeof LINE_COUNTS)[number];

interface LogsFilterBarProps {
  component: LogComponent;
  file: LogFile;
  level: LogLevel;
  lineCount: LogLineCount;
  labels: {
    component: string;
    file: string;
    level: string;
    lines: string;
    title: string;
  };
  onComponentChange(value: LogComponent): void;
  onFileChange(value: LogFile): void;
  onLevelChange(value: LogLevel): void;
  onLineCountChange(value: LogLineCount): void;
}

export function LogsFilterBar({
  component,
  file,
  labels,
  level,
  lineCount,
  onComponentChange,
  onFileChange,
  onLevelChange,
  onLineCountChange,
}: LogsFilterBarProps) {
  return (
    <div
      role="toolbar"
      aria-label={labels.title}
      className="flex min-w-0 max-w-full flex-col items-start gap-3 sm:flex-row sm:flex-wrap sm:items-start sm:gap-x-6 sm:gap-y-3"
    >
      <FilterGroup label={labels.file} className={filterGroupClass}>
        <Segmented
          className={segmentedClass}
          value={file}
          onChange={(value) => onFileChange(value as LogFile)}
          options={toSegmentOptions(FILES)}
        />
      </FilterGroup>

      <FilterGroup label={labels.level} className={filterGroupClass}>
        <Segmented
          className={segmentedClass}
          value={level}
          onChange={(value) => onLevelChange(value as LogLevel)}
          options={toSegmentOptions(LEVELS)}
        />
      </FilterGroup>

      <FilterGroup label={labels.component} className={filterGroupClass}>
        <Segmented
          className={segmentedClass}
          value={component}
          onChange={(value) => onComponentChange(value as LogComponent)}
          options={toSegmentOptions(COMPONENTS)}
        />
      </FilterGroup>

      <FilterGroup label={labels.lines} className={filterGroupClass}>
        <Segmented
          className={segmentedClass}
          value={String(lineCount)}
          onChange={(value) => onLineCountChange(Number(value) as LogLineCount)}
          options={LINE_COUNTS.map((count) => ({
            value: String(count),
            label: String(count),
          }))}
        />
      </FilterGroup>
    </div>
  );
}

interface LogOutputProps {
  emptyLabel: string;
  lines: string[];
  loading: boolean;
  scrollRef?: RefObject<HTMLDivElement | null>;
}

export function LogOutput({ emptyLabel, lines, loading, scrollRef }: LogOutputProps) {
  return (
    <div
      ref={scrollRef}
      data-testid="log-output"
      className="h-[45dvh] min-h-48 max-w-full overflow-auto p-3 font-mono-ui text-xs leading-5 sm:h-auto sm:min-h-[400px] sm:max-h-[calc(100vh-220px)] sm:p-4"
    >
      {lines.length === 0 && !loading ? (
        <p className="py-8 text-center text-muted-foreground">{emptyLabel}</p>
      ) : null}
      {lines.map((line, index) => {
        const cls = classifyLine(line);
        return (
          <div
            key={index}
            className={`${LINE_COLORS[cls]} -mx-1 min-w-max whitespace-pre px-1 hover:bg-secondary/20`}
          >
            {line}
          </div>
        );
      })}
    </div>
  );
}

export default function LogsPage() {
  const [file, setFile] = useState<LogFile>("agent");
  const [level, setLevel] = useState<LogLevel>("ALL");
  const [component, setComponent] = useState<LogComponent>("all");
  const [lineCount, setLineCount] = useState<LogLineCount>(100);
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
          className="min-h-11 min-w-11 text-muted-foreground hover:text-foreground sm:min-h-0 sm:min-w-0"
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
        <div className="flex min-h-11 items-center gap-2 sm:min-h-0">
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
    fetchLogs();
  }, [fetchLogs]);

  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(fetchLogs, 5000);
    return () => clearInterval(interval);
  }, [autoRefresh, fetchLogs]);

  return (
    <div className="flex min-w-0 max-w-full flex-col gap-4">
      <PluginSlot name="logs:top" />
      <LogsFilterBar
        component={component}
        file={file}
        level={level}
        lineCount={lineCount}
        labels={{
          component: t.logs.component,
          file: t.logs.file,
          level: t.logs.level,
          lines: t.logs.lines,
          title: t.logs.title,
        }}
        onComponentChange={setComponent}
        onFileChange={setFile}
        onLevelChange={setLevel}
        onLineCountChange={setLineCount}
      />

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

          <LogOutput
            emptyLabel={t.logs.noLogLines}
            lines={lines}
            loading={loading}
            scrollRef={scrollRef}
          />
        </CardContent>
      </Card>
      <PluginSlot name="logs:bottom" />
    </div>
  );
}
