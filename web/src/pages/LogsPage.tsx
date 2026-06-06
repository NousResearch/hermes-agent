import {
  useEffect,
  useLayoutEffect,
  useState,
  useCallback,
  useRef,
  useMemo,
} from "react";
import { FileText, RefreshCw, Search, X, Download, ArrowDown, ArrowUp, AlertCircle, AlertTriangle } from "lucide-react";
import { api } from "@/lib/api";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { FilterGroup, Segmented } from "@nous-research/ui/ui/components/segmented";
import { Input } from "@nous-research/ui/ui/components/input";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Switch } from "@nous-research/ui/ui/components/switch";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { Label } from "@nous-research/ui/ui/components/label";
import { useI18n } from "@/i18n";
import { usePageHeader } from "@/contexts/usePageHeader";
import { PluginSlot } from "@/plugins";
import { cn } from "@/lib/utils";

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

/** Render a log line with the search term highlighted. */
function HighlightedLine({ text, term }: { text: string; term: string }) {
  if (!term) return <>{text}</>;
  const lower = text.toLowerCase();
  const termLower = term.toLowerCase();
  const parts: React.ReactNode[] = [];
  let i = 0;
  while (i < text.length) {
    const idx = lower.indexOf(termLower, i);
    if (idx === -1) {
      parts.push(text.slice(i));
      break;
    }
    if (idx > i) parts.push(text.slice(i, idx));
    parts.push(
      <mark
        key={idx}
        className="rounded-sm bg-yellow-400/30 text-inherit"
      >
        {text.slice(idx, idx + term.length)}
      </mark>,
    );
    i = idx + term.length;
  }
  return <>{parts}</>;
}

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

  // Search state — searchTerm is what's sent to the API (committed on Enter /
  // clear); searchInput tracks the live text field value.
  const [searchInput, setSearchInput] = useState("");
  const [searchTerm, setSearchTerm] = useState("");

  const scrollRef = useRef<HTMLDivElement>(null);
  const { t } = useI18n();
  const { setAfterTitle, setEnd } = usePageHeader();

  // ── Derived counts ──────────────────────────────────────────────────────
  const errorCount = useMemo(
    () => lines.filter((l) => classifyLine(l) === "error").length,
    [lines],
  );
  const warningCount = useMemo(
    () => lines.filter((l) => classifyLine(l) === "warning").length,
    [lines],
  );

  // ── Fetch ───────────────────────────────────────────────────────────────
  const fetchLogs = useCallback(() => {
    setLoading(true);
    setError(null);
    api
      .getLogs({
        file,
        lines: lineCount,
        level,
        component,
        search: searchTerm || undefined,
      })
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
  }, [file, lineCount, level, component, searchTerm]);

  // ── Search handlers ─────────────────────────────────────────────────────
  const commitSearch = useCallback(() => {
    setSearchTerm(searchInput.trim());
  }, [searchInput]);

  const clearSearch = useCallback(() => {
    setSearchInput("");
    setSearchTerm("");
  }, []);

  const handleSearchKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") commitSearch();
    if (e.key === "Escape") clearSearch();
  };

  // ── Scroll helpers ──────────────────────────────────────────────────────
  const scrollToBottom = useCallback(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, []);

  const scrollToTop = useCallback(() => {
    scrollRef.current?.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  // ── Download ────────────────────────────────────────────────────────────
  const handleDownload = useCallback(() => {
    const blob = new Blob([lines.join("\n")], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `hermes-${file}.log`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [file, lines]);

  // ── Page header ─────────────────────────────────────────────────────────
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

  // ── Effects ─────────────────────────────────────────────────────────────
  useEffect(() => {
    fetchLogs();
  }, [fetchLogs]);

  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(fetchLogs, 5000);
    return () => clearInterval(interval);
  }, [autoRefresh, fetchLogs]);

  // ── Render ───────────────────────────────────────────────────────────────
  return (
    <div className="flex min-w-0 max-w-full flex-col gap-4">
      <PluginSlot name="logs:top" />

      {/* ── Filters ── */}
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

      {/* ── Search bar ── */}
      <div className="flex min-w-0 max-w-full items-center gap-2">
        <div className="relative flex-1 max-w-md">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            onKeyDown={handleSearchKeyDown}
            placeholder="Search logs… (Enter to apply)"
            className="h-8 pl-8 pr-8 text-xs font-mono-ui"
            aria-label="Search log lines"
          />
          {searchInput && (
            <button
              type="button"
              onClick={clearSearch}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              aria-label="Clear search"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
        {searchTerm && (
          <span className="shrink-0 text-xs text-muted-foreground">
            {lines.length} {lines.length === 1 ? "match" : "matches"}
          </span>
        )}
      </div>

      {/* ── Error / warning summary ── */}
      {!searchTerm && (errorCount > 0 || warningCount > 0) && (
        <div className="flex flex-wrap items-center gap-2 rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs">
          {errorCount > 0 && (
            <span className="flex items-center gap-1 text-destructive">
              <AlertCircle className="h-3.5 w-3.5" />
              {errorCount} {errorCount === 1 ? "error" : "errors"}
            </span>
          )}
          {errorCount > 0 && warningCount > 0 && (
            <span className="text-muted-foreground">·</span>
          )}
          {warningCount > 0 && (
            <span className="flex items-center gap-1 text-warning">
              <AlertTriangle className="h-3.5 w-3.5" />
              {warningCount} {warningCount === 1 ? "warning" : "warnings"}
            </span>
          )}
          <span className="ml-auto text-muted-foreground">
            in last {lineCount} lines
          </span>
        </div>
      )}

      {/* ── Log viewer card ── */}
      <Card className="min-w-0 max-w-full overflow-hidden">
        <CardHeader className="py-3 px-4">
          <CardTitle className="text-sm flex items-center gap-2 min-w-0">
            <FileText className="h-4 w-4 shrink-0" />
            <span className="truncate">{file}.log</span>
            {searchTerm && (
              <Badge tone="secondary" className="text-xs shrink-0">
                search: {searchTerm}
              </Badge>
            )}
            <div className="ml-auto flex items-center gap-1 shrink-0">
              <Button
                ghost
                size="icon"
                title="Scroll to top"
                aria-label="Scroll to top"
                onClick={scrollToTop}
                className="h-6 w-6 text-muted-foreground hover:text-foreground"
              >
                <ArrowUp className="h-3.5 w-3.5" />
              </Button>
              <Button
                ghost
                size="icon"
                title="Scroll to bottom"
                aria-label="Scroll to bottom"
                onClick={scrollToBottom}
                className="h-6 w-6 text-muted-foreground hover:text-foreground"
              >
                <ArrowDown className="h-3.5 w-3.5" />
              </Button>
              <Button
                ghost
                size="icon"
                title="Download current view"
                aria-label="Download log"
                onClick={handleDownload}
                disabled={lines.length === 0}
                className="h-6 w-6 text-muted-foreground hover:text-foreground"
              >
                <Download className="h-3.5 w-3.5" />
              </Button>
            </div>
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
                {searchTerm
                  ? `No lines match "${searchTerm}"`
                  : t.logs.noLogLines}
              </p>
            )}
            {lines.map((line, i) => {
              const cls = classifyLine(line);
              return (
                <div
                  key={i}
                  className={cn(
                    LINE_COLORS[cls],
                    "hover:bg-secondary/20 px-1 -mx-1",
                    searchTerm && "cursor-pointer",
                  )}
                  title={searchTerm ? "Click to copy" : undefined}
                  onClick={
                    searchTerm
                      ? () => navigator.clipboard.writeText(line).catch(() => {})
                      : undefined
                  }
                >
                  <HighlightedLine text={line} term={searchTerm} />
                </div>
              );
            })}
          </div>

          {lines.length > 0 && (
            <div className="border-t border-border/50 px-4 py-2 text-xs text-muted-foreground flex items-center justify-between">
              <span>
                {lines.length} {lines.length === 1 ? "line" : "lines"}
                {searchTerm && " matched"}
              </span>
              <button
                type="button"
                onClick={scrollToBottom}
                className="flex items-center gap-1 hover:text-foreground transition-colors"
              >
                <ArrowDown className="h-3 w-3" />
                bottom
              </button>
            </div>
          )}
        </CardContent>
      </Card>
      <PluginSlot name="logs:bottom" />
    </div>
  );
}
