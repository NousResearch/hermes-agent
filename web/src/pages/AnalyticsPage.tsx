import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  ArrowDown,
  ArrowUp,
  ArrowUpDown,
  BarChart3,
  Brain,
  ChevronDown,
  Cpu,
  RefreshCw,
  TrendingUp,
} from "lucide-react";
import { api } from "@/lib/api";
import type {
  AnalyticsResponse,
  AnalyticsDailyEntry,
  AnalyticsModelEntry,
  AnalyticsSkillEntry,
  SkillActivityResponse,
} from "@/lib/api";
import { timeAgo } from "@/lib/utils";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Stats } from "@nous-research/ui/ui/components/stats";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { usePageHeader } from "@/contexts/usePageHeader";
import { useI18n } from "@/i18n";
import { PluginSlot } from "@/plugins";

const PERIODS = [
  { label: "7d", days: 7 },
  { label: "30d", days: 30 },
  { label: "90d", days: 90 },
] as const;

const CHART_HEIGHT_PX = 160;
const DEFAULT_PAGE_SIZE = 15;

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function formatDate(day: string): string {
  try {
    const d = new Date(day + "T00:00:00");
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  } catch {
    return day;
  }
}

// ---------------------------------------------------------------------------
// Sorting
// ---------------------------------------------------------------------------

function useTableSort<T>(
  data: T[],
  defaultKey: keyof T & string,
  defaultDir: "asc" | "desc" = "desc",
) {
  const [sortKey, setSortKey] = useState<string>(defaultKey);
  const [sortDir, setSortDir] = useState<"asc" | "desc">(defaultDir);

  const sorted = useMemo(() => {
    return [...data].sort((a, b) => {
      const aVal = a[sortKey as keyof T];
      const bVal = b[sortKey as keyof T];
      // Nulls always last regardless of direction
      if (aVal === null || aVal === undefined) return 1;
      if (bVal === null || bVal === undefined) return -1;
      if (aVal === bVal) return 0;
      const cmp = aVal > bVal ? 1 : -1;
      return sortDir === "asc" ? cmp : -cmp;
    });
  }, [data, sortKey, sortDir]);

  const toggle = useCallback(
    (key: string) => {
      if (key === sortKey) {
        setSortDir((d) => (d === "asc" ? "desc" : "asc"));
      } else {
        setSortKey(key);
        setSortDir("desc");
      }
    },
    [sortKey],
  );

  return { sorted, sortKey, sortDir, toggle };
}

function SortHeader({
  label,
  col,
  sortKey,
  sortDir,
  toggle,
  className,
}: {
  label: string;
  col: string;
  sortKey: string;
  sortDir: "asc" | "desc";
  toggle: (key: string) => void;
  className?: string;
}) {
  const active = col === sortKey;
  return (
    <th
      onClick={() => toggle(col)}
      className={`cursor-pointer select-none ${className ?? ""}`}
    >
      <span className="inline-flex items-center gap-1.5 rounded px-1 -mx-1 py-0.5 hover:bg-secondary/20 transition-colors">
        {label}
        {active ? (
          sortDir === "asc" ? (
            <ArrowUp className="h-3.5 w-3.5 text-foreground/80 shrink-0" />
          ) : (
            <ArrowDown className="h-3.5 w-3.5 text-foreground/80 shrink-0" />
          )
        ) : (
          <ArrowUpDown className="h-3 w-3 text-muted-foreground/40 shrink-0" />
        )}
      </span>
    </th>
  );
}

// ---------------------------------------------------------------------------
// Pagination hook
// ---------------------------------------------------------------------------

function usePagination(pageSize = DEFAULT_PAGE_SIZE) {
  const [page, setPage] = useState(0);

  const reset = useCallback(() => setPage(0), []);

  const paginate = useCallback(
    <T,>(items: T[]): { pageItems: T[]; start: number; end: number; totalPages: number } => {
      const totalPages = Math.max(1, Math.ceil(items.length / pageSize));
      const safePage = Math.min(page, totalPages - 1);
      const start = safePage * pageSize;
      const end = Math.min(start + pageSize, items.length);
      return { pageItems: items.slice(start, end), start, end, totalPages };
    },
    [page, pageSize],
  );

  return { page, setPage, reset, paginate };
}

function PaginationBar({
  total,
  start,
  end,
  totalPages,
  page,
  onPage,
}: {
  total: number;
  start: number;
  end: number;
  totalPages: number;
  page: number;
  onPage: (p: number) => void;
}) {
  const { t } = useI18n();
  const [input, setInput] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  if (total <= DEFAULT_PAGE_SIZE) return null;

  const jump = () => {
    const n = parseInt(input, 10);
    if (!isNaN(n) && n >= 1 && n <= totalPages) {
      onPage(n - 1);
    }
    setInput("");
    inputRef.current?.blur();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") { jump(); }
    if (e.key === "Escape") { setInput(""); inputRef.current?.blur(); }
  };

  return (
    <div className="flex items-center justify-between pt-3 mt-3 border-t border-border/50">
      <span className="text-xs text-muted-foreground">
        {t.analytics.showing
          .replace("{start}", String(start + 1))
          .replace("{end}", String(end))
          .replace("{total}", String(total))}
      </span>
      <div className="flex items-center gap-1.5">
        <Button
          size="xs"
          outlined
          disabled={page === 0}
          onClick={() => onPage(page - 1)}
        >
          {t.analytics.prev}
        </Button>
        {totalPages <= 7 ? (
          /* Few pages: show all page buttons */
          <div className="flex items-center gap-0.5">
            {Array.from({ length: totalPages }, (_, i) => (
              <button
                key={i}
                onClick={() => onPage(i)}
                className={`min-w-[1.75rem] h-6 text-xs rounded transition-colors ${
                  i === page
                    ? "bg-primary text-primary-foreground font-medium"
                    : "text-muted-foreground hover:text-foreground hover:bg-secondary/30"
                }`}
              >
                {i + 1}
              </button>
            ))}
          </div>
        ) : (
          /* Many pages: show current/total + jump input */
          <div className="flex items-center gap-1.5">
            <span className="text-xs text-muted-foreground tabular-nums">
              {page + 1} / {totalPages}
            </span>
            <input
              ref={inputRef}
              type="text"
              inputMode="numeric"
              value={input}
              onChange={(e) => setInput(e.target.value.replace(/\D/g, ""))}
              onKeyDown={handleKeyDown}
              onBlur={() => { if (!input) return; jump(); }}
              placeholder={String(page + 1)}
              className="w-12 h-6 text-xs text-center bg-secondary/30 border border-border rounded
                         text-foreground placeholder:text-muted-foreground/50
                         focus:outline-none focus:ring-1 focus:ring-primary/50"
            />
            <Button size="xs" outlined disabled={!input} onClick={jump}>
              Go
            </Button>
          </div>
        )}
        <Button
          size="xs"
          outlined
          disabled={page >= totalPages - 1}
          onClick={() => onPage(page + 1)}
        >
          {t.analytics.next}
        </Button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// CollapsibleCard — wraps Card with collapse/expand in header
// ---------------------------------------------------------------------------

function CollapsibleCard({
  icon,
  title,
  children,
  defaultOpen = true,
  headerExtra,
}: {
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
  headerExtra?: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <Card>
      <CardHeader
        className="cursor-pointer select-none"
        onClick={() => setOpen((v) => !v)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {icon}
            <CardTitle className="text-base">{title}</CardTitle>
            <ChevronDown
              className={`h-4 w-4 text-muted-foreground transition-transform duration-200 ${open ? "" : "-rotate-90"}`}
            />
          </div>
          {headerExtra}
        </div>
      </CardHeader>
      {open && <CardContent>{children}</CardContent>}
    </Card>
  );
}


function TokenBarChart({ daily }: { daily: AnalyticsDailyEntry[] }) {
  const { t } = useI18n();
  if (daily.length === 0) return null;

  const maxTokens = Math.max(
    ...daily.map((d) => d.input_tokens + d.output_tokens),
    1,
  );

  return (
    <CollapsibleCard
      icon={<BarChart3 className="h-5 w-5 text-muted-foreground" />}
      title={t.analytics.dailyTokenUsage}
      defaultOpen={true}
      headerExtra={
        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          <div className="flex items-center gap-1.5">
            <div className="h-2.5 w-2.5 bg-[var(--midground)]/70" />
            {t.analytics.input}
          </div>
          <div className="flex items-center gap-1.5">
            <div className="h-2.5 w-2.5 bg-[var(--color-success)]" />
            {t.analytics.output}
          </div>
        </div>
      }
    >
      <div
        className="flex items-end gap-[2px]"
        style={{ height: CHART_HEIGHT_PX }}
      >
        {daily.map((d) => {
          const total = d.input_tokens + d.output_tokens;
          const inputH = Math.round(
            (d.input_tokens / maxTokens) * CHART_HEIGHT_PX,
          );
          const outputH = Math.round(
            (d.output_tokens / maxTokens) * CHART_HEIGHT_PX,
          );
          return (
            <div
              key={d.day}
              className="flex-1 min-w-0 group relative flex flex-col justify-end"
              style={{ height: CHART_HEIGHT_PX }}
            >
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block z-10 pointer-events-none">
                <div className="bg-popover border border-border px-2.5 py-1.5 text-[10px] text-foreground shadow-lg whitespace-nowrap">
                  <div className="font-medium">{formatDate(d.day)}</div>
                  <div>
                    {t.analytics.input}: {formatTokens(d.input_tokens)}
                  </div>
                  <div>
                    {t.analytics.output}: {formatTokens(d.output_tokens)}
                  </div>
                  <div>
                    {t.analytics.total}: {formatTokens(total)}
                  </div>
                </div>
              </div>

              <div
                className="w-full bg-[var(--midground)]/70"
                style={{ height: Math.max(inputH, total > 0 ? 1 : 0) }}
              />

              <div
                className="w-full bg-[var(--color-success)]/70"
                style={{
                  height: Math.max(outputH, d.output_tokens > 0 ? 1 : 0),
                }}
              />
            </div>
          );
        })}
      </div>

      <div className="flex justify-between mt-2 text-[10px] text-muted-foreground">
        <span>{daily.length > 0 ? formatDate(daily[0].day) : ""}</span>
        {daily.length > 2 && (
          <span>{formatDate(daily[Math.floor(daily.length / 2)].day)}</span>
        )}
        <span>
          {daily.length > 1 ? formatDate(daily[daily.length - 1].day) : ""}
        </span>
      </div>
    </CollapsibleCard>
  );
}

function DailyTable({ daily }: { daily: AnalyticsDailyEntry[] }) {
  const { t } = useI18n();
  const { sorted, sortKey, sortDir, toggle } = useTableSort(daily, "day", "desc");
  const { page, setPage, reset, paginate } = usePagination();

  // Reset page when sort changes or data changes
  useEffect(() => { reset(); }, [daily, reset]);

  if (daily.length === 0) return null;

  const { pageItems, start, end, totalPages } = paginate(sorted);

  return (
    <CollapsibleCard
      icon={<TrendingUp className="h-5 w-5 text-muted-foreground" />}
      title={t.analytics.dailyBreakdown}
    >
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-muted-foreground text-xs">
              <SortHeader label={t.analytics.date} col="day" sortKey={sortKey} sortDir={sortDir} toggle={toggle} className="text-left py-2 pr-4 font-medium" />
              <SortHeader label={t.sessions.title} col="sessions" sortKey={sortKey} sortDir={sortDir} toggle={toggle} className="text-right py-2 px-4 font-medium" />
              <SortHeader label={t.analytics.input} col="input_tokens" sortKey={sortKey} sortDir={sortDir} toggle={toggle} className="text-right py-2 px-4 font-medium" />
              <SortHeader label={t.analytics.output} col="output_tokens" sortKey={sortKey} sortDir={sortDir} toggle={toggle} className="text-right py-2 pl-4 font-medium" />
            </tr>
          </thead>
          <tbody>
            {pageItems.map((d) => (
              <tr
                  key={d.day}
                  className="border-b border-border/50 hover:bg-secondary/20 transition-colors"
                >
                <td className="py-2 pr-4 font-medium">
                    {formatDate(d.day)}
                </td>
                <td className="text-right py-2 px-4 text-muted-foreground">
                    {d.sessions}
                </td>
                <td className="text-right py-2 px-4">
                    <span className="text-[var(--midground)]">
                        {formatTokens(d.input_tokens)}
                    </span>
                </td>
                <td className="text-right py-2 pl-4">
                    <span className="text-[var(--color-success)]">
                        {formatTokens(d.output_tokens)}
                    </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <PaginationBar
        total={sorted.length}
        start={start}
        end={end}
        totalPages={totalPages}
        page={page}
        onPage={setPage}
      />
    </CollapsibleCard>
  );
}

function ModelTable({ models }: { models: AnalyticsModelEntry[] }) {
  const { t } = useI18n();
  const { sorted, sortKey, sortDir, toggle } = useTableSort(models, "input_tokens", "desc");
  const { page, setPage, reset, paginate } = usePagination();

  useEffect(() => { reset(); }, [models, reset]);

  if (models.length === 0) return null;

  const { pageItems, start, end, totalPages } = paginate(sorted);

  return (
    <CollapsibleCard
      icon={<Cpu className="h-5 w-5 text-muted-foreground" />}
      title={t.analytics.perModelBreakdown}
    >
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-muted-foreground text-xs">
              <SortHeader label={t.analytics.model} col="model" sortKey={sortKey} sortDir={sortDir} toggle={toggle} className="text-left py-2 pr-4 font-medium" />
              <SortHeader label={t.sessions.title} col="sessions" sortKey={sortKey} sortDir={sortDir} toggle={toggle} className="text-right py-2 px-4 font-medium" />
              <SortHeader label={t.analytics.tokens} col="input_tokens" sortKey={sortKey} sortDir={sortDir} toggle={toggle} className="text-right py-2 pl-4 font-medium" />
            </tr>
          </thead>
          <tbody>
            {pageItems.map((m) => (
              <tr
                key={m.model}
                className="border-b border-border/50 hover:bg-secondary/20 transition-colors"
              >
                <td className="py-2 pr-4">
                    <span className="font-mono-ui text-xs">{m.model}</span>
                </td>
                <td className="text-right py-2 px-4 text-muted-foreground">
                    {m.sessions}
                </td>
                <td className="text-right py-2 pl-4">
                    <span className="text-[var(--midground)]">
                        {formatTokens(m.input_tokens)}
                    </span>
                    {" / "}
                    <span className="text-[var(--color-success)]">
                        {formatTokens(m.output_tokens)}
                    </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <PaginationBar
        total={sorted.length}
        start={start}
        end={end}
        totalPages={totalPages}
        page={page}
        onPage={setPage}
      />
    </CollapsibleCard>
  );
}

function SkillTable({ skills }: { skills: AnalyticsSkillEntry[] }) {
  const { t } = useI18n();
  const { sorted, sortKey, sortDir, toggle } = useTableSort(skills, "total_count", "desc");
  const { page, setPage, reset, paginate } = usePagination();

  useEffect(() => { reset(); }, [skills, reset]);

  if (skills.length === 0) return null;

  const { pageItems, start, end, totalPages } = paginate(sorted);

  return (
    <CollapsibleCard
      icon={<Brain className="h-5 w-5 text-muted-foreground" />}
      title={t.analytics.topSkills}
    >
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-muted-foreground text-xs">
              <SortHeader label={t.analytics.skill} col="skill" sortKey={sortKey} sortDir={sortDir} toggle={toggle} className="text-left py-2 pr-4 font-medium" />
              <SortHeader label={t.analytics.loads} col="view_count" sortKey={sortKey} sortDir={sortDir} toggle={toggle} className="text-right py-2 px-4 font-medium" />
              <SortHeader label={t.analytics.edits} col="manage_count" sortKey={sortKey} sortDir={sortDir} toggle={toggle} className="text-right py-2 px-4 font-medium" />
              <SortHeader label={t.analytics.total} col="total_count" sortKey={sortKey} sortDir={sortDir} toggle={toggle} className="text-right py-2 px-4 font-medium" />
              <SortHeader label={t.analytics.lastUsed} col="last_used_at" sortKey={sortKey} sortDir={sortDir} toggle={toggle} className="text-right py-2 pl-4 font-medium" />
            </tr>
          </thead>
          <tbody>
            {pageItems.map((skill) => (
              <tr
                key={skill.skill}
                className="border-b border-border/50 hover:bg-secondary/20 transition-colors"
              >
                <td className="py-2 pr-4">
                    <span className="font-mono-ui text-xs">{skill.skill}</span>
                </td>
                <td className="text-right py-2 px-4 text-muted-foreground">
                    {skill.view_count}
                </td>
                <td className="text-right py-2 px-4 text-muted-foreground">
                    {skill.manage_count}
                </td>
                <td className="text-right py-2 px-4">{skill.total_count}</td>
                <td className="text-right py-2 pl-4 text-muted-foreground">
                    {skill.last_used_at ? timeAgo(skill.last_used_at) : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <PaginationBar
        total={sorted.length}
        start={start}
        end={end}
        totalPages={totalPages}
        page={page}
        onPage={setPage}
      />
    </CollapsibleCard>
  );
}

// ---------------------------------------------------------------------------
// Skill Activity Panel — usage stats + enable/disable management
// ---------------------------------------------------------------------------

function SkillActivityPanel({ days }: { days: number }) {
  const { t } = useI18n();
  const [data, setData] = useState<SkillActivityResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [toggling, setToggling] = useState<string | null>(null);
  const [cleanupPreview, setCleanupPreview] = useState<{ count: number; names: string[] } | null>(null);
  const [cleanupLoading, setCleanupLoading] = useState(false);
  const { page, setPage, reset, paginate } = usePagination(DEFAULT_PAGE_SIZE);

  const load = useCallback(() => {
    setLoading(true);
    api.getSkillActivity(days)
      .then(setData)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [days]);

  useEffect(() => { load(); }, [load]);
  useEffect(() => { reset(); }, [days, reset]);

  const handleToggle = async (name: string, enabled: boolean) => {
    setToggling(name);
    try {
      await api.toggleSkill(name, enabled);
      setData((prev) => prev ? {
        ...prev,
        skills: prev.skills.map((s) => s.name === name ? { ...s, enabled } : s),
        summary: {
          ...prev.summary,
          enabled_count: prev.summary.enabled_count + (enabled ? 1 : -1),
          disabled_count: prev.summary.disabled_count + (enabled ? -1 : 1),
        },
      } : null);
    } catch { /* ignore */ }
    setToggling(null);
  };

  const handleCleanupPreview = async () => {
    setCleanupLoading(true);
    try {
      const res = await api.autoCleanupSkills(days, true);
      setCleanupPreview({ count: res.count, names: res.would_disable ?? [] });
    } catch { /* ignore */ }
    setCleanupLoading(false);
  };

  const handleCleanupConfirm = async () => {
    setCleanupLoading(true);
    try {
      await api.autoCleanupSkills(days, false);
      setCleanupPreview(null);
      load();
    } catch { /* ignore */ }
    setCleanupLoading(false);
  };

  const statusBadge = (status: string) => {
    const colors: Record<string, string> = {
      active: "bg-[var(--color-success)]/15 text-[var(--color-success)]",
      idle: "bg-[var(--color-warning)]/15 text-[var(--color-warning)]",
      never_used: "bg-muted text-muted-foreground",
    };
    const labels: Record<string, string> = {
      active: t.analytics.skillActive,
      idle: t.analytics.skillIdle,
      never_used: t.analytics.skillNeverUsed,
    };
    return <Badge tone="secondary" className={`text-[10px] px-1.5 ${colors[status] ?? ""}`}>{labels[status] ?? status}</Badge>;
  };

  if (loading && !data) return <Spinner className="mx-auto my-8 text-primary" />;
  if (!data) return null;

  const { summary, skills } = data;
  const { pageItems, start, end, totalPages } = paginate(skills);

  return (
    <CollapsibleCard
      icon={<Brain className="h-5 w-5 text-muted-foreground" />}
      title={t.analytics.skillActivity}
      defaultOpen={true}
      headerExtra={
        <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
          {summary.never_used_count > 0 && (
            <>
              {cleanupPreview ? (
                <div className="flex items-center gap-1.5">
                  <span className="text-xs text-muted-foreground">
                    {t.analytics.skillCleanupPreview}: {cleanupPreview.count}
                  </span>
                  <Button size="sm" onClick={handleCleanupConfirm} disabled={cleanupLoading} prefix={cleanupLoading ? <Spinner /> : undefined}>
                    {t.analytics.skillCleanupConfirm}
                  </Button>
                  <Button size="sm" ghost onClick={() => setCleanupPreview(null)}>✕</Button>
                </div>
              ) : (
                <Button size="sm" outlined onClick={handleCleanupPreview} disabled={cleanupLoading} prefix={cleanupLoading ? <Spinner /> : undefined}>
                  {t.analytics.skillCleanup}
                </Button>
              )}
            </>
          )}
          <Button size="sm" outlined onClick={load} prefix={loading ? <Spinner /> : <RefreshCw />}>
            {t.common.refresh}
          </Button>
        </div>
      }
    >
      {/* Summary pills */}
      <div className="flex flex-wrap gap-2 mb-4">
        {[
          { label: t.analytics.skillActive, count: summary.active_count, color: "bg-[var(--color-success)]/15 text-[var(--color-success)]" },
          { label: t.analytics.skillIdle, count: summary.idle_count, color: "bg-[var(--color-warning)]/15 text-[var(--color-warning)]" },
          { label: t.analytics.skillNeverUsed, count: summary.never_used_count, color: "bg-muted text-muted-foreground" },
          { label: t.analytics.skillEnabled, count: summary.enabled_count, color: "bg-primary/15 text-primary" },
          { label: "Disabled", count: summary.disabled_count, color: "bg-[var(--color-destructive)]/15 text-[var(--color-destructive)]" },
        ].map((pill) => (
          <span key={pill.label} className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-medium ${pill.color}`}>
            {pill.count} {pill.label}
          </span>
        ))}
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-muted-foreground text-xs">
              <th className="text-left py-2 pr-2 font-medium">{t.analytics.skillEnabled}</th>
              <th className="text-left py-2 pr-2 font-medium">{t.analytics.skill}</th>
              <th className="text-left py-2 pr-2 font-medium">{t.analytics.skillStatus}</th>
              <th className="text-right py-2 px-2 font-medium">{t.analytics.loads}</th>
              <th className="text-right py-2 px-2 font-medium">{t.analytics.edits}</th>
              <th className="text-right py-2 px-2 font-medium">{t.analytics.total}</th>
              <th className="text-right py-2 pl-2 font-medium">{t.analytics.lastUsed}</th>
            </tr>
          </thead>
          <tbody>
            {pageItems.map((s) => (
              <tr key={s.name} className="border-b border-border/50 hover:bg-secondary/20 transition-colors">
                <td className="py-2 pr-2">
                  <button
                    onClick={() => handleToggle(s.name, !s.enabled)}
                    disabled={toggling === s.name}
                    className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer items-center rounded-full transition-colors ${
                      s.enabled
                        ? "bg-primary"
                        : "bg-muted-foreground/30"
                    } ${toggling === s.name ? "opacity-50" : ""}`}
                  >
                    <span className={`inline-block h-3.5 w-3.5 rounded-full bg-card-foreground shadow-sm transition-transform ${s.enabled ? "translate-x-4.5" : "translate-x-0.5"}`} />
                  </button>
                </td>
                <td className="py-2 pr-2">
                  <span className="font-mono-ui text-xs">{s.name}</span>
                </td>
                <td className="py-2 pr-2">{statusBadge(s.status)}</td>
                <td className="text-right py-2 px-2 text-muted-foreground">{s.view_count}</td>
                <td className="text-right py-2 px-2 text-muted-foreground">{s.manage_count}</td>
                <td className="text-right py-2 px-2 font-medium">{s.total_count}</td>
                <td className="text-right py-2 pl-2 text-muted-foreground">
                  {s.last_used_at ? timeAgo(s.last_used_at) : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <PaginationBar
        total={skills.length}
        start={start}
        end={end}
        totalPages={totalPages}
        page={page}
        onPage={setPage}
      />
    </CollapsibleCard>
  );
}

export default function AnalyticsPage() {
  const [days, setDays] = useState(30);
  const [data, setData] = useState<AnalyticsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { t } = useI18n();
  const { setAfterTitle, setEnd } = usePageHeader();

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    api
      .getAnalytics(days)
      .then(setData)
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, [days]);

  useLayoutEffect(() => {
    const periodLabel =
      PERIODS.find((p) => p.days === days)?.label ?? `${days}d`;
    setAfterTitle(
      <span className="flex items-center gap-2">
        {loading && <Spinner className="shrink-0 text-base text-primary" />}
        <Badge tone="secondary" className="text-[10px]">
          {periodLabel}
        </Badge>
      </span>,
    );
    setEnd(
      <div className="flex w-full min-w-0 flex-wrap items-center justify-end gap-2 sm:gap-2">
        <div className="flex flex-wrap items-center gap-1.5">
          {PERIODS.map((p) => (
            <Button
              key={p.label}
              type="button"
              size="sm"
              outlined={days !== p.days}
              onClick={() => setDays(p.days)}
            >
              {p.label}
            </Button>
          ))}
        </div>
        <Button
          type="button"
          size="sm"
          outlined
          onClick={load}
          disabled={loading}
          prefix={loading ? <Spinner /> : <RefreshCw />}
        >
          {t.common.refresh}
        </Button>
      </div>,
    );
    return () => {
      setAfterTitle(null);
      setEnd(null);
    };
  }, [days, loading, load, setAfterTitle, setEnd, t.common.refresh]);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div className="flex flex-col gap-6">
      <PluginSlot name="analytics:top" />
      {loading && !data && (
        <div className="flex items-center justify-center py-24">
          <Spinner className="text-2xl text-primary" />
        </div>
      )}

      {error && (
        <Card>
          <CardContent className="py-6">
            <p className="text-sm text-destructive text-center">{error}</p>
          </CardContent>
        </Card>
      )}

      {data && (
        <>
          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardContent className="py-6">
                <Stats
                  items={[
                    {
                      label: t.analytics.totalTokens,
                      value: formatTokens(
                        data.totals.total_input + data.totals.total_output,
                      ),
                    },
                    {
                      label: t.analytics.input,
                      value: formatTokens(data.totals.total_input),
                    },
                    {
                      label: t.analytics.output,
                      value: formatTokens(data.totals.total_output),
                    },
                    {
                      label: t.analytics.totalSessions,
                      value: `${data.totals.total_sessions} (~${(data.totals.total_sessions / days).toFixed(1)}${t.analytics.perDayAvg})`,
                    },
                    {
                      label: t.analytics.apiCalls,
                      value: String(
                        data.totals.total_api_calls ??
                          data.daily.reduce((sum, d) => sum + d.sessions, 0),
                      ),
                    },
                  ]}
                />
              </CardContent>
            </Card>

            <TokenBarChart daily={data.daily} />
          </div>

          <DailyTable daily={data.daily} />
          <ModelTable models={data.by_model} />
          <SkillTable skills={data.skills.top_skills} />
          <SkillActivityPanel days={days} />
        </>
      )}

      {data &&
        data.daily.length === 0 &&
        data.by_model.length === 0 &&
        data.skills.top_skills.length === 0 && (
          <Card>
            <CardContent className="py-12">
              <div className="flex flex-col items-center text-muted-foreground">
                <BarChart3 className="h-8 w-8 mb-3 opacity-40" />
                <p className="text-sm font-medium">{t.analytics.noUsageData}</p>
                <p className="text-xs mt-1 text-muted-foreground/60">
                  {t.analytics.startSession}
                </p>
              </div>
            </CardContent>
          </Card>
        )}
      <PluginSlot name="analytics:bottom" />
    </div>
  );
}
