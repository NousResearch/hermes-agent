/* eslint-disable react-hooks/set-state-in-effect */
import { useCallback, useEffect, useMemo, useState, type ComponentType } from "react";
import {
  AlertTriangle,
  Archive,
  BarChart3,
  CalendarDays,
  CheckCircle2,
  Clock3,
  Download,
  Eye,
  FileText,
  FileVideo,
  Gauge,
  Link2,
  ListChecks,
  Loader2,
  LockKeyhole,
  PlaySquare,
  Plus,
  RefreshCw,
  Search,
  ShieldCheck,
  UploadCloud,
  UserCheck,
} from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { usePageHeader } from "@/contexts/usePageHeader";
import { api } from "@/lib/api";
import type {
  YouTubeChannelConfig,
  YouTubeChannelId,
  YouTubeDashboardResponse,
  YouTubeManifestImportResponse,
  YouTubePublishPlan,
  YouTubeQueueFormat,
  YouTubeQueueItem,
  YouTubeQueueStatus,
  YouTubeReviewStatus,
  YouTubeRisk,
} from "@/lib/api";

type ChannelFilter = YouTubeChannelId | "all";
type ManifestFormat = "csv" | "json";

const STATUS_LABELS: Record<YouTubeQueueStatus, string> = {
  idea: "Idea",
  metadata: "Metadata",
  assets: "Assets",
  review: "Review",
  ready: "Ready",
  scheduled_local: "Scheduled local",
  published_manual: "Published manual",
  archived: "Archived",
};

const STATUS_ORDER: YouTubeQueueStatus[] = [
  "idea",
  "metadata",
  "assets",
  "review",
  "ready",
  "scheduled_local",
  "published_manual",
  "archived",
];

const FIELD_LABELS = [
  "Video file",
  "Thumbnail",
  "Title",
  "Description",
  "Tags",
  "Playlist",
  "Visibility",
  "Schedule",
  "Pinned comment",
  "Sources / scripture refs",
];

const FORMAT_LABELS: Record<YouTubeQueueFormat, string> = {
  short: "Short",
  long_form: "Long-form",
  clip: "Clip",
};

const REVIEW_LABELS: Record<YouTubeReviewStatus, string> = {
  needs_review: "Needs review",
  approved: "Approved",
  changes_requested: "Changes requested",
  rejected: "Rejected",
};

const REVIEW_ORDER: YouTubeReviewStatus[] = ["needs_review", "approved", "changes_requested", "rejected"];

const REQUIRED_CHECK_LABELS: Record<string, string> = {
  video_file: "Video file",
  thumbnail: "Thumbnail",
  title: "Title",
  description: "Description",
  captions: "Captions",
  sources_or_scripture_refs: "Sources / refs",
  fact_check: "Fact check",
  human_approval: "Approval",
};

const DEFAULT_FORM = {
  channel_id: "scripturedepth" as YouTubeChannelId,
  title: "",
  format: "short" as YouTubeQueueFormat,
  owner: "Hermes",
};

function channelName(channels: YouTubeChannelConfig[], id: YouTubeChannelId) {
  return channels.find((channel) => channel.id === id)?.name ?? id;
}

function riskTone(risk: YouTubeRisk) {
  if (risk === "high") return "border-red-500/40 bg-red-500/10 text-red-200";
  if (risk === "medium") return "border-amber-500/40 bg-amber-500/10 text-amber-100";
  return "border-emerald-500/40 bg-emerald-500/10 text-emerald-100";
}

function statusTone(status: YouTubeQueueStatus) {
  if (status === "ready" || status === "scheduled_local" || status === "published_manual") {
    return "border-emerald-500/40 bg-emerald-500/10 text-emerald-100";
  }
  if (status === "review") return "border-amber-500/40 bg-amber-500/10 text-amber-100";
  if (status === "archived") return "border-slate-500/40 bg-slate-500/10 text-slate-300";
  return "border-border bg-muted text-muted-foreground";
}

function reviewTone(status: YouTubeReviewStatus) {
  if (status === "approved") return "border-emerald-500/40 bg-emerald-500/10 text-emerald-100";
  if (status === "changes_requested") return "border-amber-500/40 bg-amber-500/10 text-amber-100";
  if (status === "rejected") return "border-red-500/40 bg-red-500/10 text-red-200";
  return "border-border bg-muted text-muted-foreground";
}

function channelTint(id: YouTubeChannelId) {
  return id === "scripturedepth" ? "from-amber-500/20 to-stone-500/10" : "from-sky-500/20 to-slate-500/10";
}

function StatCard({
  icon: Icon,
  label,
  value,
  detail,
}: {
  icon: ComponentType<{ className?: string }>;
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <Card>
      <CardContent className="p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">{label}</p>
            <p className="mt-2 text-2xl font-semibold text-foreground">{value}</p>
            <p className="mt-1 text-sm text-muted-foreground">{detail}</p>
          </div>
          <div className="rounded-lg border border-border bg-muted/40 p-2">
            <Icon className="h-5 w-5 text-muted-foreground" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function emptyDashboard(): YouTubeDashboardResponse {
  return {
    channels: [],
    items: [],
    summary: { total: 0, ready: 0, blocked: 0, needs_approval: 0, review_changes_requested: 0, review_rejected: 0, archived: 0 },
    capabilities: { local_queue: true, youtube_publish: false, youtube_analytics: false },
    updated_at: "",
    schema_version: 1,
  };
}

function downloadText(filename: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function renderPublishValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  if (Array.isArray(value)) return value.length ? value.join(", ") : "—";
  if (typeof value === "boolean") return value ? "true" : "false";
  return String(value);
}

function YouTubeDashboardPage() {
  usePageHeader();
  const [data, setData] = useState<YouTubeDashboardResponse>(emptyDashboard);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedChannel, setSelectedChannel] = useState<ChannelFilter>("all");
  const [showArchived, setShowArchived] = useState(false);
  const [query, setQuery] = useState("");
  const [form, setForm] = useState(DEFAULT_FORM);
  const [manifestFormat, setManifestFormat] = useState<ManifestFormat>("csv");
  const [manifestText, setManifestText] = useState("");
  const [importResult, setImportResult] = useState<YouTubeManifestImportResponse | null>(null);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [focusedItemId, setFocusedItemId] = useState<string | null>(null);
  const [bulkOwner, setBulkOwner] = useState("Hermes");
  const [bulkStatus, setBulkStatus] = useState<YouTubeQueueStatus>("review");
  const [publishPlan, setPublishPlan] = useState<YouTubePublishPlan | null>(null);
  const [publishPlanLoading, setPublishPlanLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await api.getYouTubeDashboard());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const filteredQueue = useMemo(() => {
    const q = query.trim().toLowerCase();
    return data.items.filter((item) => {
      if (!showArchived && item.status === "archived") return false;
      const channelMatch = selectedChannel === "all" || item.channel_id === selectedChannel;
      const queryMatch =
        !q ||
        `${item.title} ${item.owner} ${channelName(data.channels, item.channel_id)} ${item.tags.join(" ")}`
          .toLowerCase()
          .includes(q);
      return channelMatch && queryMatch;
    });
  }, [data.channels, data.items, query, selectedChannel, showArchived]);

  const selectedItems = useMemo(
    () => data.items.filter((item) => selectedIds.includes(item.id)),
    [data.items, selectedIds],
  );
  const focusedItem = useMemo(() => {
    if (focusedItemId) {
      return data.items.find((item) => item.id === focusedItemId) ?? null;
    }
    return selectedItems[0] ?? filteredQueue[0] ?? null;
  }, [data.items, filteredQueue, focusedItemId, selectedItems]);
  const visibleIds = filteredQueue.map((item) => item.id);
  const allVisibleSelected = visibleIds.length > 0 && visibleIds.every((id) => selectedIds.includes(id));

  const refreshPublishPlan = useCallback(async (itemId: string | null) => {
    if (!itemId) {
      setPublishPlan(null);
      return;
    }
    setPublishPlanLoading(true);
    try {
      setPublishPlan(await api.getYouTubePublishPlan(itemId));
    } catch (err) {
      setPublishPlan(null);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setPublishPlanLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshPublishPlan(focusedItem?.id ?? null);
  }, [focusedItem?.id, focusedItem?.updated_at, refreshPublishPlan]);

  async function createItem() {
    const title = form.title.trim();
    if (!title) {
      setError("Title is required");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      await api.createYouTubeQueueItem({
        channel_id: form.channel_id,
        title,
        format: form.format,
        owner: form.owner || "Hermes",
        checks: { title: true },
      });
      setForm(DEFAULT_FORM);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  async function patchItem(id: string, updates: Partial<YouTubeQueueItem>) {
    setSaving(true);
    setError(null);
    try {
      await api.patchYouTubeQueueItem(id, updates);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  async function archiveItem(id: string) {
    setSaving(true);
    setError(null);
    try {
      await api.archiveYouTubeQueueItem(id);
      setSelectedIds((current) => current.filter((selectedId) => selectedId !== id));
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  function toggleSelection(id: string, checked: boolean) {
    setSelectedIds((current) => checked ? Array.from(new Set([...current, id])) : current.filter((selectedId) => selectedId !== id));
    if (checked) setFocusedItemId(id);
  }

  function toggleAllVisible(checked: boolean) {
    setSelectedIds((current) => checked ? Array.from(new Set([...current, ...visibleIds])) : current.filter((id) => !visibleIds.includes(id)));
  }

  async function bulkPatch(updates: Partial<YouTubeQueueItem>) {
    if (selectedIds.length === 0) {
      setError("Select at least one queue item first");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const result = await api.bulkPatchYouTubeQueueItems(selectedIds, updates);
      if (result.error_count > 0) {
        setError(`${result.updated_count} updated; ${result.error_count} failed: ${result.errors[0]?.error ?? "unknown error"}`);
      }
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  async function bulkArchive() {
    if (selectedIds.length === 0) {
      setError("Select at least one queue item first");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const result = await api.bulkArchiveYouTubeQueueItems(selectedIds);
      setSelectedIds([]);
      if (result.error_count > 0) {
        setError(`${result.updated_count} archived; ${result.error_count} failed: ${result.errors[0]?.error ?? "unknown error"}`);
      }
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  async function downloadTemplate(format: ManifestFormat) {
    setSaving(true);
    setError(null);
    try {
      const file = await api.getYouTubeManifestTemplate(format);
      downloadText(`youtube-manifest-template.${format}`, file.content, format === "csv" ? "text/csv" : "application/json");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  async function exportManifest(format: ManifestFormat) {
    setSaving(true);
    setError(null);
    try {
      const file = await api.exportYouTubeManifest(format, showArchived);
      downloadText(`youtube-queue-export.${format}`, file.content, format === "csv" ? "text/csv" : "application/json");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  async function importManifest() {
    const content = manifestText.trim();
    if (!content) {
      setError("Paste or upload a CSV/JSON manifest first");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const result = await api.importYouTubeManifest(manifestFormat, content);
      setImportResult(result);
      setManifestText("");
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  async function readManifestFile(file: File | null) {
    if (!file) return;
    const lowered = file.name.toLowerCase();
    setManifestFormat(lowered.endsWith(".json") ? "json" : "csv");
    setManifestText(await file.text());
  }

  return (
    <div className="space-y-6 pb-10">
      <section className="rounded-2xl border border-border bg-card/70 p-6 shadow-sm">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
          <div className="max-w-3xl">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <PlaySquare className="h-4 w-4" />
              Local queue control plane for ScriptureDepth + Newslish
            </div>
            <h1 className="mt-3 text-3xl font-semibold tracking-tight text-foreground">
              Batch YouTube production, review, and scheduling
            </h1>
            <p className="mt-3 text-sm leading-6 text-muted-foreground">
              This slice is intentionally local-state only: queue, metadata, checks, archive, and audit. YouTube upload,
              analytics, and public publishing are disabled until OAuth, quota, audit, and approval gates are explicit.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button outlined size="sm" onClick={() => void downloadTemplate("csv")} disabled={saving}>
              <FileText className="mr-2 h-4 w-4" /> Template CSV
            </Button>
            <Button outlined size="sm" onClick={() => void exportManifest("json")} disabled={saving}>
              <Download className="mr-2 h-4 w-4" /> Export JSON
            </Button>
            <Button outlined size="sm" disabled>
              <BarChart3 className="mr-2 h-4 w-4" /> Analytics later
            </Button>
            <Button size="sm" onClick={load} disabled={loading || saving}>
              {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <RefreshCw className="mr-2 h-4 w-4" />}
              Refresh
            </Button>
          </div>
        </div>
      </section>

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-100">
          {error}
        </div>
      )}

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <StatCard icon={FileVideo} label="Queue" value={`${data.summary.total}`} detail="active draft videos" />
        <StatCard icon={CheckCircle2} label="Ready" value={`${data.summary.ready}`} detail="locally ready after checks" />
        <StatCard icon={AlertTriangle} label="Blocked" value={`${data.summary.blocked}`} detail="missing checks or high risk" />
        <StatCard icon={LockKeyhole} label="Approval gate" value={`${data.summary.needs_approval}`} detail="still needs human approval" />
      </section>

      <section className="grid gap-4 xl:grid-cols-2">
        {data.channels.map((channel) => (
          <Card key={channel.id} className="overflow-hidden">
            <div className={`h-1 bg-gradient-to-r ${channelTint(channel.id)}`} />
            <CardHeader>
              <div className="flex items-start justify-between gap-4">
                <div>
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <PlaySquare className="h-5 w-5 text-muted-foreground" />
                    {channel.name}
                  </CardTitle>
                  <p className="mt-1 text-sm text-muted-foreground">{channel.handle}</p>
                </div>
                <Badge tone="outline">{channel.default_visibility}</Badge>
              </div>
            </CardHeader>
            <CardContent className="grid gap-4 text-sm md:grid-cols-2">
              <div>
                <p className="text-xs uppercase tracking-[0.16em] text-muted-foreground">Cadence</p>
                <p className="mt-1 text-foreground">{channel.cadence}</p>
              </div>
              <div>
                <p className="text-xs uppercase tracking-[0.16em] text-muted-foreground">Playlist</p>
                <p className="mt-1 text-foreground">{channel.playlist}</p>
              </div>
              <div className="md:col-span-2">
                <p className="text-xs uppercase tracking-[0.16em] text-muted-foreground">Voice</p>
                <p className="mt-1 text-foreground">{channel.voice}</p>
              </div>
              <div className="md:col-span-2 rounded-lg border border-amber-500/25 bg-amber-500/10 p-3 text-amber-100">
                {channel.guardrail}
              </div>
            </CardContent>
          </Card>
        ))}
      </section>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Plus className="h-5 w-5 text-muted-foreground" />
            Add queue item
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 lg:grid-cols-[180px_160px_1fr_140px_120px]">
            <select
              value={form.channel_id}
              onChange={(event) => setForm((current) => ({ ...current, channel_id: event.target.value as YouTubeChannelId }))}
              className="h-10 rounded-md border border-border bg-background px-3 text-sm outline-none focus:ring-2 focus:ring-ring"
            >
              {data.channels.map((channel) => (
                <option key={channel.id} value={channel.id}>{channel.name}</option>
              ))}
            </select>
            <select
              value={form.format}
              onChange={(event) => setForm((current) => ({ ...current, format: event.target.value as YouTubeQueueFormat }))}
              className="h-10 rounded-md border border-border bg-background px-3 text-sm outline-none focus:ring-2 focus:ring-ring"
            >
              <option value="short">Short</option>
              <option value="long_form">Long-form</option>
              <option value="clip">Clip</option>
            </select>
            <input
              value={form.title}
              onChange={(event) => setForm((current) => ({ ...current, title: event.target.value }))}
              onKeyDown={(event) => {
                if (event.key === "Enter") void createItem();
              }}
              placeholder="Video title / working hook..."
              className="h-10 rounded-md border border-border bg-background px-3 text-sm outline-none focus:ring-2 focus:ring-ring"
            />
            <input
              value={form.owner}
              onChange={(event) => setForm((current) => ({ ...current, owner: event.target.value }))}
              placeholder="Owner"
              className="h-10 rounded-md border border-border bg-background px-3 text-sm outline-none focus:ring-2 focus:ring-ring"
            />
            <Button onClick={createItem} disabled={saving || loading}>
              {saving ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Plus className="mr-2 h-4 w-4" />}
              Add
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <CardTitle className="flex items-center gap-2 text-lg">
              <UploadCloud className="h-5 w-5 text-muted-foreground" />
              Batch manifest import / export
            </CardTitle>
            <div className="flex flex-wrap gap-2">
              <Button outlined size="sm" onClick={() => void downloadTemplate("csv")} disabled={saving}>
                <FileText className="mr-2 h-4 w-4" /> Template CSV
              </Button>
              <Button outlined size="sm" onClick={() => void downloadTemplate("json")} disabled={saving}>
                <FileText className="mr-2 h-4 w-4" /> Template JSON
              </Button>
              <Button outlined size="sm" onClick={() => void exportManifest("csv")} disabled={saving}>
                <Download className="mr-2 h-4 w-4" /> Export CSV
              </Button>
              <Button outlined size="sm" onClick={() => void exportManifest("json")} disabled={saving}>
                <Download className="mr-2 h-4 w-4" /> Export JSON
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 lg:grid-cols-[140px_1fr_140px]">
            <select
              value={manifestFormat}
              onChange={(event) => setManifestFormat(event.target.value as ManifestFormat)}
              className="h-10 rounded-md border border-border bg-background px-3 text-sm outline-none focus:ring-2 focus:ring-ring"
            >
              <option value="csv">CSV</option>
              <option value="json">JSON</option>
            </select>
            <label className="flex h-10 cursor-pointer items-center justify-center rounded-md border border-dashed border-border bg-muted/20 px-3 text-sm text-muted-foreground hover:bg-muted/40">
              <input
                type="file"
                accept=".csv,.json,text/csv,application/json"
                className="hidden"
                onChange={(event) => void readManifestFile(event.currentTarget.files?.[0] ?? null)}
              />
              Upload manifest file or paste below
            </label>
            <Button onClick={importManifest} disabled={saving || !manifestText.trim()}>
              {saving ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <UploadCloud className="mr-2 h-4 w-4" />}
              Import
            </Button>
          </div>
          <textarea
            value={manifestText}
            onChange={(event) => setManifestText(event.target.value)}
            placeholder="Paste CSV with channel_id,title,format,... or JSON { items: [...] }"
            className="min-h-28 w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-xs outline-none ring-offset-background placeholder:text-muted-foreground focus:ring-2 focus:ring-ring"
          />
          {importResult && (
            <div className="rounded-lg border border-border bg-muted/20 p-3 text-sm">
              <div className="font-medium text-foreground">
                Imported {importResult.created_count} item{importResult.created_count === 1 ? "" : "s"}
                {importResult.error_count > 0 ? ` with ${importResult.error_count} row error${importResult.error_count === 1 ? "" : "s"}` : ""}.
              </div>
              {importResult.errors.length > 0 && (
                <ul className="mt-2 space-y-1 text-xs text-amber-100">
                  {importResult.errors.slice(0, 5).map((rowError) => (
                    <li key={`${rowError.row}-${rowError.error}`}>Row {rowError.row}: {rowError.error}</li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      <section className="grid gap-4 xl:grid-cols-[1.25fr_0.75fr]">
        <Card>
          <CardHeader>
            <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
              <CardTitle className="flex items-center gap-2 text-lg">
                <ListChecks className="h-5 w-5 text-muted-foreground" />
                Publishing queue
              </CardTitle>
              <div className="flex flex-col gap-2 sm:flex-row">
                <label className="flex items-center gap-2 rounded-md border border-border bg-background px-3 text-xs text-muted-foreground">
                  <input type="checkbox" checked={showArchived} onChange={(event) => setShowArchived(event.target.checked)} />
                  Archived
                </label>
                <div className="relative">
                  <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
                  <input
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    placeholder="Search queue..."
                    className="h-9 rounded-md border border-border bg-background pl-9 pr-3 text-sm outline-none ring-offset-background placeholder:text-muted-foreground focus:ring-2 focus:ring-ring"
                  />
                </div>
                <select
                  value={selectedChannel}
                  onChange={(event) => setSelectedChannel(event.target.value as ChannelFilter)}
                  className="h-9 rounded-md border border-border bg-background px-3 text-sm outline-none focus:ring-2 focus:ring-ring"
                >
                  <option value="all">All channels</option>
                  {data.channels.map((channel) => (
                    <option key={channel.id} value={channel.id}>{channel.name}</option>
                  ))}
                </select>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="mb-4 grid gap-3 rounded-xl border border-border bg-muted/20 p-3 lg:grid-cols-[1fr_160px_180px_auto_auto_auto]">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <ListChecks className="h-4 w-4" />
                {selectedIds.length} selected for bulk actions
              </div>
              <select
                value={bulkStatus}
                onChange={(event) => setBulkStatus(event.target.value as YouTubeQueueStatus)}
                className="h-9 rounded-md border border-border bg-background px-3 text-sm outline-none focus:ring-2 focus:ring-ring"
              >
                {STATUS_ORDER.filter((status) => status !== "published_manual").map((status) => (
                  <option key={status} value={status}>{STATUS_LABELS[status]}</option>
                ))}
              </select>
              <input
                value={bulkOwner}
                onChange={(event) => setBulkOwner(event.target.value)}
                placeholder="Bulk owner"
                className="h-9 rounded-md border border-border bg-background px-3 text-sm outline-none focus:ring-2 focus:ring-ring"
              />
              <Button outlined size="sm" onClick={() => void bulkPatch({ status: bulkStatus })} disabled={saving || selectedIds.length === 0}>Set status</Button>
              <Button outlined size="sm" onClick={() => void bulkPatch({ owner: bulkOwner || "Hermes" })} disabled={saving || selectedIds.length === 0}>Assign</Button>
              <Button outlined size="sm" onClick={() => void bulkArchive()} disabled={saving || selectedIds.length === 0}>
                <Archive className="mr-2 h-4 w-4" /> Archive
              </Button>
            </div>
            {loading ? (
              <div className="flex items-center gap-2 rounded-xl border border-border p-6 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" /> Loading YouTube queue...
              </div>
            ) : filteredQueue.length === 0 ? (
              <div className="rounded-xl border border-dashed border-border p-8 text-center text-sm text-muted-foreground">
                <p>No queue items match this view.</p>
                <p className="mt-2">
                  Add a first draft above, download the CSV/JSON template, or paste a prepared manifest into the import box.
                </p>
                <p className="mt-2 text-xs text-amber-100">
                  Public YouTube publishing stays disabled; this queue is local prep and review only.
                </p>
              </div>
            ) : (
              <div className="overflow-hidden rounded-xl border border-border">
                <table className="w-full text-left text-sm">
                  <thead className="border-b border-border bg-muted/40 text-xs uppercase tracking-[0.14em] text-muted-foreground">
                    <tr>
                      <th className="px-4 py-3">
                        <input type="checkbox" checked={allVisibleSelected} onChange={(event) => toggleAllVisible(event.target.checked)} />
                      </th>
                      <th className="px-4 py-3">Video</th>
                      <th className="px-4 py-3">Channel</th>
                      <th className="px-4 py-3">Status</th>
                      <th className="px-4 py-3">Review</th>
                      <th className="px-4 py-3">Checks</th>
                      <th className="px-4 py-3">Risk</th>
                      <th className="px-4 py-3">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredQueue.map((item) => (
                      <tr key={item.id} className="border-b border-border last:border-b-0">
                        <td className="px-4 py-3 align-top">
                          <input
                            type="checkbox"
                            checked={selectedIds.includes(item.id)}
                            onChange={(event) => toggleSelection(item.id, event.target.checked)}
                          />
                        </td>
                        <td className="px-4 py-3 align-top">
                          <div className="font-medium text-foreground">{item.title}</div>
                          <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                            <span>{FORMAT_LABELS[item.format]}</span>
                            <span>•</span>
                            <span>{item.owner}</span>
                            {item.scheduled_for && <><span>•</span><span>{item.scheduled_for}</span></>}
                          </div>
                          {item.missing.length > 0 && (
                            <div className="mt-2 text-xs text-amber-100">Missing: {item.missing.join(", ")}</div>
                          )}
                        </td>
                        <td className="px-4 py-3 align-top text-muted-foreground">{channelName(data.channels, item.channel_id)}</td>
                        <td className="px-4 py-3 align-top">
                          <select
                            value={item.status}
                            onChange={(event) => void patchItem(item.id, { status: event.target.value as YouTubeQueueStatus })}
                            disabled={saving}
                            className="h-8 rounded-md border border-border bg-background px-2 text-xs outline-none focus:ring-2 focus:ring-ring"
                          >
                            {STATUS_ORDER.filter((status) => status !== "published_manual").map((status) => (
                              <option key={status} value={status}>{STATUS_LABELS[status]}</option>
                            ))}
                          </select>
                          <Badge tone="outline" className={`ml-2 ${statusTone(item.status)}`}>{STATUS_LABELS[item.status]}</Badge>
                        </td>
                        <td className="px-4 py-3 align-top">
                          <Badge tone="outline" className={reviewTone(item.review_status)}>{REVIEW_LABELS[item.review_status]}</Badge>
                          {item.reviewer && <div className="mt-1 text-xs text-muted-foreground">{item.reviewer}</div>}
                        </td>
                        <td className="px-4 py-3 align-top">
                          <div className="flex max-w-[220px] flex-wrap gap-1.5">
                            {Object.entries(REQUIRED_CHECK_LABELS).map(([key, label]) => (
                              <button
                                key={key}
                                onClick={() => void patchItem(item.id, { checks: { ...item.checks, [key]: !item.checks[key] } })}
                                disabled={saving}
                                className={`rounded border px-2 py-1 text-[10px] transition-colors ${
                                  item.checks[key]
                                    ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-100"
                                    : "border-border bg-muted/20 text-muted-foreground hover:bg-muted/50"
                                }`}
                              >
                                {label}
                              </button>
                            ))}
                          </div>
                        </td>
                        <td className="px-4 py-3 align-top">
                          <Badge tone="outline" className={riskTone(item.risk)}>{item.risk}</Badge>
                        </td>
                        <td className="px-4 py-3 align-top">
                          <div className="flex flex-wrap gap-2">
                            <Button outlined size="sm" onClick={() => setFocusedItemId(item.id)} disabled={saving}>
                              <Eye className="mr-2 h-4 w-4" /> Review
                            </Button>
                            {item.status !== "archived" && (
                              <Button outlined size="sm" onClick={() => void archiveItem(item.id)} disabled={saving}>
                                <Archive className="mr-2 h-4 w-4" /> Archive
                              </Button>
                            )}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>

        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg">
                <UserCheck className="h-5 w-5 text-muted-foreground" />
                Review workspace
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              {focusedItem ? (
                <>
                  <div>
                    <p className="font-medium text-foreground">{focusedItem.title}</p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {channelName(data.channels, focusedItem.channel_id)} • {FORMAT_LABELS[focusedItem.format]} • {focusedItem.owner}
                    </p>
                  </div>
                  <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2">
                    <select
                      value={focusedItem.review_status}
                      onChange={(event) => void patchItem(focusedItem.id, { review_status: event.target.value as YouTubeReviewStatus })}
                      disabled={saving}
                      className="h-9 rounded-md border border-border bg-background px-3 text-sm outline-none focus:ring-2 focus:ring-ring"
                    >
                      {REVIEW_ORDER.map((status) => (
                        <option key={status} value={status}>{REVIEW_LABELS[status]}</option>
                      ))}
                    </select>
                    <input
                      key={`${focusedItem.id}-reviewer`}
                      defaultValue={focusedItem.reviewer}
                      onBlur={(event) => void patchItem(focusedItem.id, { reviewer: event.currentTarget.value })}
                      placeholder="Reviewer"
                      disabled={saving}
                      className="h-9 rounded-md border border-border bg-background px-3 text-sm outline-none focus:ring-2 focus:ring-ring"
                    />
                  </div>
                  <textarea
                    key={`${focusedItem.id}-review-notes`}
                    defaultValue={focusedItem.review_notes}
                    onBlur={(event) => void patchItem(focusedItem.id, { review_notes: event.currentTarget.value })}
                    placeholder="Review notes / requested changes..."
                    disabled={saving}
                    className="min-h-24 w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none ring-offset-background placeholder:text-muted-foreground focus:ring-2 focus:ring-ring"
                  />
                  <div className="grid gap-2">
                    {Object.entries(REQUIRED_CHECK_LABELS).map(([key, label]) => (
                      <button
                        key={key}
                        onClick={() => void patchItem(focusedItem.id, { checks: { ...focusedItem.checks, [key]: !focusedItem.checks[key] } })}
                        disabled={saving}
                        className={`flex items-center justify-between rounded border px-3 py-2 text-xs transition-colors ${
                          focusedItem.checks[key]
                            ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-100"
                            : "border-border bg-muted/20 text-muted-foreground hover:bg-muted/50"
                        }`}
                      >
                        <span>{label}</span>
                        <span>{focusedItem.checks[key] ? "Done" : "Missing"}</span>
                      </button>
                    ))}
                  </div>
                  <div className="rounded-lg border border-border bg-muted/20 p-3 text-xs text-muted-foreground">
                    Ready is server-gated: all required checks plus approved review. YouTube publish remains disabled.
                  </div>
                </>
              ) : (
                <div className="rounded-xl border border-dashed border-border p-6 text-center text-sm text-muted-foreground">
                  Select or create a queue item to review.
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg">
                <ShieldCheck className="h-5 w-5 text-muted-foreground" />
                Publish plan dry run
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              {focusedItem ? (
                publishPlanLoading ? (
                  <div className="flex items-center gap-2 rounded-xl border border-border p-4 text-sm text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" /> Building publish plan...
                  </div>
                ) : publishPlan ? (
                  <>
                    <div className={`rounded-lg border p-3 ${publishPlan.readiness.ready ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-100" : "border-amber-500/30 bg-amber-500/10 text-amber-100"}`}>
                      <div className="flex items-center justify-between gap-3">
                        <span className="font-medium">{publishPlan.readiness.ready ? "Ready for dry-run handoff" : "Not publish-ready"}</span>
                        <Badge tone="outline">publish disabled</Badge>
                      </div>
                      <p className="mt-2 text-xs opacity-90">{publishPlan.safety_note}</p>
                    </div>

                    {publishPlan.readiness.blockers.length > 0 && (
                      <div>
                        <p className="text-xs uppercase tracking-[0.16em] text-muted-foreground">Blockers</p>
                        <ul className="mt-2 space-y-1 text-xs text-amber-100">
                          {publishPlan.readiness.blockers.map((blocker) => <li key={blocker}>• {blocker}</li>)}
                        </ul>
                      </div>
                    )}

                    {publishPlan.readiness.warnings.length > 0 && (
                      <div>
                        <p className="text-xs uppercase tracking-[0.16em] text-muted-foreground">Warnings</p>
                        <ul className="mt-2 space-y-1 text-xs text-muted-foreground">
                          {publishPlan.readiness.warnings.map((warning) => <li key={warning}>• {warning}</li>)}
                        </ul>
                      </div>
                    )}

                    <div>
                      <p className="text-xs uppercase tracking-[0.16em] text-muted-foreground">Channel rules</p>
                      <div className="mt-2 grid gap-2">
                        {publishPlan.readiness.channel_rule_checks.map((rule) => (
                          <div key={rule.id} className="flex items-center justify-between rounded border border-border bg-muted/20 px-3 py-2 text-xs">
                            <span>{rule.label}</span>
                            <span className={rule.ok ? "text-emerald-100" : "text-amber-100"}>{rule.ok ? "OK" : "Missing"}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div>
                      <p className="text-xs uppercase tracking-[0.16em] text-muted-foreground">Payload preview</p>
                      <div className="mt-2 max-h-80 overflow-auto rounded-lg border border-border bg-background/70">
                        {Object.entries(publishPlan.payload_preview).map(([key, value]) => (
                          <div key={key} className="grid grid-cols-[130px_1fr] border-b border-border px-3 py-2 text-xs last:border-b-0">
                            <span className="font-mono text-muted-foreground">{key}</span>
                            <span className="break-words text-foreground">{renderPublishValue(value)}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    <Button outlined size="sm" onClick={() => void refreshPublishPlan(focusedItem.id)} disabled={publishPlanLoading}>
                      <RefreshCw className="mr-2 h-4 w-4" /> Refresh plan
                    </Button>
                  </>
                ) : (
                  <div className="rounded-xl border border-dashed border-border p-6 text-center text-sm text-muted-foreground">
                    Publish plan unavailable for this item.
                  </div>
                )
              ) : (
                <div className="rounded-xl border border-dashed border-border p-6 text-center text-sm text-muted-foreground">
                  Select a queue item to preview the dry-run YouTube payload.
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg">
                <Gauge className="h-5 w-5 text-muted-foreground" />
                Pipeline lanes
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {STATUS_ORDER.map((status) => {
                const count = data.items.filter((item) => item.status === status).length;
                return (
                  <div key={status} className="flex items-center justify-between rounded-lg border border-border bg-muted/20 px-3 py-2">
                    <span className="text-sm text-foreground">{STATUS_LABELS[status]}</span>
                    <Badge tone="outline">{count}</Badge>
                  </div>
                );
              })}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg">
                <CalendarDays className="h-5 w-5 text-muted-foreground" />
                Batch metadata fields
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2">
                {FIELD_LABELS.map((field) => (
                  <div key={field} className="flex items-center gap-2 rounded-md border border-border bg-background px-3 py-2 text-sm">
                    <CheckCircle2 className="h-4 w-4 text-muted-foreground" />
                    {field}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Link2 className="h-5 w-5 text-muted-foreground" />
              OAuth + permissions
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm leading-6 text-muted-foreground">
            Not implemented yet. Later: channel-scoped OAuth, no shared passwords, and separate submitter/reviewer/publisher roles.
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Clock3 className="h-5 w-5 text-muted-foreground" />
              Schedule safety
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm leading-6 text-muted-foreground">
            Ready/scheduled transitions are server-gated by required checks. Public publish is not available in this slice.
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <ShieldCheck className="h-5 w-5 text-muted-foreground" />
              Local audit trail
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm leading-6 text-muted-foreground">
            Every create, patch, and archive writes to <code>~/.hermes/youtube/audit.jsonl</code>. Boring, useful, hard to regret.
          </CardContent>
        </Card>
      </section>
    </div>
  );
}

export default YouTubeDashboardPage;
