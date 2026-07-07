import { useCallback, useEffect, useLayoutEffect, useMemo, useState } from "react";
import {
  Database,
  ExternalLink,
  FileText,
  MessageSquare,
  RefreshCw,
  Search,
  Video,
} from "lucide-react";
import { api, type TeamsContextItem, type TeamsContextSource, type TeamsContextSourceType } from "@/lib/api";
import { cn, isoTimeAgo } from "@/lib/utils";
import { usePageHeader } from "@/contexts/usePageHeader";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { Input } from "@nous-research/ui/ui/components/input";
import { Spinner } from "@nous-research/ui/ui/components/spinner";

const SOURCE_TYPES: Array<{ value: "" | TeamsContextSourceType; label: string }> = [
  { value: "", label: "All" },
  { value: "channel", label: "Channels" },
  { value: "meeting", label: "Meetings" },
  { value: "recording", label: "Recordings" },
  { value: "transcript", label: "Transcripts" },
];

function displayText(item: TeamsContextItem): string {
  const raw = item.text || item.html || "";
  if (!raw.includes("<")) return raw;
  const doc = new DOMParser().parseFromString(raw, "text/html");
  return doc.body.textContent?.trim() || raw;
}

function formatTime(value: string | null): string {
  if (!value) return "No timestamp";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function sourceMeta(source: TeamsContextSource): string {
  const latest = source.latest_at ? isoTimeAgo(source.latest_at) : "unknown";
  return `${source.item_count} items - ${latest}`;
}

function SourceIcon({ type }: { type: TeamsContextSourceType }) {
  if (type === "recording") return <Video className="h-4 w-4" />;
  if (type === "transcript") return <FileText className="h-4 w-4" />;
  if (type === "meeting") return <Database className="h-4 w-4" />;
  return <MessageSquare className="h-4 w-4" />;
}

export default function TeamsContextPage() {
  const [sources, setSources] = useState<TeamsContextSource[]>([]);
  const [items, setItems] = useState<TeamsContextItem[]>([]);
  const [selectedSource, setSelectedSource] = useState("");
  const [selectedType, setSelectedType] = useState<"" | TeamsContextSourceType>("");
  const [query, setQuery] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [total, setTotal] = useState(0);
  const [storePath, setStorePath] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { setAfterTitle, setEnd, setTitle } = usePageHeader();

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    api
      .getTeamsContext({
        sourceId: selectedSource || undefined,
        sourceType: selectedType || undefined,
        q: submittedQuery || undefined,
        limit: 200,
      })
      .then((resp) => {
        setSources(resp.sources);
        setItems(resp.items);
        setTotal(resp.total);
        setStorePath(resp.store_path);
      })
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, [selectedSource, selectedType, submittedQuery]);

  useEffect(() => {
    load();
  }, [load]);

  useLayoutEffect(() => {
    setTitle("Teams KB");
    setAfterTitle(
      <span className="flex items-center gap-1.5">
        <Badge tone="secondary" className="text-xs">
          {total} shown
        </Badge>
        <Button
          type="button"
          ghost
          size="icon"
          className="text-muted-foreground hover:text-foreground"
          onClick={load}
          disabled={loading}
          aria-label="Refresh Teams context"
        >
          {loading ? <Spinner /> : <RefreshCw />}
        </Button>
      </span>,
    );
    setEnd(null);
    return () => {
      setTitle(null);
      setAfterTitle(null);
      setEnd(null);
    };
  }, [load, loading, setAfterTitle, setEnd, setTitle, total]);

  const selectedLabel = useMemo(() => {
    if (!selectedSource) return selectedType ? SOURCE_TYPES.find((item) => item.value === selectedType)?.label || "Filtered" : "All sources";
    return sources.find((source) => source.source_id === selectedSource)?.label || selectedSource;
  }, [selectedSource, selectedType, sources]);

  return (
    <div className="grid min-h-0 min-w-0 grid-cols-1 gap-4 xl:grid-cols-[22rem_minmax(0,1fr)]">
      <aside className="flex min-w-0 flex-col gap-4">
        <Card className="min-w-0 overflow-hidden">
          <CardHeader className="px-4 py-3">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Database className="h-4 w-4" />
              Sources
            </CardTitle>
          </CardHeader>
          <CardContent className="flex min-w-0 flex-col gap-2 p-3">
            <div className="grid min-w-0 grid-cols-2 gap-2">
              {SOURCE_TYPES.map((type) => (
                <button
                  key={type.value || "all"}
                  type="button"
                  onClick={() => {
                    setSelectedType(type.value);
                    setSelectedSource("");
                  }}
                  className={cn(
                    "border border-border px-2 py-1.5 text-left text-xs hover:bg-secondary/60",
                    selectedType === type.value && "border-primary/60 bg-primary/10",
                  )}
                >
                  {type.label}
                </button>
              ))}
            </div>
            <button
              type="button"
              onClick={() => setSelectedSource("")}
              className={cn(
                "mt-2 flex min-w-0 flex-col border border-border px-3 py-2 text-left hover:bg-secondary/60",
                !selectedSource && "border-primary/60 bg-primary/10",
              )}
            >
              <span className="truncate text-sm text-text-primary">All matching sources</span>
              <span className="text-xs text-text-secondary">{total} matching items</span>
            </button>
            {sources
              .filter((source) => !selectedType || source.source_type === selectedType)
              .map((source) => (
                <button
                  key={`${source.source_type}:${source.source_id}`}
                  type="button"
                  onClick={() => setSelectedSource(source.source_id)}
                  className={cn(
                    "flex min-w-0 gap-2 border border-border px-3 py-2 text-left hover:bg-secondary/60",
                    selectedSource === source.source_id && "border-primary/60 bg-primary/10",
                  )}
                >
                  <span className="mt-0.5 shrink-0 text-text-secondary">
                    <SourceIcon type={source.source_type} />
                  </span>
                  <span className="min-w-0">
                    <span className="block truncate text-sm text-text-primary">{source.label}</span>
                    <span className="block text-xs text-text-secondary">{sourceMeta(source)}</span>
                  </span>
                </button>
              ))}
          </CardContent>
        </Card>

        <Card className="min-w-0 overflow-hidden">
          <CardHeader className="px-4 py-3">
            <CardTitle className="text-sm">Store</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 px-4 pb-4">
            <p className="break-all font-mono-ui text-xs text-text-secondary">
              {storePath || "No Teams context store found"}
            </p>
          </CardContent>
        </Card>
      </aside>

      <main className="flex min-h-0 min-w-0 flex-col gap-4">
        <form
          className="flex min-w-0 flex-col gap-2 sm:flex-row"
          onSubmit={(event) => {
            event.preventDefault();
            setSubmittedQuery(query.trim());
          }}
        >
          <div className="relative min-w-0 flex-1">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-tertiary" />
            <Input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search Teams KB"
              className="pl-9"
            />
          </div>
          <Button type="submit" className="shrink-0">
            Search
          </Button>
          {submittedQuery && (
            <Button
              type="button"
              ghost
              className="shrink-0"
              onClick={() => {
                setQuery("");
                setSubmittedQuery("");
              }}
            >
              Clear
            </Button>
          )}
        </form>

        <div className="flex min-w-0 items-center justify-between gap-3">
          <div className="min-w-0">
            <h2 className="truncate text-base font-semibold text-text-primary">{selectedLabel}</h2>
            <p className="text-sm text-text-secondary">
              {items.length} loaded{total > items.length ? ` of ${total}` : ""}
            </p>
          </div>
          {submittedQuery && (
            <Badge tone="secondary" className="max-w-[50%] truncate text-xs">
              {submittedQuery}
            </Badge>
          )}
        </div>

        {error && (
          <div className="border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
            {error}
          </div>
        )}

        <div className="flex min-w-0 flex-col gap-3">
          {loading && items.length === 0 ? (
            <div className="flex items-center gap-2 text-sm text-text-secondary">
              <Spinner />
              Loading Teams KB
            </div>
          ) : items.length === 0 ? (
            <Card>
              <CardContent className="flex items-center gap-3 p-4 text-sm text-text-secondary">
                <Database className="h-4 w-4" />
                No Teams KB items found
              </CardContent>
            </Card>
          ) : (
            items.map((item) => (
              <Card key={`${item.source_type}:${item.source_id}:${item.item_id}`} className="min-w-0 overflow-hidden">
                <CardHeader className="flex-row items-start justify-between gap-3 px-4 py-3">
                  <div className="min-w-0">
                    <CardTitle className="truncate text-sm">
                      {item.sender_name || item.source_label || "TeamContext"}
                    </CardTitle>
                    <p className="mt-1 truncate text-xs text-text-secondary">
                      {item.source_label} - {formatTime(item.created_at || item.ingested_at)}
                    </p>
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    <Badge tone="secondary" className="text-xs">
                      {item.source_type}
                    </Badge>
                    {item.chunk_index !== null && (
                      <Badge tone="secondary" className="text-xs">
                        chunk {item.chunk_index + 1}
                      </Badge>
                    )}
                    {item.web_url && (
                      <a href={item.web_url} target="_blank" rel="noreferrer" className="text-text-secondary hover:text-text-primary" aria-label="Open source">
                        <ExternalLink className="h-4 w-4" />
                      </a>
                    )}
                  </div>
                </CardHeader>
                <CardContent className="px-4 pb-4 pt-0">
                  <p className="whitespace-pre-wrap break-words text-sm leading-6 text-text-primary">
                    {displayText(item)}
                  </p>
                </CardContent>
              </Card>
            ))
          )}
        </div>
      </main>
    </div>
  );
}
