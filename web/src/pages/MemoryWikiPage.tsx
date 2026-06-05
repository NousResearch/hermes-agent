import { useCallback, useEffect, useMemo, useState } from "react";
import { BookOpen, Brain, Database, FileText, Search } from "lucide-react";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Input } from "@nous-research/ui/ui/components/input";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { api, type KnowledgeEntry, type KnowledgeReadResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function formatUpdated(ts?: number | null): string {
  if (!ts) return "sin fecha";
  return new Date(ts * 1000).toLocaleString();
}

function sourceIcon(source: KnowledgeEntry["source"]) {
  return source === "wiki" ? BookOpen : Brain;
}

export default function MemoryWikiPage() {
  const [entries, setEntries] = useState<KnowledgeEntry[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [document, setDocument] = useState<KnowledgeReadResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [reading, setReading] = useState(false);
  const [query, setQuery] = useState("");
  const [source, setSource] = useState<"all" | "memory" | "wiki">("all");
  const { toast, showToast } = useToast();

  const loadEntries = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.getKnowledgeEntries();
      setEntries(res.entries);
      setSelectedId((current) => current ?? res.entries[0]?.id ?? null);
    } catch (e) {
      showToast(`Error cargando memoria/wiki: ${e}`, "error");
    } finally {
      setLoading(false);
    }
  }, [showToast]);

  useEffect(() => {
    void loadEntries();
  }, [loadEntries]);

  useEffect(() => {
    if (!selectedId) {
      setDocument(null);
      return;
    }
    setReading(true);
    api
      .readKnowledgeEntry(selectedId)
      .then((res) => setDocument(res))
      .catch((e) => showToast(`Error leyendo documento: ${e}`, "error"))
      .finally(() => setReading(false));
  }, [selectedId, showToast]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return entries.filter((entry) => {
      if (source !== "all" && entry.source !== source) return false;
      if (!q) return true;
      return [entry.title, entry.group, entry.path, entry.kind]
        .filter(Boolean)
        .join(" ")
        .toLowerCase()
        .includes(q);
    });
  }, [entries, query, source]);

  const counts = useMemo(
    () => ({
      all: entries.length,
      memory: entries.filter((entry) => entry.source === "memory").length,
      wiki: entries.filter((entry) => entry.source === "wiki").length,
    }),
    [entries],
  );

  return (
    <div className="flex flex-col gap-5">
      <Toast toast={toast} />

      <section className="grid gap-4 md:grid-cols-[1.1fr_0.9fr]">
        <Card className="overflow-hidden border-primary/20 bg-gradient-to-br from-primary/10 via-card to-card">
          <CardContent className="p-5">
            <div className="flex items-start gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-primary/25 bg-primary/15 text-primary">
                <Database className="h-7 w-7" />
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-xs uppercase tracking-[0.24em] text-muted-foreground">
                  Knowledge layers
                </p>
                <h1 className="mt-1 text-2xl font-semibold tracking-tight">
                  Memorias y Wiki
                </h1>
                <p className="mt-2 max-w-3xl text-sm text-muted-foreground">
                  Visor canónico del dashboard original: memoria corta por perfil
                  (<code className="mx-1 rounded bg-muted px-1 py-0.5">MEMORY.md</code>
                  / <code className="mx-1 rounded bg-muted px-1 py-0.5">USER.md</code>) y conocimiento persistente en
                  <code className="mx-1 rounded bg-muted px-1 py-0.5">~/wiki</code>.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="grid grid-cols-3 gap-3">
          {([
            ["all", "Todo", counts.all],
            ["memory", "Memoria", counts.memory],
            ["wiki", "Wiki", counts.wiki],
          ] as const).map(([key, label, count]) => (
            <button
              key={key}
              type="button"
              onClick={() => setSource(key)}
              className={cn(
                "rounded-xl border bg-card p-4 text-left transition hover:border-primary/40",
                source === key && "border-primary/60 bg-primary/10",
              )}
            >
              <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
                {label}
              </p>
              <p className="mt-2 text-2xl font-semibold">{count}</p>
            </button>
          ))}
        </div>
      </section>

      <div className="grid min-h-[620px] gap-4 lg:grid-cols-[380px_1fr]">
        <Card className="overflow-hidden">
          <CardContent className="flex h-full flex-col gap-3 p-4">
            <div className="relative">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                className="pl-9"
                placeholder="Buscar memoria, perfil o wiki…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>

            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>{filtered.length} documentos</span>
              <Button ghost size="sm" onClick={() => void loadEntries()}>
                refrescar
              </Button>
            </div>

            <div className="flex-1 overflow-y-auto pr-1">
              {loading ? (
                <div className="py-10 text-center text-sm text-muted-foreground">
                  Cargando…
                </div>
              ) : filtered.length === 0 ? (
                <div className="py-10 text-center text-sm text-muted-foreground">
                  No hay documentos para ese filtro.
                </div>
              ) : (
                <div className="flex flex-col gap-2">
                  {filtered.map((entry) => {
                    const Icon = sourceIcon(entry.source);
                    return (
                      <button
                        key={entry.id}
                        type="button"
                        onClick={() => setSelectedId(entry.id)}
                        className={cn(
                          "rounded-xl border bg-card/80 p-3 text-left transition hover:border-primary/40 hover:bg-primary/5",
                          selectedId === entry.id && "border-primary/70 bg-primary/10",
                        )}
                      >
                        <div className="flex items-start gap-3">
                          <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border bg-background/60 text-muted-foreground">
                            <Icon className="h-4 w-4" />
                          </div>
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center gap-2">
                              <p className="truncate text-sm font-medium">{entry.title}</p>
                              <Badge tone={entry.source === "wiki" ? "outline" : "secondary"}>
                                {entry.source}
                              </Badge>
                            </div>
                            <p className="mt-1 truncate text-xs text-muted-foreground">
                              {entry.group} · {entry.path}
                            </p>
                            <p className="mt-1 text-xs text-muted-foreground">
                              {formatBytes(entry.size)} · {formatUpdated(entry.updated_at)}
                            </p>
                          </div>
                        </div>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        <Card className="overflow-hidden">
          <CardContent className="flex h-full min-h-[620px] flex-col p-0">
            {reading ? (
              <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">
                Leyendo documento…
              </div>
            ) : !document ? (
              <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">
                Selecciona una memoria o página wiki.
              </div>
            ) : (
              <>
                <header className="border-b border-border p-4">
                  <div className="flex flex-wrap items-center gap-2">
                    <FileText className="h-4 w-4 text-muted-foreground" />
                    <h2 className="text-lg font-semibold">{document.title}</h2>
                    <Badge tone={document.source === "wiki" ? "outline" : "secondary"}>
                      {document.source}
                    </Badge>
                  </div>
                  <p className="mt-1 text-xs text-muted-foreground">{document.path}</p>
                </header>
                <pre className="flex-1 overflow-auto whitespace-pre-wrap p-5 font-mono text-sm leading-6 text-foreground">
                  {document.content || "(archivo vacío)"}
                </pre>
              </>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
