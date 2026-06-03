import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { BookOpen, CalendarDays, MessageSquare, RefreshCw, Search, Tag } from "lucide-react";
import { api } from "@/lib/api";
import type {
  MemoryWikiDayDetail,
  MemoryWikiDayEntry,
  MemoryWikiSubjectDetail,
  MemoryWikiSubjectEntry,
} from "@/lib/api";
import { Button } from "@nous-research/ui/ui/components/button";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { usePageHeader } from "@/contexts/usePageHeader";
import { timeAgo } from "@/lib/utils";

type WikiTab = "subjects" | "days";

type SelectedWikiItem =
  | { kind: "subject"; slug: string; title: string }
  | { kind: "day"; date: string; title: string }
  | null;

function formatDate(date: string): string {
  try {
    return new Date(`${date}T00:00:00`).toLocaleDateString(undefined, {
      weekday: "short",
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  } catch {
    return date;
  }
}

function formatDateTime(timestamp?: number | null): string {
  if (!timestamp) return "Unknown time";
  try {
    return new Date(timestamp * 1000).toLocaleString();
  } catch {
    return String(timestamp);
  }
}

function countLabel(count: number, singular: string, plural = `${singular}s`): string {
  return `${count} ${count === 1 ? singular : plural}`;
}

function EmptyState({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="rounded border border-dashed border-border bg-card/40 p-8 text-center">
      <BookOpen className="mx-auto mb-3 h-8 w-8 text-muted-foreground" />
      <div className="font-medium">{title}</div>
      <p className="mx-auto mt-2 max-w-md text-sm text-muted-foreground">{detail}</p>
    </div>
  );
}

function SummaryCard({ summary }: { summary: { headline: string; bullets: string[]; generated_by: string; generated_at: number } }) {
  return (
    <Card className="border-primary/30 bg-primary/[0.04]">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <BookOpen className="h-4 w-4 text-primary" />
          Captain&apos;s log
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-sm font-medium leading-relaxed">{summary.headline}</p>
        <ul className="space-y-2 text-sm text-muted-foreground">
          {summary.bullets.map((bullet, index) => (
            <li key={`${index}-${bullet}`} className="flex gap-2">
              <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-primary/70" />
              <span>{bullet}</span>
            </li>
          ))}
        </ul>
        <div className="text-xs text-muted-foreground">
          Cached summary · {summary.generated_by} · {formatDateTime(summary.generated_at)}
        </div>
      </CardContent>
    </Card>
  );
}

function CollapsibleDetailSection({
  title,
  count,
  children,
}: {
  title: string;
  count?: number;
  children: ReactNode;
}) {
  return (
    <details className="rounded border border-border bg-card">
      <summary className="cursor-pointer select-none px-4 py-3 text-sm font-medium hover:bg-secondary/30">
        {title}{typeof count === "number" ? ` (${count})` : ""}
      </summary>
      <div className="border-t border-border p-4">{children}</div>
    </details>
  );
}

function SessionList({ sessions }: { sessions: MemoryWikiDayDetail["sessions"] }) {
  if (sessions.length === 0) {
    return <EmptyState title="No sessions found" detail="This slice of the wiki does not have any sessions yet." />;
  }

  return (
    <div className="space-y-3">
      {sessions.map((session) => (
        <Link
          key={session.id}
          to={`/sessions?session=${encodeURIComponent(session.id)}`}
          title={`Open session ${session.id}`}
          className="block rounded border border-border bg-card p-4 transition-colors hover:bg-secondary/30"
        >
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
              <div className="truncate font-medium">{session.title || session.id}</div>
              <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                <span>{formatDateTime(session.started_at)}</span>
                {session.source && <Badge tone="secondary">{session.source}</Badge>}
                <span>{countLabel(session.message_count, "message")}</span>
              </div>
            </div>
            <MessageSquare className="h-4 w-4 shrink-0 text-muted-foreground" />
          </div>
          {session.preview && <p className="mt-3 line-clamp-2 text-sm text-muted-foreground">{session.preview}</p>}
        </Link>
      ))}
    </div>
  );
}

function SubjectsList({
  subjects,
  selected,
  onSelect,
}: {
  subjects: MemoryWikiSubjectEntry[];
  selected: SelectedWikiItem;
  onSelect: (subject: MemoryWikiSubjectEntry) => void;
}) {
  if (subjects.length === 0) {
    return <EmptyState title="No subjects yet" detail="Subjects are built from session titles and message text once conversations exist." />;
  }

  return (
    <div className="space-y-2">
      {subjects.map((subject) => {
        const active = selected?.kind === "subject" && selected.slug === subject.slug;
        return (
          <button
            key={subject.slug}
            type="button"
            onClick={() => onSelect(subject)}
            className={`w-full rounded border p-3 text-left transition-colors ${
              active ? "border-primary bg-primary/10" : "border-border bg-card hover:bg-secondary/30"
            }`}
          >
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0">
                <div className="truncate font-medium">{subject.title}</div>
                <div className="mt-1 text-xs text-muted-foreground">
                  {countLabel(subject.session_count, "session")} · {countLabel(subject.message_count, "message")}
                </div>
              </div>
              <Tag className="h-4 w-4 shrink-0 text-muted-foreground" />
            </div>
          </button>
        );
      })}
    </div>
  );
}

function DaysList({
  days,
  selected,
  onSelect,
}: {
  days: MemoryWikiDayEntry[];
  selected: SelectedWikiItem;
  onSelect: (day: MemoryWikiDayEntry) => void;
}) {
  if (days.length === 0) {
    return <EmptyState title="No daily logs yet" detail="Daily logs appear automatically as Hermes records sessions." />;
  }

  return (
    <div className="space-y-2">
      {days.map((day) => {
        const active = selected?.kind === "day" && selected.date === day.date;
        return (
          <button
            key={day.date}
            type="button"
            onClick={() => onSelect(day)}
            className={`w-full rounded border p-3 text-left transition-colors ${
              active ? "border-primary bg-primary/10" : "border-border bg-card hover:bg-secondary/30"
            }`}
          >
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0">
                <div className="truncate font-medium">{formatDate(day.date)}</div>
                <div className="mt-1 text-xs text-muted-foreground">
                  {countLabel(day.session_count, "session")} · {countLabel(day.message_count, "message")}
                </div>
              </div>
              <CalendarDays className="h-4 w-4 shrink-0 text-muted-foreground" />
            </div>
          </button>
        );
      })}
    </div>
  );
}

function SubjectDetail({ detail }: { detail: MemoryWikiSubjectDetail }) {
  return (
    <div className="space-y-4">
      <SummaryCard summary={detail.wiki_summary} />

      <div className="grid gap-3 sm:grid-cols-3">
        <Card>
          <CardContent className="p-4">
            <div className="text-2xl font-semibold">{detail.summary.session_count}</div>
            <div className="text-xs text-muted-foreground">Sessions</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="text-2xl font-semibold">{detail.summary.message_count}</div>
            <div className="text-xs text-muted-foreground">Related messages</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="text-sm font-medium">{detail.summary.latest_at ? timeAgo(detail.summary.latest_at) : "Unknown"}</div>
            <div className="text-xs text-muted-foreground">Latest activity</div>
          </CardContent>
        </Card>
      </div>

      <CollapsibleDetailSection title="Related sessions" count={detail.sessions.length}>
        <SessionList sessions={detail.sessions} />
      </CollapsibleDetailSection>

      <CollapsibleDetailSection title="Message hits" count={detail.message_hits.length}>
        {detail.message_hits.length === 0 ? (
          <p className="text-sm text-muted-foreground">No direct message hits found yet; this subject may come from session titles.</p>
        ) : (
          <div className="space-y-3">
            {detail.message_hits.map((hit) => (
              <Link
                key={hit.id}
                to={`/sessions?session=${encodeURIComponent(hit.session_id)}`}
                title={`Open session ${hit.session_id}`}
                className="block rounded border border-border bg-background p-3 hover:bg-secondary/30"
              >
                <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                  <Badge tone="secondary">{hit.role}</Badge>
                  <span>{hit.session_title || hit.session_id}</span>
                  <span>{formatDateTime(hit.timestamp)}</span>
                </div>
                <p className="text-sm">{hit.content}</p>
              </Link>
            ))}
          </div>
        )}
      </CollapsibleDetailSection>
    </div>
  );
}

function DayDetail({ detail }: { detail: MemoryWikiDayDetail }) {
  return (
    <div className="space-y-4">
      <SummaryCard summary={detail.wiki_summary} />

      <div className="grid gap-3 sm:grid-cols-3">
        <Card>
          <CardContent className="p-4">
            <div className="text-2xl font-semibold">{detail.summary.session_count}</div>
            <div className="text-xs text-muted-foreground">Sessions</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="text-2xl font-semibold">{detail.summary.message_count}</div>
            <div className="text-xs text-muted-foreground">Messages</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="text-sm font-medium">{detail.summary.sources.join(", ") || "No source"}</div>
            <div className="text-xs text-muted-foreground">Sources</div>
          </CardContent>
        </Card>
      </div>

      <CollapsibleDetailSection title={`Sessions on ${formatDate(detail.date)}`} count={detail.sessions.length}>
        <SessionList sessions={detail.sessions} />
      </CollapsibleDetailSection>
    </div>
  );
}

export default function MemoryWikiPage() {
  const [tab, setTab] = useState<WikiTab>("days");
  const [query, setQuery] = useState("");
  const [days, setDays] = useState<MemoryWikiDayEntry[]>([]);
  const [subjects, setSubjects] = useState<MemoryWikiSubjectEntry[]>([]);
  const [selected, setSelected] = useState<SelectedWikiItem>(null);
  const [subjectDetail, setSubjectDetail] = useState<MemoryWikiSubjectDetail | null>(null);
  const [dayDetail, setDayDetail] = useState<MemoryWikiDayDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const detailScrollRef = useRef<HTMLDivElement | null>(null);
  const { setAfterTitle, setEnd } = usePageHeader();

  const refresh = async () => {
    setLoading(true);
    setError(null);
    try {
      const [subjectResponse, dayResponse] = await Promise.all([
        api.getMemoryWikiSubjects(),
        api.getMemoryWikiDays(),
      ]);
      setSubjects(subjectResponse.subjects);
      setDays(dayResponse.days);
      if (!selected) {
        if (dayResponse.days.length > 0) {
          setSelected({ kind: "day", date: dayResponse.days[0].date, title: formatDate(dayResponse.days[0].date) });
          setTab("days");
        } else if (subjectResponse.subjects.length > 0) {
          setSelected({ kind: "subject", slug: subjectResponse.subjects[0].slug, title: subjectResponse.subjects[0].title });
          setTab("subjects");
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load memory wiki");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!selected) return;
    detailScrollRef.current?.scrollTo({ top: 0 });
    setDetailLoading(true);
    setError(null);
    if (selected.kind === "subject") {
      api
        .getMemoryWikiSubject(selected.slug)
        .then((detail) => {
          setSubjectDetail(detail);
          setDayDetail(null);
        })
        .catch((err) => setError(err instanceof Error ? err.message : "Failed to load subject"))
        .finally(() => setDetailLoading(false));
    } else {
      api
        .getMemoryWikiDay(selected.date)
        .then((detail) => {
          setDayDetail(detail);
          setSubjectDetail(null);
        })
        .catch((err) => setError(err instanceof Error ? err.message : "Failed to load day"))
        .finally(() => setDetailLoading(false));
    }
  }, [selected]);

  useLayoutEffect(() => {
    setAfterTitle(<span className="text-xs text-muted-foreground">Browse remembered conversations by subject or day</span>);
    setEnd(
      <Button size="sm" outlined onClick={() => void refresh()} disabled={loading}>
        <RefreshCw className={`mr-2 h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
        Refresh
      </Button>,
    );
    return () => {
      setAfterTitle(null);
      setEnd(null);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loading, setAfterTitle, setEnd]);

  const filteredSubjects = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return subjects;
    return subjects.filter((subject) => subject.title.toLowerCase().includes(q) || subject.slug.includes(q));
  }, [query, subjects]);

  const filteredDays = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return days;
    return days.filter((day) => day.date.includes(q) || formatDate(day.date).toLowerCase().includes(q));
  }, [days, query]);

  const hideSelectedSubject = async () => {
    if (!selected || selected.kind !== "subject") return;
    setDetailLoading(true);
    setError(null);
    try {
      await api.hideMemoryWikiSubject(selected.slug);
      const nextSubjects = subjects.filter((subject) => subject.slug !== selected.slug);
      setSubjects(nextSubjects);
      setSubjectDetail(null);
      if (nextSubjects.length > 0) {
        setSelected({ kind: "subject", slug: nextSubjects[0].slug, title: nextSubjects[0].title });
      } else if (days.length > 0) {
        setTab("days");
        setSelected({ kind: "day", date: days[0].date, title: formatDate(days[0].date) });
      } else {
        setSelected(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to hide subject");
    } finally {
      setDetailLoading(false);
    }
  };

  return (
    <div className="normal-case p-4 sm:p-6 lg:h-[calc(100vh-5rem)] lg:overflow-hidden">
      <div className="grid gap-4 lg:h-full lg:min-h-0 lg:grid-cols-[360px_1fr]">
        <div className="space-y-4 lg:min-h-0 lg:overflow-y-auto lg:pr-1">
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <BookOpen className="h-5 w-5 text-muted-foreground" />
                <CardTitle className="text-base">Session Summary</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-2">
                <Button outlined={tab !== "days"} onClick={() => setTab("days")}>
                  By Day
                </Button>
                <Button outlined={tab !== "subjects"} onClick={() => setTab("subjects")}>
                  By Subject
                </Button>
              </div>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <input
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder={tab === "subjects" ? "Search subjects…" : "Search days…"}
                  className="h-10 w-full rounded border border-input bg-background pl-9 pr-3 text-sm outline-none focus:ring-1 focus:ring-ring"
                />
              </div>
            </CardContent>
          </Card>

          {loading ? (
            <div className="flex items-center justify-center p-8"><Spinner /></div>
          ) : tab === "subjects" ? (
            <SubjectsList
              subjects={filteredSubjects}
              selected={selected}
              onSelect={(subject) => setSelected({ kind: "subject", slug: subject.slug, title: subject.title })}
            />
          ) : (
            <DaysList
              days={filteredDays}
              selected={selected}
              onSelect={(day) => setSelected({ kind: "day", date: day.date, title: formatDate(day.date) })}
            />
          )}
        </div>

        <div ref={detailScrollRef} className="min-w-0 space-y-4 lg:min-h-0 lg:overflow-y-auto lg:pr-1">
          {error && <div className="rounded border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">{error}</div>}

          {!selected && !loading ? (
            <EmptyState title="Nothing selected" detail="Choose a subject or daily log from the left to inspect the conversations behind it." />
          ) : (
            <Card>
              <CardHeader>
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <CardTitle className="text-xl">{selected?.title ?? "Loading…"}</CardTitle>
                    <p className="mt-1 text-sm text-muted-foreground">
                      {selected?.kind === "subject" ? "Subject detail" : "Daily log detail"}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    {selected?.kind === "subject" && (
                      <Button size="sm" outlined onClick={() => void hideSelectedSubject()} disabled={detailLoading}>
                        Hide subject
                      </Button>
                    )}
                    {selected?.kind === "subject" ? <Tag className="h-5 w-5 text-muted-foreground" /> : <CalendarDays className="h-5 w-5 text-muted-foreground" />}
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                {detailLoading ? (
                  <div className="flex items-center justify-center p-12"><Spinner /></div>
                ) : selected?.kind === "subject" && subjectDetail ? (
                  <SubjectDetail detail={subjectDetail} />
                ) : selected?.kind === "day" && dayDetail ? (
                  <DayDetail detail={dayDetail} />
                ) : (
                  <EmptyState title="No detail loaded" detail="Select another item or refresh the wiki." />
                )}
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
