import { useEffect, useLayoutEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { CalendarDays, Database, MessageSquare, Search, Tags } from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { usePageHeader } from "@/contexts/usePageHeader";
import { api } from "@/lib/api";
import type {
  DailyMemoryLog,
  MemoryOverviewResponse,
  MemorySessionInfo,
  MemorySubject,
} from "@/lib/api";

function formatTime(timestamp: number): string {
  if (!timestamp) return "Unknown";
  return new Date(timestamp * 1000).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

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

function PageShell({ children }: { children: React.ReactNode }) {
  const { setEnd } = usePageHeader();
  useLayoutEffect(() => {
    setEnd(null);
    return () => setEnd(null);
  }, [setEnd]);
  return <div className="space-y-6 p-4 sm:p-6">{children}</div>;
}

function LoadingState() {
  return (
    <PageShell>
      <div className="flex min-h-[240px] items-center justify-center text-muted-foreground">
        <Spinner className="mr-2 h-4 w-4" /> Loading memory wiki…
      </div>
    </PageShell>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <PageShell>
      <Card>
        <CardContent className="py-8 text-sm text-destructive">{message}</CardContent>
      </Card>
    </PageShell>
  );
}

function StatCard({ label, value, icon: Icon }: { label: string; value: string | number; icon: typeof Database }) {
  return (
    <Card>
      <CardContent className="flex items-center gap-3 py-5">
        <Icon className="h-5 w-5 text-muted-foreground" />
        <div>
          <div className="text-2xl font-semibold leading-none">{value}</div>
          <div className="mt-1 text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
        </div>
      </CardContent>
    </Card>
  );
}

function SessionList({ sessions }: { sessions: MemorySessionInfo[] }) {
  if (sessions.length === 0) {
    return <p className="text-sm text-muted-foreground">No sessions found.</p>;
  }
  return (
    <div className="space-y-2">
      {sessions.map((session) => (
        <Link
          key={session.id}
          to="/sessions"
          title={`Open Sessions to inspect ${session.id}`}
          className="block rounded-md border border-border/60 p-3 transition-colors hover:bg-muted/30 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <div className="flex flex-wrap items-center gap-2">
            <MessageSquare className="h-4 w-4 text-muted-foreground" />
            <span className="font-medium">{session.title || session.id}</span>
            {session.source && <Badge tone="secondary">{session.source}</Badge>}
            <span className="text-xs text-muted-foreground">{formatTime(session.last_active)}</span>
          </div>
          {session.preview && <p className="mt-2 line-clamp-2 text-sm text-muted-foreground">{session.preview}</p>}
        </Link>
      ))}
    </div>
  );
}

function SubjectCard({ subject }: { subject: MemorySubject }) {
  return (
    <Link to={`/memory/subjects/${subject.slug}`} className="block rounded-lg outline-none ring-offset-background focus-visible:ring-2 focus-visible:ring-ring">
      <Card className="h-full transition-colors hover:bg-muted/30">
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between gap-3">
            <CardTitle className="text-base">{subject.name}</CardTitle>
            <Badge>{subject.session_count} sessions</Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap gap-1.5">
            {subject.keywords.slice(0, 5).map((keyword) => (
              <Badge key={keyword} tone="secondary">{keyword}</Badge>
            ))}
          </div>
          <p className="text-xs text-muted-foreground">
            {subject.message_count} messages · last touched {formatTime(subject.last_seen)}
          </p>
          {subject.snippets[0] && (
            <p className="line-clamp-3 text-sm text-muted-foreground">{subject.snippets[0].text}</p>
          )}
        </CardContent>
      </Card>
    </Link>
  );
}

function DayCard({ day }: { day: DailyMemoryLog }) {
  return (
    <Link to={`/memory/days/${day.date}`} className="block rounded-lg outline-none ring-offset-background focus-visible:ring-2 focus-visible:ring-ring">
      <Card className="transition-colors hover:bg-muted/30">
        <CardHeader className="pb-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <CardTitle className="text-base">{formatDate(day.date)}</CardTitle>
            <Badge>{day.session_count} sessions</Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap gap-1.5">
            {day.subjects.slice(0, 6).map((subject) => (
              <Badge key={subject.slug} tone="secondary">{subject.name}</Badge>
            ))}
          </div>
          <ul className="space-y-1 text-sm text-muted-foreground">
            {day.work_items.slice(0, 3).map((item, idx) => (
              <li key={`${item.session_id}-${item.timestamp}-${idx}`} className="line-clamp-1">
                <span className="font-medium text-foreground/80">{item.kind}:</span> {item.text}
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </Link>
  );
}

export default function MemoryWikiPage() {
  const [overview, setOverview] = useState<MemoryOverviewResponse | null>(null);
  const [search, setSearch] = useState("");
  const [searchResults, setSearchResults] = useState<MemorySubject[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getMemoryOverview()
      .then(setOverview)
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load memory wiki"))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    const trimmed = search.trim();
    if (!trimmed) {
      setSearchResults(null);
      return;
    }
    let cancelled = false;
    const timer = window.setTimeout(() => {
      api
        .getMemorySubjects(trimmed)
        .then((resp) => {
          if (!cancelled) setSearchResults(resp.subjects);
        })
        .catch(() => {
          if (!cancelled) setSearchResults([]);
        });
    }, 250);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [search]);

  const visibleSubjects = searchResults ?? overview?.subjects ?? [];

  if (loading) return <LoadingState />;
  if (error || !overview) return <ErrorState message={error ?? "Memory wiki unavailable"} />;

  return (
    <PageShell>
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Database className="h-4 w-4" /> Local session history
        </div>
        <h1 className="text-3xl font-semibold tracking-tight">Memory Wiki</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          Browse deterministic subjects and daily logs generated from this Hermes profile’s local session database.
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        <StatCard icon={Tags} label="Subjects" value={overview.subjects.length} />
        <StatCard icon={CalendarDays} label="Daily logs" value={overview.daily_logs.length} />
        <StatCard icon={MessageSquare} label="Recent sessions" value={overview.recent_sessions.length} />
      </div>

      <div className="relative max-w-xl">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <input
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="Search subjects…"
          className="w-full rounded-md border border-border bg-background py-2 pl-9 pr-3 text-sm outline-none ring-offset-background focus:ring-2 focus:ring-ring"
        />
      </div>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">Subjects</h2>
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {visibleSubjects.map((subject) => <SubjectCard key={subject.slug} subject={subject} />)}
        </div>
        {visibleSubjects.length === 0 && <p className="text-sm text-muted-foreground">No subjects match your search.</p>}
      </section>

      <section className="grid gap-6 xl:grid-cols-[2fr_1fr]">
        <div className="space-y-3">
          <h2 className="text-xl font-semibold">Daily Logs</h2>
          <div className="space-y-3">
            {overview.daily_logs.map((day) => <DayCard key={day.date} day={day} />)}
          </div>
        </div>
        <Card>
          <CardHeader><CardTitle className="text-base">Recent Activity</CardTitle></CardHeader>
          <CardContent><SessionList sessions={overview.recent_sessions} /></CardContent>
        </Card>
      </section>
    </PageShell>
  );
}

export function MemorySubjectPage() {
  const { slug = "" } = useParams();
  const [subject, setSubject] = useState<MemorySubject | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    setSubject(null);
    api
      .getMemorySubject(slug)
      .then((resp) => setSubject(resp.subject))
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load subject"))
      .finally(() => setLoading(false));
  }, [slug]);

  if (loading) return <LoadingState />;
  if (error || !subject) return <ErrorState message={error ?? "Subject not found"} />;

  return (
    <PageShell>
      <Link to="/memory"><Button ghost size="sm" className="border border-border">← Memory Wiki</Button></Link>
      <div className="space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight">{subject.name}</h1>
        <p className="text-sm text-muted-foreground">
          {subject.session_count} sessions · {subject.message_count} messages · first seen {formatTime(subject.first_seen)}
        </p>
        <div className="flex flex-wrap gap-1.5">
          {subject.keywords.map((keyword) => <Badge key={keyword} tone="secondary">{keyword}</Badge>)}
        </div>
      </div>
      <div className="grid gap-6 xl:grid-cols-[1fr_1fr]">
        <Card>
          <CardHeader><CardTitle className="text-base">Related Sessions</CardTitle></CardHeader>
          <CardContent><SessionList sessions={subject.sessions} /></CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle className="text-base">Snippets</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            {subject.snippets.map((snippet) => (
              <blockquote key={`${snippet.session_id}-${snippet.message_id}-${snippet.timestamp}`} className="border-l-2 border-border pl-3 text-sm text-muted-foreground">
                <span className="font-medium text-foreground/80">{snippet.role ?? "message"}</span>: {snippet.text}
              </blockquote>
            ))}
          </CardContent>
        </Card>
      </div>
    </PageShell>
  );
}

export function MemoryDayPage() {
  const { date = "" } = useParams();
  const [day, setDay] = useState<DailyMemoryLog | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    setDay(null);
    api
      .getMemoryDay(date)
      .then((resp) => setDay(resp.daily_log))
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load daily log"))
      .finally(() => setLoading(false));
  }, [date]);

  if (loading) return <LoadingState />;
  if (error || !day) return <ErrorState message={error ?? "Daily log not found"} />;

  return (
    <PageShell>
      <Link to="/memory"><Button ghost size="sm" className="border border-border">← Memory Wiki</Button></Link>
      <div className="space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight">{formatDate(day.date)}</h1>
        <p className="text-sm text-muted-foreground">
          {day.session_count} sessions · {day.message_count} messages · active until {formatTime(day.last_active_max)}
        </p>
        <div className="flex flex-wrap gap-1.5">
          {day.subjects.map((subject) => (
            <Link key={subject.slug} to={`/memory/subjects/${subject.slug}`}>
              <Badge tone="secondary">{subject.name}</Badge>
            </Link>
          ))}
        </div>
      </div>
      <div className="grid gap-6 xl:grid-cols-[1fr_1fr]">
        <Card>
          <CardHeader><CardTitle className="text-base">What we did</CardTitle></CardHeader>
          <CardContent>
            <ul className="space-y-2 text-sm text-muted-foreground">
              {day.work_items.map((item, idx) => (
                <li key={`${item.session_id}-${item.timestamp}-${idx}`}>
                  <span className="font-medium text-foreground/80">{item.kind}</span>: {item.text}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle className="text-base">Sessions</CardTitle></CardHeader>
          <CardContent><SessionList sessions={day.sessions} /></CardContent>
        </Card>
      </div>
    </PageShell>
  );
}
