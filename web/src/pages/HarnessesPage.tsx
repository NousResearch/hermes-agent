import { useCallback, useEffect, useState } from "react";
import { Activity, RefreshCw, Server, Square, Play, RotateCw } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { fetchJSON } from "@/lib/api";

type Harness = {
  name: string;
  gateway_state: string;
  pid: number | null;
  platforms: Array<{ name: string; state?: string }>;
  served_profiles: string[];
  updated_at: string | null;
};

type HarnessesResponse = { profiles: Harness[]; count: number };
type ActionResponse = { ok: boolean; invocation_id: string };
type ActionStatus = { status: string; invocation_id: string };

const verbs = [
  { name: "start", label: "Start", icon: Play },
  { name: "stop", label: "Stop", icon: Square },
  { name: "restart", label: "Restart", icon: RotateCw },
] as const;

export default function HarnessesPage() {
  const [profiles, setProfiles] = useState<Harness[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const response = await fetchJSON<HarnessesResponse>("/api/harnesses");
      setProfiles(response.profiles);
      setError(null);
    } catch {
      setError("Unable to load harness status.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
    const timer = window.setInterval(() => void load(), 10_000);
    return () => window.clearInterval(timer);
  }, [load]);

  const run = async (profile: string, verb: typeof verbs[number]["name"]) => {
    const key = `${profile}:${verb}`;
    setBusy(key);
    setNotice(null);
    try {
      const started = await fetchJSON<ActionResponse>(
        `/api/harnesses/${encodeURIComponent(profile)}/actions/${verb}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ confirmed: true }),
        },
      );
      let status = await fetchJSON<ActionStatus>(
        `/api/harnesses/${encodeURIComponent(profile)}/actions/${verb}/status?invocation_id=${encodeURIComponent(started.invocation_id)}`,
      );
      for (let i = 0; i < 30 && status.status === "running"; i += 1) {
        await new Promise((resolve) => window.setTimeout(resolve, 500));
        status = await fetchJSON<ActionStatus>(
          `/api/harnesses/${encodeURIComponent(profile)}/actions/${verb}/status?invocation_id=${encodeURIComponent(started.invocation_id)}`,
        );
      }
      setNotice(status.status === "completed" ? `${verb} completed.` : `${verb} did not complete.`);
      await load();
    } catch {
      setNotice("The requested action could not be completed.");
    } finally {
      setBusy(null);
    }
  };

  return (
    <main className="mx-auto flex w-full max-w-6xl flex-col gap-6 p-6" aria-labelledby="harnesses-title">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5" aria-hidden="true" />
            <h1 id="harnesses-title" className="text-2xl font-semibold">Harnesses</h1>
          </div>
          <p className="mt-1 text-sm text-muted-foreground">Profiles, gateway state, connected platforms, and lifecycle controls.</p>
        </div>
        <Button outlined onClick={() => void load()} disabled={loading} aria-label="Refresh harnesses">
          <RefreshCw className="mr-2 h-4 w-4" aria-hidden="true" /> Refresh
        </Button>
      </header>
      {error && <p role="alert" className="rounded-md border border-destructive/40 p-3 text-sm">{error}</p>}
      {notice && <p role="status" className="rounded-md border border-border p-3 text-sm">{notice}</p>}
      {loading && profiles.length === 0 ? <div className="flex justify-center p-12"><Spinner /></div> : (
        <section className="grid gap-4 md:grid-cols-2" aria-label="Harness profiles">
          {profiles.map((profile) => (
            <Card key={profile.name}>
              <CardHeader className="flex flex-row items-center justify-between gap-3">
                <CardTitle className="flex items-center gap-2"><Server className="h-4 w-4" aria-hidden="true" />{profile.name}</CardTitle>
                <Badge tone={profile.gateway_state === "running" ? "success" : "outline"}>{profile.gateway_state}</Badge>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div><span className="text-muted-foreground">PID</span><div>{profile.pid ?? "—"}</div></div>
                  <div><span className="text-muted-foreground">Platforms</span><div>{profile.platforms.length}</div></div>
                </div>
                <div className="flex flex-wrap gap-2">
                  {profile.platforms.length ? profile.platforms.map((platform) => <Badge key={platform.name} tone="outline">{platform.name}{platform.state ? ` · ${platform.state}` : ""}</Badge>) : <span className="text-sm text-muted-foreground">No connected platforms</span>}
                </div>
                <div className="flex flex-wrap gap-2" aria-label={`${profile.name} lifecycle actions`}>
                  {verbs.map(({ name, label, icon: Icon }) => {
                    const actionKey = `${profile.name}:${name}`;
                    return <Button key={name} size="sm" outlined onClick={() => void run(profile.name, name)} disabled={busy !== null} aria-label={`${label} ${profile.name}`}><Icon className="mr-1 h-3.5 w-3.5" aria-hidden="true" />{busy === actionKey ? "Working…" : label}</Button>;
                  })}
                </div>
              </CardContent>
            </Card>
          ))}
        </section>
      )}
      {!loading && profiles.length === 0 && <p className="p-8 text-center text-sm text-muted-foreground">No harness profiles found.</p>}
    </main>
  );
}
