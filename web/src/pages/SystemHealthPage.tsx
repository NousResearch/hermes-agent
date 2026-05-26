import { useEffect, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Cpu,
  Database,
  GitBranch,
  HardDrive,
  RefreshCw,
  Server,
  Shield,
  Terminal,
} from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { api, type SystemHealthResponse, type SystemPathStat } from "@/lib/api";
import { usePageHeader } from "@/contexts/usePageHeader";

function valueOrUnknown(value: string | number | null | undefined): string {
  if (value === null || value === undefined || value === "") return "Unavailable";
  return String(value);
}

function bytes(value: number | null): string {
  if (value === null) return "Unavailable";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 * 1024 * 1024) return `${(value / 1024 / 1024).toFixed(1)} MB`;
  return `${(value / 1024 / 1024 / 1024).toFixed(1)} GB`;
}

function Field({ label, value }: { label: string; value: string | number | null | undefined }) {
  return (
    <div className="rounded border border-border/50 bg-background/30 p-3">
      <div className="text-[11px] uppercase tracking-[0.16em] text-text-tertiary">{label}</div>
      <div className="mt-1 break-all font-mono text-sm text-text-primary">{valueOrUnknown(value)}</div>
    </div>
  );
}

function PathRow({ name, stat }: { name: string; stat: SystemPathStat }) {
  return (
    <div className="rounded border border-border/60 bg-background/30 p-3">
      <div className="flex items-center justify-between gap-3">
        <div className="font-medium text-text-primary">{name}</div>
        <Badge tone={stat.exists ? "success" : "secondary"}>{stat.exists ? "present" : "missing"}</Badge>
      </div>
      <div className="mt-1 break-all font-mono text-xs text-text-secondary">{stat.path}</div>
      <div className="mt-2 text-xs text-text-tertiary">
        {stat.is_dir ? "directory" : "file"} · {bytes(stat.size)}
        {stat.mtime ? ` · ${new Date(stat.mtime * 1000).toLocaleString()}` : ""}
      </div>
    </div>
  );
}

export default function SystemHealthPage() {
  const { setTitle, setAfterTitle } = usePageHeader();
  const [data, setData] = useState<SystemHealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);

  useEffect(() => {
    setTitle("System / Health");
    setAfterTitle(<span className="text-xs text-text-tertiary">Read-only local Hermes runtime diagnostics.</span>);
    return () => {
      setTitle(null);
      setAfterTitle(null);
    };
  }, [setTitle, setAfterTitle]);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const result = await api.getSystemHealth();
      setData(result);
      setUpdatedAt(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    const id = window.setInterval(load, 30000);
    return () => window.clearInterval(id);
  }, []);

  const gatewayOk = Boolean(data?.gateway.running);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-primary/20 bg-primary/5 p-4">
        <div className="flex items-center gap-3">
          {gatewayOk ? <CheckCircle2 className="h-5 w-5 text-success" /> : <AlertTriangle className="h-5 w-5 text-warning" />}
          <div>
            <div className="font-semibold text-text-primary">Local system health</div>
            <div className="text-sm text-text-secondary">No secrets shown. No process inventory beyond Hermes dashboard/gateway metadata.</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {updatedAt ? <span className="text-xs text-text-tertiary">Updated {updatedAt.toLocaleTimeString()}</span> : null}
          <Button size="sm" onClick={load} disabled={loading}>
            <RefreshCw className="mr-2 h-4 w-4" /> Refresh
          </Button>
        </div>
      </div>

      {error ? <div className="rounded border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">{error}</div> : null}

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Card className="border-border/60 bg-surface/70"><CardContent className="p-4"><Shield className="mb-3 h-5 w-5 text-primary" /><div className="text-xs uppercase tracking-[0.16em] text-text-tertiary">Gateway</div><div className="mt-2 text-xl font-semibold text-text-primary">{data?.gateway.state ?? (gatewayOk ? "running" : "stopped")}</div><div className="text-xs text-text-secondary">PID {valueOrUnknown(data?.gateway.pid)}</div></CardContent></Card>
        <Card className="border-border/60 bg-surface/70"><CardContent className="p-4"><GitBranch className="mb-3 h-5 w-5 text-primary" /><div className="text-xs uppercase tracking-[0.16em] text-text-tertiary">Git</div><div className="mt-2 text-xl font-semibold text-text-primary">{valueOrUnknown(data?.git.branch)}</div><div className="text-xs text-text-secondary">{valueOrUnknown(data?.git.commit)} · {data?.git.dirty ? `${data.git.dirty_count} changed` : "clean"}</div></CardContent></Card>
        <Card className="border-border/60 bg-surface/70"><CardContent className="p-4"><Cpu className="mb-3 h-5 w-5 text-primary" /><div className="text-xs uppercase tracking-[0.16em] text-text-tertiary">Runtime</div><div className="mt-2 text-xl font-semibold text-text-primary">Python {valueOrUnknown(data?.runtime.python)}</div><div className="text-xs text-text-secondary">{valueOrUnknown(data?.runtime.system)} {valueOrUnknown(data?.runtime.machine)}</div></CardContent></Card>
        <Card className="border-border/60 bg-surface/70"><CardContent className="p-4"><Server className="mb-3 h-5 w-5 text-primary" /><div className="text-xs uppercase tracking-[0.16em] text-text-tertiary">Dashboard</div><div className="mt-2 text-xl font-semibold text-text-primary">PID {valueOrUnknown(data?.runtime.process_pid)}</div><div className="text-xs text-text-secondary">Hermes {valueOrUnknown(data?.hermes.version)}</div></CardContent></Card>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Card className="border-border/60 bg-surface/70">
          <CardHeader><CardTitle className="flex items-center gap-2 text-sm uppercase tracking-[0.14em]"><Terminal className="h-4 w-4 text-primary" /> Hermes runtime</CardTitle></CardHeader>
          <CardContent className="grid gap-3">
            <Field label="Hermes home" value={data?.hermes.home} />
            <Field label="Config path" value={data?.hermes.config_path} />
            <Field label="Env path" value={data?.hermes.env_path} />
            <Field label="Project root" value={data?.hermes.project_root} />
            <Field label="Web dist" value={data?.hermes.web_dist} />
            <Field label="Python executable" value={data?.runtime.python_executable} />
            <Field label="Platform" value={data?.runtime.platform} />
          </CardContent>
        </Card>

        <Card className="border-border/60 bg-surface/70">
          <CardHeader><CardTitle className="flex items-center gap-2 text-sm uppercase tracking-[0.14em]"><HardDrive className="h-4 w-4 text-primary" /> Local storage</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            {data ? Object.entries(data.paths).map(([name, stat]) => <PathRow key={name} name={name.replace(/_/g, " ")} stat={stat} />) : null}
          </CardContent>
        </Card>
      </div>

      <Card className="border-border/60 bg-surface/70">
        <CardHeader><CardTitle className="flex items-center gap-2 text-sm uppercase tracking-[0.14em]"><Database className="h-4 w-4 text-primary" /> Last error lines</CardTitle></CardHeader>
        <CardContent className="space-y-2">
          {data?.last_errors?.length ? data.last_errors.map((line, idx) => (
            <pre key={idx} className="overflow-x-auto rounded border border-warning/20 bg-warning/5 p-2 text-xs text-text-secondary">{line}</pre>
          )) : <div className="text-sm text-text-secondary">No recent error lines found.</div>}
        </CardContent>
      </Card>
    </div>
  );
}
