import { useCallback, useEffect, useRef, useState } from "react";
import {
  Activity,
  Brain,
  Database,
  KeyRound,
  Play,
  Power,
  RotateCw,
  ShieldCheck,
  Stethoscope,
  Terminal,
  Trash2,
  X,
} from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Select, SelectOption } from "@nous-research/ui/ui/components/select";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { H2 } from "@nous-research/ui/ui/components/typography/h2";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { useConfirmDelete } from "@nous-research/ui/hooks/use-confirm-delete";
import { DeleteConfirmDialog } from "@/components/DeleteConfirmDialog";
import { api } from "@/lib/api";
import type {
  StatusResponse,
  MemoryStatus,
  CredentialPoolProvider,
  CheckpointsResponse,
  HooksResponse,
} from "@/lib/api";

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * A running-action log viewer.  The spawn-based admin actions (doctor,
 * security audit, backup, import, skills update, checkpoints prune,
 * gateway start/stop) stream their stdout to a per-action log file the
 * server tails via /api/actions/<name>/status.  When an action is launched
 * we poll that endpoint until the process exits, showing live output.
 */
function ActionLogViewer({
  action,
  onClose,
}: {
  action: string;
  onClose: () => void;
}) {
  const [lines, setLines] = useState<string[]>([]);
  const [running, setRunning] = useState(true);
  const [exitCode, setExitCode] = useState<number | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const st = await api.getActionStatus(action, 400);
        if (cancelled) return;
        setLines(st.lines);
        setRunning(st.running);
        setExitCode(st.exit_code);
        if (st.running) {
          timer.current = setTimeout(poll, 1200);
        }
      } catch {
        if (!cancelled) {
          setRunning(false);
        }
      }
    };
    poll();
    return () => {
      cancelled = true;
      if (timer.current) clearTimeout(timer.current);
    };
  }, [action]);

  return (
    <Card>
      <CardContent className="py-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Terminal className="h-4 w-4 text-muted-foreground" />
            <span className="font-mono text-sm">{action}</span>
            {running ? (
              <Badge tone="warning">running</Badge>
            ) : (
              <Badge tone={exitCode === 0 ? "success" : "destructive"}>
                {exitCode === 0 ? "done" : `exit ${exitCode}`}
              </Badge>
            )}
          </div>
          <Button ghost size="icon" onClick={onClose} aria-label="Close log">
            <X />
          </Button>
        </div>
        <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words bg-background/50 border border-border p-3 text-xs font-mono text-muted-foreground">
          {lines.length ? lines.join("\n") : "Starting…"}
        </pre>
      </CardContent>
    </Card>
  );
}

export default function SystemPage() {
  const { toast, showToast } = useToast();

  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [memory, setMemory] = useState<MemoryStatus | null>(null);
  const [pool, setPool] = useState<CredentialPoolProvider[]>([]);
  const [checkpoints, setCheckpoints] = useState<CheckpointsResponse | null>(
    null,
  );
  const [hooks, setHooks] = useState<HooksResponse | null>(null);
  const [loading, setLoading] = useState(true);

  // Which spawn-action log is currently shown (null = none).
  const [activeAction, setActiveAction] = useState<string | null>(null);

  // Add-credential form.
  const [credProvider, setCredProvider] = useState("openrouter");
  const [credKey, setCredKey] = useState("");
  const [credLabel, setCredLabel] = useState("");
  const [addingCred, setAddingCred] = useState(false);

  // Import archive path.
  const [importPath, setImportPath] = useState("");

  const loadAll = useCallback(() => {
    Promise.allSettled([
      api.getStatus(),
      api.getMemory(),
      api.getCredentialPool(),
      api.getCheckpoints(),
      api.getHooks(),
    ])
      .then(([s, m, p, c, h]) => {
        if (s.status === "fulfilled") setStatus(s.value);
        if (m.status === "fulfilled") setMemory(m.value);
        if (p.status === "fulfilled") setPool(p.value.providers);
        if (c.status === "fulfilled") setCheckpoints(c.value);
        if (h.status === "fulfilled") setHooks(h.value);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    loadAll();
  }, [loadAll]);

  // ── Gateway lifecycle ──────────────────────────────────────────────
  const runGateway = async (
    verb: "start" | "stop" | "restart",
  ) => {
    try {
      if (verb === "start") {
        await api.startGateway();
        setActiveAction("gateway-start");
      } else if (verb === "stop") {
        await api.stopGateway();
        setActiveAction("gateway-stop");
      } else {
        await api.restartGateway();
        setActiveAction("gateway-restart");
      }
      showToast(`Gateway ${verb} started`, "success");
      setTimeout(loadAll, 3000);
    } catch (e) {
      showToast(`Gateway ${verb} failed: ${e}`, "error");
    }
  };

  // ── Memory ─────────────────────────────────────────────────────────
  const setMemoryProvider = async (provider: string) => {
    try {
      await api.setMemoryProvider(provider);
      showToast(
        `Memory provider: ${provider || "built-in only"}`,
        "success",
      );
      loadAll();
    } catch (e) {
      showToast(`Failed to set provider: ${e}`, "error");
    }
  };

  const memoryReset = useConfirmDelete({
    onDelete: useCallback(
      async (target: string) => {
        try {
          const res = await api.resetMemory(
            target as "all" | "memory" | "user",
          );
          showToast(`Reset: ${res.deleted.join(", ") || "nothing"}`, "success");
          loadAll();
        } catch (e) {
          showToast(`Reset failed: ${e}`, "error");
          throw e;
        }
      },
      [loadAll, showToast],
    ),
  });

  // ── Credential pool ────────────────────────────────────────────────
  const addCredential = async () => {
    if (!credProvider.trim() || !credKey.trim()) {
      showToast("Provider and API key required", "error");
      return;
    }
    setAddingCred(true);
    try {
      await api.addCredentialPoolEntry(
        credProvider.trim(),
        credKey.trim(),
        credLabel.trim() || undefined,
      );
      showToast("Credential added", "success");
      setCredKey("");
      setCredLabel("");
      loadAll();
    } catch (e) {
      showToast(`Failed to add credential: ${e}`, "error");
    } finally {
      setAddingCred(false);
    }
  };

  const credDelete = useConfirmDelete({
    onDelete: useCallback(
      async (key: string) => {
        const [provider, idxStr] = key.split("|");
        try {
          await api.removeCredentialPoolEntry(provider, Number(idxStr));
          showToast("Credential removed", "success");
          loadAll();
        } catch (e) {
          showToast(`Failed to remove: ${e}`, "error");
          throw e;
        }
      },
      [loadAll, showToast],
    ),
  });

  // ── Operations ─────────────────────────────────────────────────────
  const runOp = async (
    fn: () => Promise<{ name: string }>,
    label: string,
  ) => {
    try {
      const res = await fn();
      setActiveAction(res.name);
      showToast(`${label} started`, "success");
    } catch (e) {
      showToast(`${label} failed: ${e}`, "error");
    }
  };

  const checkpointsPrune = useConfirmDelete({
    onDelete: useCallback(async () => {
      try {
        const res = await api.pruneCheckpoints();
        setActiveAction(res.name);
        showToast("Checkpoint prune started", "success");
      } catch (e) {
        showToast(`Prune failed: ${e}`, "error");
        throw e;
      }
    }, [showToast]),
  });

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner className="text-2xl text-primary" />
      </div>
    );
  }

  const gatewayRunning = status?.gateway_running;

  return (
    <div className="flex flex-col gap-8">
      <Toast toast={toast} />

      <DeleteConfirmDialog
        open={memoryReset.isOpen}
        onCancel={memoryReset.cancel}
        onConfirm={memoryReset.confirm}
        title="Reset memory"
        description="This permanently erases the selected built-in memory files. This cannot be undone."
        loading={memoryReset.isDeleting}
      />
      <DeleteConfirmDialog
        open={credDelete.isOpen}
        onCancel={credDelete.cancel}
        onConfirm={credDelete.confirm}
        title="Remove credential"
        description="Remove this pooled API key? The agent will no longer rotate through it."
        loading={credDelete.isDeleting}
      />
      <DeleteConfirmDialog
        open={checkpointsPrune.isOpen}
        onCancel={checkpointsPrune.cancel}
        onConfirm={checkpointsPrune.confirm}
        title="Prune checkpoints"
        description="Delete the rollback checkpoint shadow store? Existing /rollback points will be lost."
        loading={checkpointsPrune.isDeleting}
      />

      {/* Live action log */}
      {activeAction && (
        <ActionLogViewer
          action={activeAction}
          onClose={() => setActiveAction(null)}
        />
      )}

      {/* ── Gateway ───────────────────────────────────────────────── */}
      <section className="flex flex-col gap-3">
        <H2 variant="sm" className="flex items-center gap-2 text-muted-foreground">
          <Power className="h-4 w-4" /> Gateway
        </H2>
        <Card>
          <CardContent className="flex items-center justify-between py-4">
            <div className="flex items-center gap-3">
              <Badge tone={gatewayRunning ? "success" : "secondary"}>
                {gatewayRunning ? "running" : "stopped"}
              </Badge>
              <span className="text-sm text-muted-foreground">
                {status?.gateway_state ?? "—"}
                {status?.gateway_pid ? ` · pid ${status.gateway_pid}` : ""}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Button
                size="sm"
                className="uppercase"
                onClick={() => runGateway("start")}
                disabled={gatewayRunning}
                prefix={<Play className="h-3.5 w-3.5" />}
              >
                Start
              </Button>
              <Button
                size="sm"
                className="uppercase"
                onClick={() => runGateway("restart")}
                prefix={<RotateCw className="h-3.5 w-3.5" />}
              >
                Restart
              </Button>
              <Button
                size="sm"
                className="uppercase text-warning"
                ghost
                onClick={() => runGateway("stop")}
                disabled={!gatewayRunning}
                prefix={<Power className="h-3.5 w-3.5" />}
              >
                Stop
              </Button>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* ── Memory ────────────────────────────────────────────────── */}
      <section className="flex flex-col gap-3">
        <H2 variant="sm" className="flex items-center gap-2 text-muted-foreground">
          <Brain className="h-4 w-4" /> Memory
        </H2>
        <Card>
          <CardContent className="flex flex-col gap-4 py-4">
            <div className="grid gap-2 max-w-sm">
              <Label htmlFor="mem-provider">External provider</Label>
              <Select
                id="mem-provider"
                value={memory?.active || ""}
                onValueChange={setMemoryProvider}
              >
                <SelectOption value="">Built-in only</SelectOption>
                {(memory?.providers ?? []).map((p) => (
                  <SelectOption key={p.name} value={p.name}>
                    {p.name}
                    {p.configured ? " (configured)" : ""}
                  </SelectOption>
                ))}
              </Select>
              <p className="text-xs text-muted-foreground">
                Set up a new provider's credentials with{" "}
                <span className="font-mono">hermes memory setup</span>.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-3 border-t border-border pt-3">
              <span className="text-xs text-muted-foreground">
                Built-in files — MEMORY.md:{" "}
                {formatBytes(memory?.builtin_files.memory ?? 0)} · USER.md:{" "}
                {formatBytes(memory?.builtin_files.user ?? 0)}
              </span>
              <div className="flex items-center gap-2 ml-auto">
                <Button
                  size="sm"
                  ghost
                  className="text-destructive"
                  onClick={() => memoryReset.requestDelete("memory")}
                >
                  Reset MEMORY.md
                </Button>
                <Button
                  size="sm"
                  ghost
                  className="text-destructive"
                  onClick={() => memoryReset.requestDelete("user")}
                >
                  Reset USER.md
                </Button>
                <Button
                  size="sm"
                  ghost
                  className="text-destructive"
                  onClick={() => memoryReset.requestDelete("all")}
                >
                  Reset all
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* ── Credential pool ───────────────────────────────────────── */}
      <section className="flex flex-col gap-3">
        <H2 variant="sm" className="flex items-center gap-2 text-muted-foreground">
          <KeyRound className="h-4 w-4" /> Credential pool
        </H2>
        <Card>
          <CardContent className="flex flex-col gap-4 py-4">
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 items-end">
              <div className="grid gap-2">
                <Label htmlFor="cred-provider">Provider</Label>
                <Input
                  id="cred-provider"
                  value={credProvider}
                  onChange={(e) => setCredProvider(e.target.value)}
                  placeholder="openrouter"
                />
              </div>
              <div className="grid gap-2 sm:col-span-2">
                <Label htmlFor="cred-key">API key</Label>
                <Input
                  id="cred-key"
                  type="password"
                  value={credKey}
                  onChange={(e) => setCredKey(e.target.value)}
                  placeholder="sk-…"
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="cred-label">Label</Label>
                <Input
                  id="cred-label"
                  value={credLabel}
                  onChange={(e) => setCredLabel(e.target.value)}
                  placeholder="optional"
                />
              </div>
            </div>
            <div className="flex justify-end">
              <Button
                size="sm"
                className="uppercase"
                onClick={addCredential}
                disabled={addingCred}
                prefix={addingCred ? <Spinner /> : undefined}
              >
                Add key
              </Button>
            </div>

            {pool.length === 0 && (
              <p className="text-sm text-muted-foreground">
                No pooled credentials. Add one above to enable key rotation.
              </p>
            )}
            {pool.map((prov) => (
              <div key={prov.provider} className="flex flex-col gap-2">
                <span className="text-xs uppercase tracking-wider text-muted-foreground">
                  {prov.provider}
                </span>
                {prov.entries.map((entry) => (
                  <div
                    key={`${prov.provider}-${entry.index}`}
                    className="flex items-center gap-3 border border-border bg-background/40 px-3 py-2"
                  >
                    <span className="text-sm font-medium">{entry.label}</span>
                    <span className="font-mono text-xs text-muted-foreground">
                      {entry.token_preview}
                    </span>
                    <Badge tone="outline">{entry.auth_type}</Badge>
                    {entry.last_status && (
                      <Badge tone="secondary">{entry.last_status}</Badge>
                    )}
                    <Button
                      ghost
                      size="icon"
                      className="ml-auto text-destructive"
                      aria-label="Remove credential"
                      onClick={() =>
                        credDelete.requestDelete(
                          `${prov.provider}|${entry.index}`,
                        )
                      }
                    >
                      <Trash2 />
                    </Button>
                  </div>
                ))}
              </div>
            ))}
          </CardContent>
        </Card>
      </section>

      {/* ── Operations ────────────────────────────────────────────── */}
      <section className="flex flex-col gap-3">
        <H2 variant="sm" className="flex items-center gap-2 text-muted-foreground">
          <Activity className="h-4 w-4" /> Operations
        </H2>
        <Card>
          <CardContent className="flex flex-wrap gap-2 py-4">
            <Button
              size="sm"
              ghost
              prefix={<Stethoscope className="h-3.5 w-3.5" />}
              onClick={() => runOp(api.runDoctor, "Doctor")}
            >
              Run doctor
            </Button>
            <Button
              size="sm"
              ghost
              prefix={<ShieldCheck className="h-3.5 w-3.5" />}
              onClick={() => runOp(api.runSecurityAudit, "Security audit")}
            >
              Security audit
            </Button>
            <Button
              size="sm"
              ghost
              prefix={<Database className="h-3.5 w-3.5" />}
              onClick={() => runOp(() => api.runBackup(), "Backup")}
            >
              Backup
            </Button>
            <Button
              size="sm"
              ghost
              prefix={<RotateCw className="h-3.5 w-3.5" />}
              onClick={() => runOp(api.updateSkillsFromHub, "Skills update")}
            >
              Update skills
            </Button>
          </CardContent>
        </Card>

        {/* Import from backup */}
        <Card>
          <CardContent className="flex flex-col gap-3 py-4 sm:flex-row sm:items-end">
            <div className="grid gap-2 flex-1">
              <Label htmlFor="import-path">Restore from backup archive</Label>
              <Input
                id="import-path"
                value={importPath}
                onChange={(e) => setImportPath(e.target.value)}
                placeholder="/path/to/hermes-backup.zip"
              />
            </div>
            <Button
              size="sm"
              ghost
              disabled={!importPath.trim()}
              onClick={() => {
                if (!importPath.trim()) return;
                runOp(() => api.runImport(importPath.trim()), "Import");
              }}
            >
              Import
            </Button>
          </CardContent>
        </Card>
      </section>

      {/* ── Checkpoints ───────────────────────────────────────────── */}
      <section className="flex flex-col gap-3">
        <H2 variant="sm" className="flex items-center gap-2 text-muted-foreground">
          <Database className="h-4 w-4" /> Checkpoints
        </H2>
        <Card>
          <CardContent className="flex items-center justify-between py-4">
            <span className="text-sm text-muted-foreground">
              {checkpoints?.sessions.length ?? 0} session(s) ·{" "}
              {formatBytes(checkpoints?.total_bytes ?? 0)}
            </span>
            <Button
              size="sm"
              ghost
              className="text-destructive"
              disabled={!checkpoints?.sessions.length}
              prefix={<Trash2 className="h-3.5 w-3.5" />}
              onClick={() => checkpointsPrune.requestDelete("all")}
            >
              Prune
            </Button>
          </CardContent>
        </Card>
      </section>

      {/* ── Hooks ─────────────────────────────────────────────────── */}
      <section className="flex flex-col gap-3">
        <H2 variant="sm" className="flex items-center gap-2 text-muted-foreground">
          <Terminal className="h-4 w-4" /> Shell hooks
        </H2>
        {(!hooks || hooks.hooks.length === 0) && (
          <Card>
            <CardContent className="py-6 text-center text-sm text-muted-foreground">
              No shell hooks configured.
            </CardContent>
          </Card>
        )}
        {hooks?.hooks.map((h, i) => (
          <Card key={`${h.event}-${i}`}>
            <CardContent className="flex items-center gap-3 py-3">
              <Badge tone="outline">{h.event}</Badge>
              {h.matcher && (
                <span className="text-xs text-muted-foreground">
                  matcher: {h.matcher}
                </span>
              )}
              <span className="font-mono text-xs truncate flex-1">
                {h.command}
              </span>
              <Badge tone={h.allowed ? "success" : "warning"}>
                {h.allowed ? "allowed" : "not approved"}
              </Badge>
            </CardContent>
          </Card>
        ))}
      </section>
    </div>
  );
}
