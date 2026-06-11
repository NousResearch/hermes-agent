import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  Brain,
  Check,
  Clock,
  Copy,
  Cpu,
  Database,
  Download,
  Globe,
  HardDrive,
  KeyRound,
  Link2,
  Play,
  Plus,
  Power,
  RotateCw,
  Server,
  Share2,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  Terminal,
  Trash2,
  X,
} from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { H2 } from "@nous-research/ui/ui/components/typography/h2";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { Select, SelectOption } from "@nous-research/ui/ui/components/select";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { useConfirmDelete } from "@nous-research/ui/hooks/use-confirm-delete";
import { ConfirmDialog } from "@nous-research/ui/ui/components/confirm-dialog";
import { useModalBehavior } from "@/hooks/useModalBehavior";
import { DeleteConfirmDialog } from "@/components/DeleteConfirmDialog";
import { useI18n } from "@/i18n/context";
import { en } from "@/i18n/en";
import { cn, themedBody } from "@/lib/utils";
import { api } from "@/lib/api";
import type {
  StatusResponse,
  MemoryStatus,
  CredentialPoolProvider,
  CheckpointsResponse,
  HooksResponse,
  HookEntry,
  SystemStats,
  UpdateCheckResponse,
  CuratorStatus,
  PortalStatus,
  DebugShareResponse,
} from "@/lib/api";

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

function formatDuration(
  seconds: number,
  units: { day: string; hour: string; minute: string },
): string {
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (d > 0) return `${d}${units.day} ${h}${units.hour} ${m}${units.minute}`;
  if (h > 0) return `${h}${units.hour} ${m}${units.minute}`;
  return `${m}${units.minute}`;
}

function interpolate(
  template: string,
  values: Record<string, string | number>,
): string {
  return Object.entries(values).reduce(
    (result, [key, value]) => result.replaceAll(`{${key}}`, String(value)),
    template,
  );
}

/**
 * Live action-log viewer for the spawn-based admin actions (doctor, audit,
 * backup, import, skills update, checkpoints prune, gateway start/stop).
 * Polls /api/actions/<name>/status until the process exits.
 */
function ActionLogViewer({
  action,
  onClose,
}: {
  action: string;
  onClose: () => void;
}) {
  const { t } = useI18n();
  const copy = t.systemPage ?? en.systemPage!;
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
        if (st.running) timer.current = setTimeout(poll, 1200);
      } catch {
        if (!cancelled) setRunning(false);
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
              <Badge tone="warning">{copy.actionRunning}</Badge>
            ) : (
              <Badge tone={exitCode === 0 ? "success" : "destructive"}>
                {exitCode === 0
                  ? copy.actionDone
                  : interpolate(copy.actionExit, { code: exitCode ?? "—" })}
              </Badge>
            )}
          </div>
          <Button
            ghost
            size="icon"
            onClick={onClose}
            aria-label={copy.closeLog}
          >
            <X />
          </Button>
        </div>
        <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words bg-background/50 border border-border p-3 text-xs font-mono text-muted-foreground">
          {lines.length ? lines.join("\n") : copy.actionStarting}
        </pre>
      </CardContent>
    </Card>
  );
}

const HOOK_EVENTS_FALLBACK = [
  "pre_tool_call",
  "post_tool_call",
  "pre_llm_call",
  "post_llm_call",
  "on_session_start",
  "on_session_end",
];

export default function SystemPage() {
  const { toast, showToast } = useToast();
  const { t } = useI18n();
  const copy = t.systemPage ?? en.systemPage!;

  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [memory, setMemory] = useState<MemoryStatus | null>(null);
  const [pool, setPool] = useState<CredentialPoolProvider[]>([]);
  const [checkpoints, setCheckpoints] = useState<CheckpointsResponse | null>(
    null,
  );
  const [hooks, setHooks] = useState<HooksResponse | null>(null);
  const [curator, setCurator] = useState<CuratorStatus | null>(null);
  const [portal, setPortal] = useState<PortalStatus | null>(null);
  const [loading, setLoading] = useState(true);

  const [activeAction, setActiveAction] = useState<string | null>(null);

  // Add-credential form.
  const [credProvider, setCredProvider] = useState("openrouter");
  const [credKey, setCredKey] = useState("");
  const [credLabel, setCredLabel] = useState("");
  const [addingCred, setAddingCred] = useState(false);

  const [importPath, setImportPath] = useState("");
  // Restore-from-backup is destructive (overwrites the live config) and the
  // spawned `hermes import` runs non-interactively (stdin is /dev/null), so
  // its CLI "Continue? [y/N]" prompt would auto-abort. The dashboard owns the
  // consent: confirm here, then call the endpoint with force=true.
  const [importConfirmOpen, setImportConfirmOpen] = useState(false);

  // Create-hook modal.
  const [hookModalOpen, setHookModalOpen] = useState(false);
  const closeHookModal = useCallback(() => setHookModalOpen(false), []);
  const hookModalRef = useModalBehavior({
    open: hookModalOpen,
    onClose: closeHookModal,
  });
  const [hookEvent, setHookEvent] = useState("pre_tool_call");
  const [hookCommand, setHookCommand] = useState("");
  const [hookMatcher, setHookMatcher] = useState("");
  const [hookTimeout, setHookTimeout] = useState("");
  const [hookApprove, setHookApprove] = useState(true);
  const [creatingHook, setCreatingHook] = useState(false);

  // ── Update check ───────────────────────────────────────────────────
  const [updateInfo, setUpdateInfo] = useState<UpdateCheckResponse | null>(
    null,
  );
  const [checkingUpdate, setCheckingUpdate] = useState(false);
  const [updateConfirmOpen, setUpdateConfirmOpen] = useState(false);

  const loadAll = useCallback(() => {
    Promise.allSettled([
      api.getStatus(),
      api.getSystemStats(),
      api.getMemory(),
      api.getCredentialPool(),
      api.getCheckpoints(),
      api.getHooks(),
      api.getCurator(),
      api.getPortal(),
      // Cached (non-forced) check so the version row shows update status on
      // load without a separate effect / a forced network round-trip.
      api.checkHermesUpdate(false),
    ])
      .then(([s, st, m, p, c, h, cur, prt, upd]) => {
        if (s.status === "fulfilled") setStatus(s.value);
        if (st.status === "fulfilled") setStats(st.value);
        if (m.status === "fulfilled") setMemory(m.value);
        if (p.status === "fulfilled") setPool(p.value.providers);
        if (c.status === "fulfilled") setCheckpoints(c.value);
        if (h.status === "fulfilled") setHooks(h.value);
        if (cur.status === "fulfilled") setCurator(cur.value);
        if (prt.status === "fulfilled") setPortal(prt.value);
        if (upd.status === "fulfilled") setUpdateInfo(upd.value);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    loadAll();
  }, [loadAll]);

  // ── Gateway lifecycle ──────────────────────────────────────────────
  const runGateway = async (verb: "start" | "stop" | "restart") => {
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
      const localizedVerb = {
        start: copy.gatewayVerbStart,
        stop: copy.gatewayVerbStop,
        restart: copy.gatewayVerbRestart,
      }[verb];
      showToast(
        interpolate(copy.gatewayStarted, { verb: localizedVerb }),
        "success",
      );
      setTimeout(loadAll, 3000);
    } catch (e) {
      const localizedVerb = {
        start: copy.gatewayVerbStart,
        stop: copy.gatewayVerbStop,
        restart: copy.gatewayVerbRestart,
      }[verb];
      showToast(
        interpolate(copy.gatewayFailed, {
          verb: localizedVerb,
          error: String(e),
        }),
        "error",
      );
    }
  };

  // ── Curator ────────────────────────────────────────────────────────
  const toggleCuratorPaused = async () => {
    if (!curator) return;
    try {
      await api.setCuratorPaused(!curator.paused);
      showToast(
        curator.paused ? copy.curatorResumed : copy.curatorPaused,
        "success",
      );
      loadAll();
    } catch (e) {
      showToast(
        interpolate(copy.curatorToggleFailed, { error: String(e) }),
        "error",
      );
    }
  };

  // ── Memory ─────────────────────────────────────────────────────────
  // Memory provider selection lives on the /plugins page now (see the
  // read-only display + link below); the dropdown was intentionally
  // dropped from this card during the admin-panel refresh.
  const memoryReset = useConfirmDelete({
    onDelete: useCallback(
      async (target: string) => {
        try {
          const res = await api.resetMemory(
            target as "all" | "memory" | "user",
          );
          showToast(
            interpolate(copy.resetResult, {
              items: res.deleted.join(", ") || copy.nothing,
            }),
            "success",
          );
          loadAll();
        } catch (e) {
          showToast(
            interpolate(copy.resetFailed, { error: String(e) }),
            "error",
          );
          throw e;
        }
      },
      [copy, loadAll, showToast],
    ),
  });

  // ── Credential pool ────────────────────────────────────────────────
  const addCredential = async () => {
    if (!credProvider.trim() || !credKey.trim()) {
      showToast(copy.providerKeyRequired, "error");
      return;
    }
    setAddingCred(true);
    try {
      await api.addCredentialPoolEntry(
        credProvider.trim(),
        credKey.trim(),
        credLabel.trim() || undefined,
      );
      showToast(copy.credentialAdded, "success");
      setCredKey("");
      setCredLabel("");
      loadAll();
    } catch (e) {
      showToast(
        interpolate(copy.credentialAddFailed, { error: String(e) }),
        "error",
      );
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
          showToast(copy.credentialRemoved, "success");
          loadAll();
        } catch (e) {
          showToast(
            interpolate(copy.removeFailed, { error: String(e) }),
            "error",
          );
          throw e;
        }
      },
      [copy, loadAll, showToast],
    ),
  });

  // ── Operations ─────────────────────────────────────────────────────
  const runOp = async (fn: () => Promise<{ name: string }>, label: string) => {
    try {
      const res = await fn();
      setActiveAction(res.name);
      showToast(interpolate(copy.operationStarted, { label }), "success");
    } catch (e) {
      showToast(
        interpolate(copy.operationFailed, { label, error: String(e) }),
        "error",
      );
    }
  };

  // ── Debug share ────────────────────────────────────────────────────
  // Unlike the fire-and-forget ops above, `debug share` produces shareable
  // paste URLs that are the whole point — so we surface them as real,
  // copyable links rather than a log tail.
  const [shareRedact, setShareRedact] = useState(true);
  const [sharing, setSharing] = useState(false);
  const [shareResult, setShareResult] = useState<DebugShareResponse | null>(
    null,
  );
  const [copiedLabel, setCopiedLabel] = useState<string | null>(null);

  const copyToClipboard = useCallback(
    async (text: string, label: string) => {
      try {
        await navigator.clipboard.writeText(text);
        setCopiedLabel(label);
        setTimeout(
          () => setCopiedLabel((cur) => (cur === label ? null : cur)),
          1500,
        );
      } catch {
        showToast(copy.copyFailed, "error");
      }
    },
    [copy, showToast],
  );

  const runDebugShare = useCallback(async () => {
    setSharing(true);
    setShareResult(null);
    try {
      const res = await api.runDebugShare({ redact: shareRedact });
      setShareResult(res);
      const n = Object.keys(res.urls).length;
      showToast(
        interpolate(copy.pastesUploaded, {
          count: n,
          redacted: res.redacted ? copy.redactedSuffix : "",
        }),
        "success",
      );
    } catch (e) {
      showToast(
        interpolate(copy.debugShareFailed, { error: String(e) }),
        "error",
      );
    } finally {
      setSharing(false);
    }
  }, [copy, shareRedact, showToast]);

  // ── Update check / apply ───────────────────────────────────────────
  const checkForUpdate = useCallback(
    async (force = false) => {
      setCheckingUpdate(true);
      try {
        const info = await api.checkHermesUpdate(force);
        setUpdateInfo(info);
        if (force) {
          if (info.update_available) {
            showToast(
              info.behind && info.behind > 0
                ? interpolate(copy.updateAvailableBehind, {
                    count: info.behind,
                  })
                : copy.updateAvailable,
              "success",
            );
          } else if (info.behind === 0) {
            showToast(copy.latestVersion, "success");
          } else if (info.message) {
            showToast(info.message, "error");
          }
        }
      } catch (e) {
        showToast(
          interpolate(copy.updateCheckFailed, { error: String(e) }),
          "error",
        );
      } finally {
        setCheckingUpdate(false);
      }
    },
    [copy, showToast],
  );

  // Auto-check (cached) runs inside loadAll on mount; this is the
  // user-triggered forced re-check from the "Check for updates" button.
  const applyUpdate = async () => {
    setUpdateConfirmOpen(false);
    try {
      const resp = await api.updateHermes();
      if (!resp.ok && resp.error === "docker_update_unsupported") {
        showToast(resp.message ?? copy.dockerUpdateUnsupported, "error");
        return;
      }
      setActiveAction(resp.name ?? "hermes-update");
      showToast(copy.updateStarted, "success");
    } catch (e) {
      showToast(interpolate(copy.updateFailed, { error: String(e) }), "error");
    }
  };

  const checkpointsPrune = useConfirmDelete({
    onDelete: useCallback(async () => {
      try {
        const res = await api.pruneCheckpoints();
        setActiveAction(res.name);
        showToast(copy.pruneStarted, "success");
      } catch (e) {
        showToast(interpolate(copy.pruneFailed, { error: String(e) }), "error");
        throw e;
      }
    }, [copy, showToast]),
  });

  // ── Hooks ──────────────────────────────────────────────────────────
  const createHook = async () => {
    if (!hookCommand.trim()) {
      showToast(copy.commandRequired, "error");
      return;
    }
    setCreatingHook(true);
    try {
      await api.createHook({
        event: hookEvent,
        command: hookCommand.trim(),
        matcher: hookMatcher.trim() || undefined,
        timeout: hookTimeout.trim() ? Number(hookTimeout) : undefined,
        approve: hookApprove,
      });
      showToast(copy.hookCreated, "success");
      setHookCommand("");
      setHookMatcher("");
      setHookTimeout("");
      setHookModalOpen(false);
      loadAll();
    } catch (e) {
      showToast(
        interpolate(copy.hookCreateFailed, { error: String(e) }),
        "error",
      );
    } finally {
      setCreatingHook(false);
    }
  };

  const hookDelete = useConfirmDelete({
    onDelete: useCallback(
      async (key: string) => {
        const sep = key.indexOf("|");
        const event = key.slice(0, sep);
        const command = key.slice(sep + 1);
        try {
          await api.deleteHook(event, command);
          showToast(copy.hookRemoved, "success");
          loadAll();
        } catch (e) {
          showToast(
            interpolate(copy.hookRemoveFailed, { error: String(e) }),
            "error",
          );
          throw e;
        }
      },
      [copy, loadAll, showToast],
    ),
  });

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner className="text-2xl text-primary" />
      </div>
    );
  }

  const gatewayRunning = status?.gateway_running;
  const validEvents = hooks?.valid_events?.length
    ? hooks.valid_events
    : HOOK_EVENTS_FALLBACK;

  return (
    <div className="flex flex-col gap-8">
      <Toast toast={toast} />

      <ConfirmDialog
        open={updateConfirmOpen}
        onCancel={() => setUpdateConfirmOpen(false)}
        onConfirm={() => void applyUpdate()}
        title={copy.updateDialogTitle}
        description={
          updateInfo && updateInfo.behind && updateInfo.behind > 0
            ? interpolate(copy.updateDialogBehind, {
                command: updateInfo.update_command,
                count: updateInfo.behind,
              })
            : interpolate(copy.updateDialogDefault, {
                command: updateInfo?.update_command ?? "hermes update",
              })
        }
        confirmLabel={copy.updateNow}
      />

      <DeleteConfirmDialog
        open={memoryReset.isOpen}
        onCancel={memoryReset.cancel}
        onConfirm={memoryReset.confirm}
        title={copy.resetMemoryTitle}
        description={copy.resetMemoryDescription}
        loading={memoryReset.isDeleting}
      />
      <DeleteConfirmDialog
        open={credDelete.isOpen}
        onCancel={credDelete.cancel}
        onConfirm={credDelete.confirm}
        title={copy.removeCredentialTitle}
        description={copy.removeCredentialDescription}
        loading={credDelete.isDeleting}
      />
      <DeleteConfirmDialog
        open={checkpointsPrune.isOpen}
        onCancel={checkpointsPrune.cancel}
        onConfirm={checkpointsPrune.confirm}
        title={copy.pruneCheckpointsTitle}
        description={copy.pruneCheckpointsDescription}
        loading={checkpointsPrune.isDeleting}
      />
      <DeleteConfirmDialog
        open={hookDelete.isOpen}
        onCancel={hookDelete.cancel}
        onConfirm={hookDelete.confirm}
        title={copy.removeHookTitle}
        description={copy.removeHookDescription}
        loading={hookDelete.isDeleting}
      />

      {/* Create-hook modal */}
      {hookModalOpen && (
        <div
          ref={hookModalRef}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-background/85 backdrop-blur-sm p-4"
          onClick={(e) =>
            e.target === e.currentTarget && setHookModalOpen(false)
          }
          role="dialog"
          aria-modal="true"
        >
          <div
            className={cn(
              themedBody,
              "relative w-full max-w-lg border border-border bg-card shadow-2xl flex flex-col",
            )}
          >
            <Button
              ghost
              size="icon"
              onClick={() => setHookModalOpen(false)}
              className="absolute right-2 top-2 text-muted-foreground hover:text-foreground"
              aria-label={copy.close}
            >
              <X />
            </Button>
            <header className="p-5 pb-3 border-b border-border">
              <h2 className="font-mondwest text-display text-base tracking-wider">
                {copy.newShellHook}
              </h2>
            </header>
            <div className="p-5 grid gap-4">
              <div className="grid gap-2">
                <Label htmlFor="hook-event">{copy.event}</Label>
                <Select
                  id="hook-event"
                  value={hookEvent}
                  onValueChange={(v) => setHookEvent(v)}
                >
                  {validEvents.map((ev) => (
                    <SelectOption key={ev} value={ev}>
                      {ev}
                    </SelectOption>
                  ))}
                </Select>
              </div>
              <div className="grid gap-2">
                <Label htmlFor="hook-command">{copy.commandAbsolute}</Label>
                <Input
                  id="hook-command"
                  autoFocus
                  placeholder="/usr/local/bin/my-hook.sh"
                  value={hookCommand}
                  onChange={(e) => setHookCommand(e.target.value)}
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="grid gap-2">
                  <Label htmlFor="hook-matcher">{copy.matcherOptional}</Label>
                  <Input
                    id="hook-matcher"
                    placeholder={copy.matcherPlaceholder}
                    value={hookMatcher}
                    onChange={(e) => setHookMatcher(e.target.value)}
                  />
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="hook-timeout">{copy.timeoutSeconds}</Label>
                  <Input
                    id="hook-timeout"
                    placeholder="10"
                    value={hookTimeout}
                    onChange={(e) => setHookTimeout(e.target.value)}
                  />
                </div>
              </div>
              <label className="flex items-center gap-2 text-sm text-muted-foreground">
                <input
                  type="checkbox"
                  checked={hookApprove}
                  onChange={(e) => setHookApprove(e.target.checked)}
                />
                {copy.approveNow}
              </label>
              <p className="text-xs text-warning">{copy.hookSafety}</p>
              <div className="flex justify-end">
                <Button
                  className="uppercase"
                  size="sm"
                  onClick={createHook}
                  disabled={creatingHook}
                  prefix={creatingHook ? <Spinner /> : undefined}
                >
                  {creatingHook ? copy.creating : copy.createHook}
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Live action log */}
      {activeAction && (
        <ActionLogViewer
          action={activeAction}
          onClose={() => setActiveAction(null)}
        />
      )}

      {/* ── Host / system stats ───────────────────────────────────── */}
      <section className="flex flex-col gap-3">
        <H2
          variant="sm"
          className="flex items-center gap-2 text-muted-foreground"
        >
          <Server className="h-4 w-4" /> {copy.host}
        </H2>
        <Card>
          <CardContent className="py-4">
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-y-3 gap-x-6 text-sm">
              <div>
                <div className="text-xs uppercase tracking-wider text-muted-foreground">
                  {copy.os}
                </div>
                <div>
                  {stats?.os} {stats?.os_release}
                </div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-wider text-muted-foreground">
                  {copy.arch}
                </div>
                <div>{stats?.arch}</div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-wider text-muted-foreground">
                  {copy.host}
                </div>
                <div className="truncate">{stats?.hostname}</div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-wider text-muted-foreground">
                  {copy.python}
                </div>
                <div>
                  {stats?.python_impl} {stats?.python_version}
                </div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-wider text-muted-foreground">
                  {copy.hermes}
                </div>
                <div className="flex items-center gap-2">
                  <span>v{stats?.hermes_version}</span>
                  {updateInfo &&
                    (updateInfo.update_available ? (
                      <Badge tone="warning">
                        {updateInfo.behind && updateInfo.behind > 0
                          ? interpolate(copy.behind, {
                              count: updateInfo.behind,
                            })
                          : copy.updateAvailable}
                      </Badge>
                    ) : updateInfo.behind === 0 ? (
                      <Badge tone="success">{copy.latest}</Badge>
                    ) : null)}
                </div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-wider text-muted-foreground flex items-center gap-1">
                  <Cpu className="h-3 w-3" /> {copy.cpu}
                </div>
                <div>
                  {stats?.cpu_count ?? "—"} {copy.cores}
                  {typeof stats?.cpu_percent === "number"
                    ? ` · ${stats.cpu_percent.toFixed(0)}%`
                    : ""}
                </div>
              </div>
              {stats?.memory && (
                <div>
                  <div className="text-xs uppercase tracking-wider text-muted-foreground">
                    {copy.memory}
                  </div>
                  <div>
                    {formatBytes(stats.memory.used)} /{" "}
                    {formatBytes(stats.memory.total)} ({stats.memory.percent}%)
                  </div>
                </div>
              )}
              {stats?.disk && (
                <div>
                  <div className="text-xs uppercase tracking-wider text-muted-foreground flex items-center gap-1">
                    <HardDrive className="h-3 w-3" /> {copy.disk}
                  </div>
                  <div>
                    {formatBytes(stats.disk.used)} /{" "}
                    {formatBytes(stats.disk.total)} ({stats.disk.percent}%)
                  </div>
                </div>
              )}
              {typeof stats?.uptime_seconds === "number" && (
                <div>
                  <div className="text-xs uppercase tracking-wider text-muted-foreground">
                    {copy.uptime}
                  </div>
                  <div>
                    {formatDuration(stats.uptime_seconds, {
                      day: copy.dayUnit,
                      hour: copy.hourUnit,
                      minute: copy.minuteUnit,
                    })}
                  </div>
                </div>
              )}
              {stats?.load_avg && stats.load_avg.length >= 3 && (
                <div>
                  <div className="text-xs uppercase tracking-wider text-muted-foreground">
                    {copy.loadAverage}
                  </div>
                  <div>
                    {stats.load_avg.map((n) => n.toFixed(2)).join(" / ")}
                  </div>
                </div>
              )}
            </div>
            {stats && !stats.psutil && (
              <p className="mt-3 text-xs text-muted-foreground">
                {copy.psutilHint}
              </p>
            )}
            <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-border pt-4">
              <Button
                size="sm"
                ghost
                disabled={checkingUpdate}
                prefix={
                  checkingUpdate ? (
                    <Spinner className="h-3.5 w-3.5" />
                  ) : (
                    <RotateCw className="h-3.5 w-3.5" />
                  )
                }
                onClick={() => void checkForUpdate(true)}
              >
                {copy.checkUpdates}
              </Button>
              {updateInfo?.update_available && updateInfo.can_apply && (
                <Button
                  size="sm"
                  prefix={<Download className="h-3.5 w-3.5" />}
                  onClick={() => setUpdateConfirmOpen(true)}
                >
                  {copy.updateNow}
                </Button>
              )}
              {updateInfo &&
                !updateInfo.can_apply &&
                updateInfo.update_available && (
                  <span className="text-xs text-muted-foreground">
                    {copy.updateWith}{" "}
                    <span className="font-mono">
                      {updateInfo.update_command}
                    </span>
                  </span>
                )}
              {updateInfo?.message && !updateInfo.update_available && (
                <span className="text-xs text-muted-foreground">
                  {updateInfo.message}
                </span>
              )}
            </div>
          </CardContent>
        </Card>
      </section>

      {/* ── Portal ────────────────────────────────────────────────── */}
      <section className="flex flex-col gap-3">
        <H2
          variant="sm"
          className="flex items-center gap-2 text-muted-foreground"
        >
          <Globe className="h-4 w-4" /> {copy.nousPortal}
        </H2>
        <Card>
          <CardContent className="flex flex-col gap-3 py-4">
            <div className="flex items-center gap-3">
              <Badge tone={portal?.logged_in ? "success" : "secondary"}>
                {portal?.logged_in ? copy.loggedIn : copy.notLoggedIn}
              </Badge>
              {portal?.provider && (
                <span className="text-sm text-muted-foreground">
                  {interpolate(copy.inferenceProvider, {
                    provider: portal.provider,
                  })}
                </span>
              )}
              <a
                href={
                  portal?.subscription_url ||
                  "https://portal.nousresearch.com/manage-subscription"
                }
                target="_blank"
                rel="noreferrer"
                className="ml-auto text-xs text-primary underline"
              >
                {copy.manageSubscription}
              </a>
            </div>
            {portal?.features && portal.features.length > 0 && (
              <div className="flex flex-col gap-1 border-t border-border pt-3">
                <span className="text-xs uppercase tracking-wider text-muted-foreground">
                  {copy.toolGatewayRouting}
                </span>
                {portal.features.map((f) => (
                  <div
                    key={f.label}
                    className="flex items-center justify-between text-sm"
                  >
                    <span>{f.label}</span>
                    <span className="text-muted-foreground">{f.state}</span>
                  </div>
                ))}
              </div>
            )}
            {!portal?.logged_in && (
              <p className="text-xs text-muted-foreground">
                {copy.portalLoginHint}
              </p>
            )}
          </CardContent>
        </Card>
      </section>

      {/* ── Curator ───────────────────────────────────────────────── */}
      <section className="flex flex-col gap-3">
        <H2
          variant="sm"
          className="flex items-center gap-2 text-muted-foreground"
        >
          <Sparkles className="h-4 w-4" /> {copy.skillCurator}
        </H2>
        <Card>
          <CardContent className="flex items-center justify-between py-4">
            <div className="flex items-center gap-3">
              <Badge
                tone={
                  curator?.paused
                    ? "warning"
                    : curator?.enabled
                      ? "success"
                      : "secondary"
                }
              >
                {curator?.paused
                  ? copy.paused
                  : curator?.enabled
                    ? copy.active
                    : copy.disabled}
              </Badge>
              <span className="text-sm text-muted-foreground">
                {curator?.interval_hours
                  ? interpolate(copy.everyHours, {
                      count: curator.interval_hours,
                    })
                  : ""}
                {curator?.last_run_at
                  ? ` · ${interpolate(copy.lastRun, {
                      date: new Date(curator.last_run_at).toLocaleString(),
                    })}`
                  : ` · ${copy.neverRun}`}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Button size="sm" ghost onClick={toggleCuratorPaused}>
                {curator?.paused ? copy.resume : copy.pause}
              </Button>
              <Button
                size="sm"
                ghost
                prefix={<Play className="h-3.5 w-3.5" />}
                onClick={() => runOp(api.runCurator, copy.curatorReview)}
              >
                {copy.runNow}
              </Button>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* ── Gateway ───────────────────────────────────────────────── */}
      <section className="flex flex-col gap-3">
        <H2
          variant="sm"
          className="flex items-center gap-2 text-muted-foreground"
        >
          <Power className="h-4 w-4" /> {copy.gateway}
        </H2>
        <Card>
          <CardContent className="flex items-center justify-between py-4">
            <div className="flex items-center gap-3">
              <Badge tone={gatewayRunning ? "success" : "secondary"}>
                {gatewayRunning ? copy.running : copy.stopped}
              </Badge>
              <span className="text-sm text-muted-foreground">
                {status?.gateway_state ?? "—"}
                {status?.gateway_pid
                  ? ` · ${interpolate(copy.pid, {
                      value: status.gateway_pid,
                    })}`
                  : ""}
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
                {copy.start}
              </Button>
              <Button
                size="sm"
                className="uppercase"
                onClick={() => runGateway("restart")}
                prefix={<RotateCw className="h-3.5 w-3.5" />}
              >
                {copy.restart}
              </Button>
              <Button
                size="sm"
                className="uppercase text-warning"
                ghost
                onClick={() => runGateway("stop")}
                disabled={!gatewayRunning}
                prefix={<Power className="h-3.5 w-3.5" />}
              >
                {copy.stop}
              </Button>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* ── Memory ────────────────────────────────────────────────── */}
      <section className="flex flex-col gap-3">
        <H2
          variant="sm"
          className="flex items-center gap-2 text-muted-foreground"
        >
          <Brain className="h-4 w-4" /> {copy.memory}
        </H2>
        <Card>
          <CardContent className="flex flex-col gap-4 py-4">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
              <span>
                {copy.externalProvider}{" "}
                <span className="font-mono text-foreground">
                  {memory?.active || copy.builtInOnly}
                </span>
              </span>
              <Link to="/plugins" className="underline">
                {copy.changeInPlugins}
              </Link>
              <span className="ml-auto">
                {copy.newCredentials}{" "}
                <span className="font-mono">hermes memory setup</span>
              </span>
            </div>

            <div className="flex flex-wrap items-center gap-3 border-t border-border pt-3">
              <span className="text-xs text-muted-foreground">
                {copy.builtInFiles} — MEMORY.md:{" "}
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
                  {copy.resetMemoryFile}
                </Button>
                <Button
                  size="sm"
                  ghost
                  className="text-destructive"
                  onClick={() => memoryReset.requestDelete("user")}
                >
                  {copy.resetUserFile}
                </Button>
                <Button
                  size="sm"
                  ghost
                  className="text-destructive"
                  onClick={() => memoryReset.requestDelete("all")}
                >
                  {copy.resetAll}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* ── Credential pool ───────────────────────────────────────── */}
      <section className="flex flex-col gap-3">
        <H2
          variant="sm"
          className="flex items-center gap-2 text-muted-foreground"
        >
          <KeyRound className="h-4 w-4" /> {copy.credentialPool}
        </H2>
        <Card>
          <CardContent className="flex flex-col gap-4 py-4">
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 items-end">
              <div className="grid gap-2">
                <Label htmlFor="cred-provider">{copy.provider}</Label>
                <Input
                  id="cred-provider"
                  value={credProvider}
                  onChange={(e) => setCredProvider(e.target.value)}
                  placeholder="openrouter"
                />
              </div>
              <div className="grid gap-2 sm:col-span-2">
                <Label htmlFor="cred-key">{copy.apiKey}</Label>
                <Input
                  id="cred-key"
                  type="password"
                  value={credKey}
                  onChange={(e) => setCredKey(e.target.value)}
                  placeholder="sk-…"
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="cred-label">{copy.label}</Label>
                <Input
                  id="cred-label"
                  value={credLabel}
                  onChange={(e) => setCredLabel(e.target.value)}
                  placeholder={copy.optional}
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
                {copy.addKey}
              </Button>
            </div>
            {pool.length === 0 && (
              <p className="text-sm text-muted-foreground">
                {copy.noPooledCredentials}
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
                      aria-label={copy.removeCredential}
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
        <H2
          variant="sm"
          className="flex items-center gap-2 text-muted-foreground"
        >
          <Activity className="h-4 w-4" /> {copy.operations}
        </H2>
        <Card>
          <CardContent className="flex flex-wrap gap-2 py-4">
            <Button
              size="sm"
              ghost
              prefix={<Stethoscope className="h-3.5 w-3.5" />}
              onClick={() => runOp(api.runDoctor, copy.doctor)}
            >
              {copy.runDoctor}
            </Button>
            <Button
              size="sm"
              ghost
              prefix={<ShieldCheck className="h-3.5 w-3.5" />}
              onClick={() => runOp(api.runSecurityAudit, copy.securityAudit)}
            >
              {copy.securityAudit}
            </Button>
            <Button
              size="sm"
              ghost
              prefix={<Database className="h-3.5 w-3.5" />}
              onClick={() => runOp(() => api.runBackup(), copy.backup)}
            >
              {copy.createBackup}
            </Button>
            <Button
              size="sm"
              ghost
              prefix={<RotateCw className="h-3.5 w-3.5" />}
              onClick={() => runOp(api.updateSkillsFromHub, copy.skillsUpdate)}
            >
              {copy.updateSkills}
            </Button>
            <Button
              size="sm"
              ghost
              prefix={<Activity className="h-3.5 w-3.5" />}
              onClick={() => runOp(api.runPromptSize, copy.promptSize)}
            >
              {copy.promptSize}
            </Button>
            <Button
              size="sm"
              ghost
              prefix={<Database className="h-3.5 w-3.5" />}
              onClick={() => runOp(api.runDump, copy.supportDump)}
            >
              {copy.supportDump}
            </Button>
            <Button
              size="sm"
              ghost
              prefix={<RotateCw className="h-3.5 w-3.5" />}
              onClick={() => runOp(api.runConfigMigrate, copy.configMigrate)}
            >
              {copy.migrateConfig}
            </Button>
          </CardContent>
        </Card>

        {/* Debug share — uploads a redacted report + logs, returns shareable
            links. Separated from the buttons above because its output is
            persistent, copyable URLs, not a fire-and-forget log tail. */}
        <Card>
          <CardContent className="flex flex-col gap-3 py-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-start gap-2">
                <Share2 className="h-4 w-4 mt-0.5 text-muted-foreground" />
                <div className="flex flex-col">
                  <span className="text-sm font-medium">
                    {copy.shareDebugReport}
                  </span>
                  <span className="text-xs text-muted-foreground max-w-prose">
                    {copy.shareDebugDescription}
                  </span>
                </div>
              </div>
              <Button
                size="sm"
                disabled={sharing}
                prefix={
                  sharing ? (
                    <Spinner className="h-3.5 w-3.5" />
                  ) : (
                    <Share2 className="h-3.5 w-3.5" />
                  )
                }
                onClick={() => void runDebugShare()}
              >
                {sharing ? copy.uploading : copy.generateShareLink}
              </Button>
            </div>

            <label className="flex items-center gap-2 text-xs text-muted-foreground select-none">
              <input
                type="checkbox"
                className="accent-current"
                checked={shareRedact}
                disabled={sharing}
                onChange={(e) => setShareRedact(e.target.checked)}
              />
              {copy.redactBeforeUpload}
            </label>

            {shareResult && (
              <div className="flex flex-col gap-2 border-t border-border pt-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Badge tone="success">{copy.uploaded}</Badge>
                    {shareResult.redacted ? (
                      <Badge tone="outline">{copy.redacted}</Badge>
                    ) : (
                      <Badge tone="warning">{copy.notRedacted}</Badge>
                    )}
                    <span className="flex items-center gap-1 text-xs text-muted-foreground">
                      <Clock className="h-3 w-3" />
                      {interpolate(copy.autoDeletesHours, {
                        count: Math.round(
                          shareResult.auto_delete_seconds / 3600,
                        ),
                      })}
                    </span>
                  </div>
                  {Object.keys(shareResult.urls).length > 1 && (
                    <Button
                      size="sm"
                      ghost
                      prefix={
                        copiedLabel === "__all__" ? (
                          <Check className="h-3.5 w-3.5" />
                        ) : (
                          <Copy className="h-3.5 w-3.5" />
                        )
                      }
                      onClick={() =>
                        void copyToClipboard(
                          Object.entries(shareResult.urls)
                            .map(([label, url]) => `${label}: ${url}`)
                            .join("\n"),
                          "__all__",
                        )
                      }
                    >
                      {copy.copyAll}
                    </Button>
                  )}
                </div>

                {Object.entries(shareResult.urls).map(([label, url]) => (
                  <div
                    key={label}
                    className="flex items-center gap-2 bg-background/50 border border-border px-3 py-2"
                  >
                    <Link2 className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                    <span className="font-mono text-xs shrink-0 w-24 truncate text-muted-foreground">
                      {label}
                    </span>
                    <a
                      href={url}
                      target="_blank"
                      rel="noreferrer"
                      className="font-mono text-xs truncate flex-1 text-primary hover:underline"
                    >
                      {url}
                    </a>
                    <Button
                      ghost
                      size="icon"
                      aria-label={interpolate(copy.copyLink, { label })}
                      onClick={() => void copyToClipboard(url, label)}
                    >
                      {copiedLabel === label ? <Check /> : <Copy />}
                    </Button>
                  </div>
                ))}

                {shareResult.failures.length > 0 && (
                  <span className="text-xs text-destructive">
                    {interpolate(copy.logsUploadFailed, {
                      errors: shareResult.failures.join("; "),
                    })}
                  </span>
                )}
              </div>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardContent className="flex flex-col gap-3 py-4 sm:flex-row sm:items-end">
            <div className="grid gap-2 flex-1">
              <Label htmlFor="import-path">{copy.restoreArchive}</Label>
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
                setImportConfirmOpen(true);
              }}
            >
              {copy.import}
            </Button>
            <ConfirmDialog
              open={importConfirmOpen}
              title={copy.restoreTitle}
              description={interpolate(copy.restoreDescription, {
                archive: importPath.trim() || copy.archiveFallback,
              })}
              destructive
              confirmLabel={copy.restore}
              cancelLabel={copy.cancel}
              onCancel={() => setImportConfirmOpen(false)}
              onConfirm={() => {
                setImportConfirmOpen(false);
                runOp(
                  () => api.runImport(importPath.trim(), true),
                  copy.import,
                );
              }}
            />
          </CardContent>
        </Card>
      </section>

      {/* ── Checkpoints ───────────────────────────────────────────── */}
      <section className="flex flex-col gap-3">
        <H2
          variant="sm"
          className="flex items-center gap-2 text-muted-foreground"
        >
          <Database className="h-4 w-4" /> {copy.checkpoints}
        </H2>
        <Card>
          <CardContent className="flex items-center justify-between py-4">
            <span className="text-sm text-muted-foreground">
              {interpolate(copy.sessionsCount, {
                count: checkpoints?.sessions.length ?? 0,
              })}{" "}
              · {formatBytes(checkpoints?.total_bytes ?? 0)}
            </span>
            <Button
              size="sm"
              ghost
              className="text-destructive"
              disabled={!checkpoints?.sessions.length}
              prefix={<Trash2 className="h-3.5 w-3.5" />}
              onClick={() => checkpointsPrune.requestDelete("all")}
            >
              {copy.prune}
            </Button>
          </CardContent>
        </Card>
      </section>

      {/* ── Shell hooks ───────────────────────────────────────────── */}
      <section className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <H2
            variant="sm"
            className="flex items-center gap-2 text-muted-foreground"
          >
            <Terminal className="h-4 w-4" /> {copy.shellHooks}
          </H2>
          <Button
            size="sm"
            className="uppercase"
            prefix={<Plus className="h-3.5 w-3.5" />}
            onClick={() => setHookModalOpen(true)}
          >
            {copy.newHook}
          </Button>
        </div>
        {(!hooks || hooks.hooks.length === 0) && (
          <Card>
            <CardContent className="py-6 text-center text-sm text-muted-foreground">
              {copy.noHooks}
            </CardContent>
          </Card>
        )}
        {hooks?.hooks.map((h: HookEntry, i) => (
          <Card key={`${h.event}-${i}`}>
            <CardContent className="flex items-center gap-3 py-3">
              <Badge tone="outline">{h.event}</Badge>
              {h.matcher && (
                <span className="text-xs text-muted-foreground">
                  {interpolate(copy.matcher, { value: h.matcher })}
                </span>
              )}
              <span className="font-mono text-xs truncate flex-1">
                {h.command}
              </span>
              {h.executable === false && (
                <Badge tone="destructive">{copy.notExecutable}</Badge>
              )}
              <Badge tone={h.allowed ? "success" : "warning"}>
                {h.allowed ? copy.allowed : copy.notApproved}
              </Badge>
              <Button
                ghost
                size="icon"
                className="text-destructive"
                aria-label={copy.removeHook}
                onClick={() =>
                  hookDelete.requestDelete(`${h.event}|${h.command ?? ""}`)
                }
              >
                <Trash2 />
              </Button>
            </CardContent>
          </Card>
        ))}
      </section>
    </div>
  );
}
