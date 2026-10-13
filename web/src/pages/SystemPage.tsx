import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router";
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
  Upload,
  X,
} from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { H2 } from "@nous-research/ui/ui/components/typography/h2";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Checkbox } from "@nous-research/ui/ui/components/checkbox";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { Select, SelectOption } from "@nous-research/ui/ui/components/select";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { useConfirmDelete } from "@nous-research/ui/hooks/use-confirm-delete";
import { ConfirmDialog } from "@nous-research/ui/ui/components/confirm-dialog";
import { useModalBehavior } from "@/hooks/useModalBehavior";
import { DeleteConfirmDialog } from "@/components/DeleteConfirmDialog";
import { HermesConsoleModal } from "@/components/HermesConsoleModal";
import { cn, themedBody } from "@/lib/utils";
import { api } from "@/lib/api";
import { copyTextToClipboard } from "@/lib/clipboard";
import {
  gatewayStateNeedsLogs,
  gatewayStateDescription,
  gatewayActionFailedMessage,
  servedProfileRefusal,
  sharedGatewayProfiles,
  sharedGatewayRestartDescription,
  sharedGatewayRestartedMessage,
} from "@/lib/shared-gateway";
import type {
  StatusResponse,
  MemoryStatus,
  MemoryProviderInfo,
  CredentialPoolProvider,
  CheckpointsResponse,
  HooksResponse,
  HookEntry,
  SystemStats,
  UpdateCheckResponse,
  CuratorStatus,
  PortalStatus,
  DebugShareResponse,
  GatewayMigratePlan,
} from "@/lib/api";
import { apiErrorFromResponse, errorMessage } from "@/lib/api-error";
import { useI18n } from "@/i18n";
import { en } from "@/i18n/en";
import type { Translations } from "@/i18n/types";

/** System page copy; en seeds the optional block, other locales fall back. */
type SystemPageCopy = NonNullable<Translations["systemPage"]>;
function systemPageCopy(t: Translations): SystemPageCopy {
  return t.systemPage ?? (en.systemPage as SystemPageCopy);
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

function formatDuration(seconds: number): string {
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (d > 0) return `${d}d ${h}h ${m}m`;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

type BackupImportTarget =
  | { kind: "upload"; file: File }
  | { kind: "path"; path: string };

function backupImportLabel(target: BackupImportTarget | null): string {
  if (!target) return "the archive";
  return target.kind === "upload" ? target.file.name : target.path;
}

function backupFileName(path: string | null, fallback: string): string {
  if (!path) return fallback;
  return path.split(/[\\/]/).filter(Boolean).pop() ?? path;
}

/**
 * Live action-log viewer for the spawn-based admin actions (doctor, audit,
 * backup, import, skills update, checkpoints prune, gateway start/stop).
 * Polls /api/actions/<name>/status until the process exits.
 */
function ActionLogViewer({
  action,
  onClose,
  onComplete,
}: {
  action: string;
  onClose: () => void;
  onComplete?: (action: string, exitCode: number | null) => void;
}) {
  const { t } = useI18n();
  const SP = systemPageCopy(t);
  const [lines, setLines] = useState<string[]>([]);
  const [running, setRunning] = useState(true);
  const [exitCode, setExitCode] = useState<number | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const completeRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    completeRef.current = false;
    const poll = async () => {
      try {
        const st = await api.getActionStatus(action, 400);
        if (cancelled) return;
        setLines(st.lines);
        setRunning(st.running);
        setExitCode(st.exit_code);
        if (!st.running && !completeRef.current) {
          completeRef.current = true;
          onComplete?.(action, st.exit_code);
        }
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
  }, [action, onComplete]);

  return (
    <Card>
      <CardContent className="py-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Terminal className="h-4 w-4 text-muted-foreground" />
            <span className="font-mono text-sm">{action}</span>
            {running ? (
              <Badge tone="warning">{SP.logRunning}</Badge>
            ) : (
              <Badge tone={exitCode === 0 ? "success" : "destructive"}>
                {exitCode === 0 ? SP.logDone : SP.logExit.replace("{code}", String(exitCode))}
              </Badge>
            )}
          </div>
          <Button ghost size="icon" onClick={onClose} aria-label={SP.logClose}>
            <X />
          </Button>
        </div>
        <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words bg-background/50 border border-border p-3 text-xs font-mono text-muted-foreground">
          {lines.length ? lines.join("\n") : SP.logStarting}
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

function memoryStatusLabel(
  status: MemoryProviderInfo["status"],
  SP: SystemPageCopy,
): string {
  switch (status) {
    case "ready":
      return SP.memoryStatusReady;
    case "needs_config":
      return SP.memoryStatusNeedsConfig;
    case "unavailable":
      return SP.memoryStatusUnavailable;
    default:
      return SP.memoryStatusMissing;
  }
}

const MEMORY_STATUS_TONE: Record<
  MemoryProviderInfo["status"],
  "success" | "warning" | "destructive" | "secondary"
> = {
  ready: "success",
  needs_config: "warning",
  unavailable: "destructive",
  missing: "destructive",
};

export default function SystemPage() {
  const { toast, showToast } = useToast();
  const { t } = useI18n();
  const SP = systemPageCopy(t);

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
  const [consoleOpen, setConsoleOpen] = useState(false);
  const [migratePlan, setMigratePlan] = useState<GatewayMigratePlan | null>(null);

  // Add-credential form.
  const [credProvider, setCredProvider] = useState("openrouter");
  const [credKey, setCredKey] = useState("");
  const [credLabel, setCredLabel] = useState("");
  const [addingCred, setAddingCred] = useState(false);

  const [pendingBackupArchive, setPendingBackupArchive] = useState<string | null>(
    null,
  );
  const [downloadableBackupArchive, setDownloadableBackupArchive] = useState<
    string | null
  >(null);
  const [downloadingBackup, setDownloadingBackup] = useState(false);
  const importUploadInputRef = useRef<HTMLInputElement | null>(null);
  const [importFile, setImportFile] = useState<File | null>(null);
  const [importPath, setImportPath] = useState("");
  // Restore-from-backup is destructive (overwrites the live config) and the
  // spawned `hermes import` runs non-interactively (stdin is /dev/null), so
  // its CLI "Continue? [y/N]" prompt would auto-abort. The dashboard owns the
  // consent: confirm here, then call the endpoint with force=true.
  const [importingBackup, setImportingBackup] = useState(false);
  const [importConfirmTarget, setImportConfirmTarget] =
    useState<BackupImportTarget | null>(null);

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
      api.getGatewayMigratePlan(),
    ])
      .then(([s, st, m, p, c, h, cur, prt, upd, mig]) => {
        if (s.status === "fulfilled") setStatus(s.value);
        if (st.status === "fulfilled") setStats(st.value);
        if (m.status === "fulfilled") setMemory(m.value);
        if (p.status === "fulfilled") setPool(p.value.providers);
        if (c.status === "fulfilled") setCheckpoints(c.value);
        if (h.status === "fulfilled") setHooks(h.value);
        if (cur.status === "fulfilled") setCurator(cur.value);
        if (prt.status === "fulfilled") setPortal(prt.value);
        if (upd.status === "fulfilled") setUpdateInfo(upd.value);
        if (mig.status === "fulfilled") setMigratePlan(mig.value);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    loadAll();
  }, [loadAll]);

  // ── Gateway lifecycle ──────────────────────────────────────────────
  // A profile served by the shared multiplexer has no gateway of its own: Restart restarts
  // the ONE process every bot on this device runs in, so confirm first and say so after;
  // Start/Stop answer 409 with an explanation that belongs in a notice, not a raw error.
  const sharedGateway = sharedGatewayProfiles(status);
  const [sharedRestartOpen, setSharedRestartOpen] = useState(false);
  const [servedNotice, setServedNotice] = useState<string | null>(null);
  const runGateway = async (verb: "start" | "stop" | "restart"): Promise<boolean> => {
    setServedNotice(null);
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
      showToast(SP.toastGatewayStarted.replace("{verb}", verb), "success");
      setTimeout(loadAll, 3000);
      return true;
    } catch (e) {
      const refusal = servedProfileRefusal(e);
      if (refusal) {
        setServedNotice(refusal);
        return false;
      }
      showToast(gatewayActionFailedMessage(verb, errorMessage(e), e), "error");
      return false;
    }
  };
  const requestRestart = () => {
    if (sharedGateway) {
      setSharedRestartOpen(true);
      return;
    }
    void runGateway("restart");
  };
  // Same completion rule as the Desktop: the restart child exiting 0, or still running when the
  // bounded poll ends (in a no-service install it BECOMES the gateway and never exits), is
  // success — then the "(N bots)" toast; a non-zero exit is the action log's failure to show.
  const restartShared = async () => {
    const bots = sharedGateway?.length ?? 0;
    const started = await runGateway("restart");
    if (!started) return;
    for (let attempt = 0; attempt < 18; attempt += 1) {
      await new Promise((r) => setTimeout(r, 1200));
      const st = await api.getActionStatus("gateway-restart", 1).catch(() => null);
      if (st && !st.running) {
        if (st.exit_code != null && st.exit_code !== 0) return;
        break;
      }
    }
    showToast(sharedGatewayRestartedMessage(bots), "success");
  };

  const migrateToMultiplex = async () => {
    try {
      await api.migrateGatewayToMultiplex();
      setActiveAction("gateway-migrate");
      showToast(SP.toastMigrating, "success");
      setTimeout(loadAll, 5000);
    } catch (e) {
      showToast(SP.toastMigrateFailed.replace("{error}", errorMessage(e)), "error");
    }
  };

  // ── Curator ────────────────────────────────────────────────────────
  const toggleCuratorPaused = async () => {
    if (!curator) return;
    try {
      await api.setCuratorPaused(!curator.paused);
      showToast(curator.paused ? SP.toastCuratorResumed : SP.toastCuratorPaused, "success");
      loadAll();
    } catch (e) {
      showToast(SP.toastCuratorToggleFailed.replace("{error}", errorMessage(e)), "error");
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
          showToast(SP.toastReset.replace("{items}", res.deleted.join(", ") || SP.toastResetNothing), "success");
          loadAll();
        } catch (e) {
          showToast(SP.toastResetFailed.replace("{error}", errorMessage(e)), "error");
          throw e;
        }
      },
      [loadAll, showToast],
    ),
  });

  // ── Credential pool ────────────────────────────────────────────────
  const addCredential = async () => {
    if (!credProvider.trim() || !credKey.trim()) {
      showToast(SP.toastProviderRequired, "error");
      return;
    }
    setAddingCred(true);
    try {
      await api.addCredentialPoolEntry(
        credProvider.trim(),
        credKey.trim(),
        credLabel.trim() || undefined,
      );
      showToast(SP.toastCredentialAdded, "success");
      setCredKey("");
      setCredLabel("");
      loadAll();
    } catch (e) {
      showToast(SP.toastCredentialAddFailed.replace("{error}", errorMessage(e)), "error");
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
          showToast(SP.toastCredentialRemoved, "success");
          loadAll();
        } catch (e) {
          showToast(SP.toastCredentialRemoveFailed.replace("{error}", errorMessage(e)), "error");
          throw e;
        }
      },
      [loadAll, showToast],
    ),
  });

  // ── Operations ─────────────────────────────────────────────────────
  const runOp = async (fn: () => Promise<{ name: string }>, label: string) => {
    try {
      const res = await fn();
      setActiveAction(res.name);
      showToast(SP.toastOpStarted.replace("{label}", label), "success");
    } catch (e) {
      showToast(SP.toastOpFailed.replace("{label}", label).replace("{error}", errorMessage(e)), "error");
    }
  };

  const runDashboardBackup = async () => {
    try {
      const res = await api.runBackup();
      setActiveAction(res.name);
      setPendingBackupArchive(res.archive ?? null);
      setDownloadableBackupArchive(null);
      showToast(SP.toastBackupStarted, "success");
    } catch (e) {
      showToast(SP.toastBackupFailed.replace("{error}", errorMessage(e)), "error");
    }
  };

  const handleActionComplete = useCallback(
    (action: string, exitCode: number | null) => {
      if (action === "backup" && pendingBackupArchive) {
        if (exitCode === 0) {
          setDownloadableBackupArchive(pendingBackupArchive);
          showToast(SP.toastBackupReady, "success");
        } else {
          setPendingBackupArchive(null);
        }
      }
    },
    [pendingBackupArchive, showToast],
  );

  const downloadBackup = async () => {
    const archive = downloadableBackupArchive;
    if (!archive) return;
    setDownloadingBackup(true);
    try {
      const res = await api.downloadBackup(archive);
      if (!res.ok) {
        throw apiErrorFromResponse(res.status, await res.text().catch(() => ""), res.url);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = backupFileName(archive, "hermes-backup.zip");
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      showToast(SP.toastDownloadFailed.replace("{error}", errorMessage(e)), "error");
    } finally {
      setDownloadingBackup(false);
    }
  };

  const clearImportFile = () => {
    setImportFile(null);
    if (importUploadInputRef.current) importUploadInputRef.current.value = "";
  };

  const runBackupImport = async (target: BackupImportTarget) => {
    setImportingBackup(true);
    try {
      const res =
        target.kind === "upload"
          ? await api.runImportUpload(target.file, true)
          : await api.runImport(target.path, true);
      setActiveAction(res.name);
      showToast(SP.toastImportStarted, "success");
      if (target.kind === "upload") clearImportFile();
    } catch (e) {
      showToast(SP.toastImportFailed.replace("{error}", errorMessage(e)), "error");
    } finally {
      setImportingBackup(false);
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
      if (await copyTextToClipboard(text)) {
        setCopiedLabel(label);
        setTimeout(
          () => setCopiedLabel((cur) => (cur === label ? null : cur)),
          1500,
        );
      } else {
        showToast(SP.toastCopyFailed, "error");
      }
    },
    [showToast],
  );

  const runDebugShare = useCallback(async () => {
    setSharing(true);
    setShareResult(null);
    try {
      const res = await api.runDebugShare({ redact: shareRedact });
      setShareResult(res);
      const n = Object.keys(res.urls).length;
      showToast(
        `${SP.toastUploaded.replace("{count}", String(n)).replace("{s}", n === 1 ? "" : "s")}${
          res.redacted ? " (redacted)" : ""
        }`,
        "success",
      );
    } catch (e) {
      showToast(SP.toastDebugShareFailed.replace("{error}", errorMessage(e)), "error");
    } finally {
      setSharing(false);
    }
  }, [shareRedact, showToast]);


  // ── Update check / apply ───────────────────────────────────────────
  const checkForUpdate = useCallback(
    async (force = false) => {
      if (status?.can_update_hermes === false) return;
      setCheckingUpdate(true);
      try {
        const info = await api.checkHermesUpdate(force);
        setUpdateInfo(info);
        if (force) {
          if (info.update_available) {
            showToast(
              info.behind && info.behind > 0
                ? SP.toastUpdateAvailableBehind
                    .replace("{count}", String(info.behind))
                    .replace("{s}", info.behind === 1 ? "" : "s")
                : SP.toastUpdateAvailable,
              "success",
            );
          } else if (info.behind === 0) {
            showToast(SP.toastLatest, "success");
          } else if (info.message) {
            showToast(info.message, "error");
          }
        }
      } catch (e) {
        showToast(SP.toastUpdateCheckFailed.replace("{error}", errorMessage(e)), "error");
      } finally {
        setCheckingUpdate(false);
      }
    },
    [showToast, status?.can_update_hermes],
  );

  // Auto-check (cached) runs inside loadAll on mount; this is the
  // user-triggered forced re-check from the "Check for updates" button.
  const applyUpdate = async () => {
    setUpdateConfirmOpen(false);
    if (status?.can_update_hermes === false) {
      showToast(
        SP.toastManagedOutside,
        "success",
      );
      return;
    }
    try {
      const resp = await api.updateHermes();
      if (!resp.ok) {
        showToast(
          resp.message ??
            SP.toastUpdatesDontApply,
          "success",
        );
        return;
      }
      setActiveAction(resp.name ?? "hermes-update");
      showToast(SP.toastUpdateStarted, "success");
    } catch (e) {
      showToast(SP.toastUpdateFailed.replace("{error}", errorMessage(e)), "error");
    }
  };

  const checkpointsPrune = useConfirmDelete({
    onDelete: useCallback(async () => {
      try {
        const res = await api.pruneCheckpoints();
        setActiveAction(res.name);
        showToast(SP.toastPruneStarted, "success");
      } catch (e) {
        showToast(SP.toastPruneFailed.replace("{error}", errorMessage(e)), "error");
        throw e;
      }
    }, [showToast]),
  });

  // ── Hooks ──────────────────────────────────────────────────────────
  const createHook = async () => {
    if (!hookCommand.trim()) {
      showToast(SP.toastCommandRequired, "error");
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
      showToast(SP.toastHookCreated, "success");
      setHookCommand("");
      setHookMatcher("");
      setHookTimeout("");
      setHookModalOpen(false);
      loadAll();
    } catch (e) {
      showToast(SP.toastHookCreateFailed.replace("{error}", errorMessage(e)), "error");
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
          showToast(SP.toastHookRemoved, "success");
          loadAll();
        } catch (e) {
          showToast(SP.toastHookRemoveFailed.replace("{error}", errorMessage(e)), "error");
          throw e;
        }
      },
      [loadAll, showToast],
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
  const canUpdateHermes = status?.can_update_hermes !== false;
  const activeMemoryProvider = memory?.active
    ? memory.providers.find((provider) => provider.name === memory.active)
    : null;
  const validEvents = hooks?.valid_events?.length
    ? hooks.valid_events
    : HOOK_EVENTS_FALLBACK;

  return (
    <div className="flex flex-col gap-8">
      <Toast toast={toast} />
      <input
        ref={importUploadInputRef}
        type="file"
        accept=".zip,application/zip,application/x-zip-compressed"
        className="hidden"
        onChange={(event) => {
          setImportFile(event.currentTarget.files?.[0] ?? null);
        }}
      />

      <ConfirmDialog
        open={sharedRestartOpen}
        onCancel={() => setSharedRestartOpen(false)}
        onConfirm={() => {
          setSharedRestartOpen(false);
          void restartShared();
        }}
        title={SP.restartSharedTitle}
        description={sharedGatewayRestartDescription(sharedGateway ?? [])}
        confirmLabel={SP.restartAll}
      />

      <ConfirmDialog
        open={canUpdateHermes && updateConfirmOpen}
        onCancel={() => setUpdateConfirmOpen(false)}
        onConfirm={() => void applyUpdate()}
        title={SP.updateConfirmTitle}
        description={
          updateInfo && updateInfo.behind && updateInfo.behind > 0
            ? SP.updateConfirmBehind
                .replace("{cmd}", updateInfo.update_command)
                .replace("{n}", String(updateInfo.behind))
                .replace("{s}", updateInfo.behind === 1 ? "" : "s")
            : SP.updateConfirmGeneric.replace(
                "{cmd}",
                updateInfo?.update_command ?? "hermes update",
              )
        }
        confirmLabel={SP.updateNow}
      />

      <DeleteConfirmDialog
        open={memoryReset.isOpen}
        onCancel={memoryReset.cancel}
        onConfirm={memoryReset.confirm}
        title={SP.resetMemoryTitle}
        description={SP.resetMemoryBody}
        loading={memoryReset.isDeleting}
      />
      <DeleteConfirmDialog
        open={credDelete.isOpen}
        onCancel={credDelete.cancel}
        onConfirm={credDelete.confirm}
        title={SP.removeCredentialTitle}
        description={SP.removeCredentialBody}
        loading={credDelete.isDeleting}
      />
      <DeleteConfirmDialog
        open={checkpointsPrune.isOpen}
        onCancel={checkpointsPrune.cancel}
        onConfirm={checkpointsPrune.confirm}
        title={SP.pruneCheckpointsTitle}
        description={SP.pruneCheckpointsBody}
        loading={checkpointsPrune.isDeleting}
      />
      <DeleteConfirmDialog
        open={hookDelete.isOpen}
        onCancel={hookDelete.cancel}
        onConfirm={hookDelete.confirm}
        title={SP.removeHookTitle}
        description={SP.removeHookBody}
        loading={hookDelete.isDeleting}
      />
      <HermesConsoleModal
        open={consoleOpen}
        onClose={() => setConsoleOpen(false)}
      />

      {/* Create-hook modal */}
      {hookModalOpen && (
        <div
          ref={hookModalRef}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-background/85 p-4"
          onClick={(e) => e.target === e.currentTarget && setHookModalOpen(false)}
          role="dialog"
          aria-modal="true"
        >
          <div className={cn(themedBody, "relative w-full max-w-lg border border-border bg-card shadow-2xl flex flex-col")}>
            <Button
              ghost
              size="icon"
              onClick={() => setHookModalOpen(false)}
              className="absolute right-2 top-2 text-muted-foreground hover:text-foreground"
              aria-label={SP.close}
            >
              <X />
            </Button>
            <header className="p-5 pb-3 border-b border-border">
              <h2 className="font-mondwest text-display text-base tracking-wider">
                {SP.newShellHookTitle}
              </h2>
            </header>
            <div className="p-5 grid gap-4">
              <div className="grid gap-2">
                <Label htmlFor="hook-event">{SP.eventLabel}</Label>
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
                <Label htmlFor="hook-command">{SP.commandLabel}</Label>
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
                  <Label htmlFor="hook-matcher">{SP.matcherLabel}</Label>
                  <Input
                    id="hook-matcher"
                    placeholder={SP.matcherPlaceholder}
                    value={hookMatcher}
                    onChange={(e) => setHookMatcher(e.target.value)}
                  />
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="hook-timeout">{SP.timeoutLabel}</Label>
                  <Input
                    id="hook-timeout"
                    placeholder="10"
                    value={hookTimeout}
                    onChange={(e) => setHookTimeout(e.target.value)}
                  />
                </div>
              </div>
              <div className="flex items-center gap-2.5">
                <Checkbox
                  checked={hookApprove}
                  id="hook-approve"
                  onCheckedChange={(checked) => setHookApprove(checked === true)}
                />

                <Label
                  className="cursor-pointer text-sm font-normal normal-case tracking-normal text-muted-foreground"
                  htmlFor="hook-approve"
                >
                  {SP.approveNow}
                </Label>
              </div>
              <p className="text-xs text-warning">
                {SP.hooksWarning}
              </p>
              <div className="flex justify-end">
                <Button
                  className="uppercase"
                  size="sm"
                  onClick={createHook}
                  disabled={creatingHook}
                  prefix={creatingHook ? <Spinner /> : undefined}
                >
                  {creatingHook ? SP.creating : SP.createHook}
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
          onComplete={handleActionComplete}
          onClose={() => setActiveAction(null)}
        />
      )}

      {/* ── Host / system stats ───────────────────────────────────── */}
      <section className="flex flex-col gap-3">
        <H2 variant="sm" className="flex items-center gap-2 text-muted-foreground">
          <Server className="h-4 w-4" /> {SP.hostHeading}
        </H2>
        <Card>
          <CardContent className="py-4">
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-y-3 gap-x-6 text-sm">
              <div>
                <div className="text-xs uppercase tracking-wider text-muted-foreground">{SP.osLabel}</div>
                <div>{stats?.os} {stats?.os_release}</div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-wider text-muted-foreground">{SP.archLabel}</div>
                <div>{stats?.arch}</div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-wider text-muted-foreground">{SP.hostLabel}</div>
                <div className="truncate">{stats?.hostname}</div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-wider text-muted-foreground">{SP.pythonLabel}</div>
                <div>{stats?.python_impl} {stats?.python_version}</div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-wider text-muted-foreground">{SP.hermesLabel}</div>
                <div className="flex items-center gap-2">
                  <span>v{stats?.hermes_version}</span>
                  {canUpdateHermes &&
                    updateInfo &&
                    (updateInfo.update_available ? (
                      <Badge tone="warning">
                        {updateInfo.behind && updateInfo.behind > 0
                          ? SP.behindCount.replace("{count}", String(updateInfo.behind))
                          : SP.updateAvailable}
                      </Badge>
                    ) : updateInfo.behind === 0 ? (
                      <Badge tone="success">{SP.latest}</Badge>
                    ) : null)}
                </div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-wider text-muted-foreground flex items-center gap-1">
                  <Cpu className="h-3 w-3" /> {SP.cpuLabel}
                </div>
                <div>
                  {SP.cores.replace("{count}", String(stats?.cpu_count ?? "—"))}
                  {typeof stats?.cpu_percent === "number"
                    ? ` · ${stats.cpu_percent.toFixed(0)}%`
                    : ""}
                </div>
              </div>
              {stats?.memory && (
                <div>
                  <div className="text-xs uppercase tracking-wider text-muted-foreground">{SP.memoryLabel}</div>
                  <div>
                    {formatBytes(stats.memory.used)} / {formatBytes(stats.memory.total)} ({stats.memory.percent}%)
                  </div>
                </div>
              )}
              {stats?.disk && (
                <div>
                  <div className="text-xs uppercase tracking-wider text-muted-foreground flex items-center gap-1">
                    <HardDrive className="h-3 w-3" /> {SP.diskLabel}
                  </div>
                  <div>
                    {formatBytes(stats.disk.used)} / {formatBytes(stats.disk.total)} ({stats.disk.percent}%)
                  </div>
                </div>
              )}
              {typeof stats?.uptime_seconds === "number" && (
                <div>
                  <div className="text-xs uppercase tracking-wider text-muted-foreground">{SP.uptimeLabel}</div>
                  <div>{formatDuration(stats.uptime_seconds)}</div>
                </div>
              )}
              {stats?.load_avg && stats.load_avg.length >= 3 && (
                <div>
                  <div className="text-xs uppercase tracking-wider text-muted-foreground">{SP.loadAvgLabel}</div>
                  <div>{stats.load_avg.map((n) => n.toFixed(2)).join(" / ")}</div>
                </div>
              )}
            </div>
            {stats && !stats.psutil && (
              <p className="mt-3 text-xs text-muted-foreground">
                {SP.psutilHint.split("{extra}")[0]}
                <span className="font-mono">psutil</span>
                {SP.psutilHint.split("{extra}")[1]}
              </p>
            )}
            {canUpdateHermes && (
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
                  {SP.checkForUpdates}
                </Button>
                {updateInfo?.update_available && updateInfo.can_apply && (
                  <Button
                    size="sm"
                    prefix={<Download className="h-3.5 w-3.5" />}
                    onClick={() => setUpdateConfirmOpen(true)}
                  >
                    {SP.updateNow}
                  </Button>
                )}
                {updateInfo &&
                  !updateInfo.can_apply &&
                  updateInfo.update_available && (
                    <span className="text-xs text-muted-foreground">
                      {SP.updateWith.split("{cmd}")[0]}
                      <span className="font-mono">{updateInfo.update_command}</span>
                      {SP.updateWith.split("{cmd}")[1]}
                    </span>
                  )}
                {updateInfo?.message && !updateInfo.update_available && (
                  <span className="text-xs text-muted-foreground">
                    {updateInfo.message}
                  </span>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </section>

      {/* ── Portal ────────────────────────────────────────────────── */}
      <section className="flex flex-col gap-3">
        <H2 variant="sm" className="flex items-center gap-2 text-muted-foreground">
          <Globe className="h-4 w-4" /> {SP.portalHeading}
        </H2>
        <Card>
          <CardContent className="flex flex-col gap-3 py-4">
            <div className="flex items-center gap-3">
              <Badge tone={portal?.logged_in ? "success" : "secondary"}>
                {portal?.logged_in ? SP.loggedIn : SP.notLoggedIn}
              </Badge>
              {portal?.provider && (
                <span className="text-sm text-muted-foreground">
                  {SP.inferenceProvider.replace("{provider}", portal.provider)}
                </span>
              )}
              <a
                href={portal?.subscription_url || "https://portal.nousresearch.com/manage-subscription"}
                target="_blank"
                rel="noreferrer"
                className="ml-auto text-xs text-primary underline"
              >
                {SP.manageSubscription}
              </a>
            </div>
            {portal?.features && portal.features.length > 0 && (
              <div className="flex flex-col gap-1 border-t border-border pt-3">
                <span className="text-xs uppercase tracking-wider text-muted-foreground">
                  {SP.toolGatewayRouting}
                </span>
                {portal.features.map((f) => (
                  <div key={f.label} className="flex items-center justify-between text-sm">
                    <span>{f.label}</span>
                    <span className="text-muted-foreground">{f.state}</span>
                  </div>
                ))}
              </div>
            )}
            {!portal?.logged_in && (
              <p className="text-xs text-muted-foreground">
                {SP.portalLoginHint.split("{cmd}")[0]}
                <span className="font-mono">hermes portal</span>
                {SP.portalLoginHint.split("{cmd}")[1]}
              </p>
            )}
          </CardContent>
        </Card>
      </section>

      {/* ── Curator ───────────────────────────────────────────────── */}
      <section className="flex flex-col gap-3">
        <H2 variant="sm" className="flex items-center gap-2 text-muted-foreground">
          <Sparkles className="h-4 w-4" /> {SP.curatorHeading}
        </H2>
        <Card>
          <CardContent className="flex items-center justify-between py-4">
            <div className="flex items-center gap-3">
              <Badge tone={curator?.paused ? "warning" : curator?.enabled ? "success" : "secondary"}>
                {curator?.paused ? SP.curatorPaused : curator?.enabled ? SP.curatorActive : SP.curatorDisabled}
              </Badge>
              <span className="text-sm text-muted-foreground">
                {curator?.interval_hours ? SP.curatorEvery.replace("{hours}", String(curator.interval_hours)) : ""}
                {curator?.last_run_at ? SP.curatorLastRun.replace("{time}", new Date(curator.last_run_at).toLocaleString()) : SP.curatorNever}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Button size="sm" ghost onClick={toggleCuratorPaused}>
                {curator?.paused ? SP.resume : SP.pause}
              </Button>
              <Button
                size="sm"
                ghost
                prefix={<Play className="h-3.5 w-3.5" />}
                onClick={() => runOp(api.runCurator, SP.curatorReviewAction)}
              >
                {SP.runNow}
              </Button>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* ── Gateway ───────────────────────────────────────────────── */}
      <section className="flex flex-col gap-3">
        <H2 variant="sm" className="flex items-center gap-2 text-muted-foreground">
          <Power className="h-4 w-4" /> {SP.gatewayHeading}
        </H2>
        <Card>
          <CardContent className="flex items-center justify-between py-4">
            <div className="flex items-center gap-3">
              <Badge tone={gatewayRunning ? "success" : "secondary"}>
                {gatewayRunning ? SP.gatewayRunning : SP.gatewayStopped}
              </Badge>
              <span className="text-sm text-muted-foreground">
                {gatewayStateDescription(status?.gateway_state, gatewayRunning)}
              </span>
              {gatewayStateNeedsLogs(status?.gateway_state) && (
                <Link to="/logs?file=gateway" className="text-sm underline">
                  {SP.openLogs}
                </Link>
              )}
            </div>
            <div className="flex items-center gap-2">
              <Button
                size="sm"
                className="uppercase"
                onClick={() => runGateway("start")}
                disabled={gatewayRunning}
                prefix={<Play className="h-3.5 w-3.5" />}
              >
                {SP.start}
              </Button>
              <Button
                size="sm"
                className="uppercase"
                onClick={requestRestart}
                prefix={<RotateCw className="h-3.5 w-3.5" />}
              >
                {SP.restart}
              </Button>
              <Button
                size="sm"
                className="uppercase text-warning"
                ghost
                onClick={() => runGateway("stop")}
                disabled={!gatewayRunning}
                prefix={<Power className="h-3.5 w-3.5" />}
              >
                {SP.stop}
              </Button>
            </div>
          </CardContent>
          {(sharedGateway || servedNotice) && (
            <CardContent className="border-t border-current/10 py-3 text-xs text-muted-foreground" data-slot="shared-gateway-notice">
              {servedNotice ?? SP.servedByShared.replace("{profiles}", sharedGateway!.join(", "))}
            </CardContent>
          )}
          {migratePlan && !migratePlan.already_multiplexed && migratePlan.profiles.length > 1 && (
            migratePlan.eligible || migratePlan.blockers.length > 0
          ) && (
            <CardContent className="flex flex-col gap-2 border-t border-border py-4 text-sm">
              <div className="flex items-center justify-between gap-3">
                <span className="text-muted-foreground">
                  {SP.multiplexBlurb}
                </span>
                <Button
                  size="sm"
                  className="uppercase"
                  onClick={migrateToMultiplex}
                  disabled={!migratePlan.eligible}
                  title={migratePlan.eligible ? undefined : SP.fixBlockers}
                >
                  {SP.migrateToMultiplex}
                </Button>
              </div>
              {migratePlan.blockers.map((b) => (
                <div key={b} className="text-warning">• {b}</div>
              ))}
            </CardContent>
          )}
        </Card>
      </section>

      {/* ── Memory ────────────────────────────────────────────────── */}
      <section className="flex flex-col gap-3">
        <H2 variant="sm" className="flex items-center gap-2 text-muted-foreground">
          <Brain className="h-4 w-4" /> {SP.memoryHeading}
        </H2>
        <Card>
          <CardContent className="flex flex-col gap-4 py-4">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
              <span>
                {SP.externalProvider}{" "}
                <span className="font-mono text-foreground">
                  {memory?.active || SP.builtinOnly}
                </span>
              </span>
              {activeMemoryProvider && (
                <Badge tone={MEMORY_STATUS_TONE[activeMemoryProvider.status]}>
                  {memoryStatusLabel(activeMemoryProvider.status, SP)}
                </Badge>
              )}
              <Link to="/plugins" className="underline">
                {SP.changeInPlugins}
              </Link>
              <span className="ml-auto">
                {SP.providerSetup}{" "}
                <Link to="/plugins" className="underline">
                  {SP.configureInPlugins}
                </Link>
              </span>
            </div>

            {activeMemoryProvider?.status === "missing" && (
              <p className="border border-destructive/50 px-3 py-2 text-xs text-destructive">
                {SP.providerMissing}
              </p>
            )}

            <div className="flex flex-wrap items-center gap-3 border-t border-border pt-3">
              <span className="text-xs text-muted-foreground">
                {SP.builtinFiles}{" "}
                {formatBytes(memory?.builtin_files.memory ?? 0)} · USER.md:{" "}
                {formatBytes(memory?.builtin_files.user ?? 0)}
              </span>
              <div className="flex items-center gap-2 ml-auto">
                <Button size="sm" ghost className="text-destructive" onClick={() => memoryReset.requestDelete("memory")}>
                  {SP.resetMemoryMd}
                </Button>
                <Button size="sm" ghost className="text-destructive" onClick={() => memoryReset.requestDelete("user")}>
                  {SP.resetUserMd}
                </Button>
                <Button size="sm" ghost className="text-destructive" onClick={() => memoryReset.requestDelete("all")}>
                  {SP.resetAll}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* ── Credential pool ───────────────────────────────────────── */}
      <section className="flex flex-col gap-3">
        <H2 variant="sm" className="flex items-center gap-2 text-muted-foreground">
          <KeyRound className="h-4 w-4" /> {SP.credentialHeading}
        </H2>
        <Card>
          <CardContent className="flex flex-col gap-4 py-4">
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 items-end">
              <div className="grid gap-2">
                <Label htmlFor="cred-provider">{SP.providerLabel}</Label>
                <Input id="cred-provider" value={credProvider} onChange={(e) => setCredProvider(e.target.value)} placeholder="openrouter" />
              </div>
              <div className="grid gap-2 sm:col-span-2">
                <Label htmlFor="cred-key">{SP.apiKeyLabel}</Label>
                <Input id="cred-key" type="password" value={credKey} onChange={(e) => setCredKey(e.target.value)} placeholder="sk-…" />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="cred-label">{SP.labelLabel}</Label>
                <Input id="cred-label" value={credLabel} onChange={(e) => setCredLabel(e.target.value)} placeholder={SP.optionalPlaceholder} />
              </div>
            </div>
            <div className="flex justify-end">
              <Button size="sm" className="uppercase" onClick={addCredential} disabled={addingCred} prefix={addingCred ? <Spinner /> : undefined}>
                {SP.addKey}
              </Button>
            </div>
            {pool.length === 0 && (
              <p className="text-sm text-muted-foreground">
                {SP.noPooledCredentials}
              </p>
            )}
            {pool.map((prov) => (
              <div key={prov.provider} className="flex flex-col gap-2">
                <span className="text-xs uppercase tracking-wider text-muted-foreground">
                  {prov.provider}
                </span>
                {prov.entries.map((entry) => (
                  <div key={`${prov.provider}-${entry.index}`} className="flex items-center gap-3 border border-border bg-background/40 px-3 py-2">
                    <span className="text-sm font-medium">{entry.label}</span>
                    <span className="font-mono text-xs text-muted-foreground">{entry.token_preview}</span>
                    <Badge tone="outline">{entry.auth_type}</Badge>
                    {entry.last_status && <Badge tone="secondary">{entry.last_status}</Badge>}
                    <Button ghost size="icon" className="ml-auto text-destructive" aria-label={SP.removeCredential} onClick={() => credDelete.requestDelete(`${prov.provider}|${entry.index}`)}>
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
          <Activity className="h-4 w-4" /> {SP.operationsHeading}
        </H2>
        <Card>
          <CardContent className="flex flex-wrap gap-2 py-4">
            <Button size="sm" ghost prefix={<Terminal className="h-3.5 w-3.5" />} onClick={() => setConsoleOpen(true)}>
              {SP.openConsole}
            </Button>
            <Button size="sm" ghost prefix={<Stethoscope className="h-3.5 w-3.5" />} onClick={() => runOp(api.runDoctor, SP.doctorAction)}>
              {SP.runDoctor}
            </Button>
            <Button size="sm" ghost prefix={<ShieldCheck className="h-3.5 w-3.5" />} onClick={() => runOp(api.runSecurityAudit, SP.securityAuditAction)}>
              {SP.securityAudit}
            </Button>
            <Button size="sm" ghost prefix={<RotateCw className="h-3.5 w-3.5" />} onClick={() => runOp(api.updateSkillsFromHub, SP.skillsUpdateAction)}>
              {SP.updateSkills}
            </Button>
            <Button size="sm" ghost prefix={<Activity className="h-3.5 w-3.5" />} onClick={() => runOp(api.runPromptSize, SP.promptSizeAction)}>
              {SP.promptSize}
            </Button>
            <Button size="sm" ghost prefix={<Database className="h-3.5 w-3.5" />} onClick={() => runOp(api.runDump, SP.supportDumpAction)}>
              {SP.supportDump}
            </Button>
            <Button size="sm" ghost prefix={<RotateCw className="h-3.5 w-3.5" />} onClick={() => runOp(api.runConfigMigrate, SP.configMigrateAction)}>
              {SP.migrateConfig}
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="flex flex-col gap-4 py-4">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-end">
              <div className="grid min-w-0 flex-1 gap-2">
                <Label>{SP.fullBackup}</Label>
                <div className="flex min-w-0 flex-col gap-2 sm:flex-row sm:items-center">
                  <Button
                    size="sm"
                    ghost
                    prefix={<Database className="h-3.5 w-3.5" />}
                    onClick={() => void runDashboardBackup()}
                  >
                    {SP.createBackup}
                  </Button>
                  <Button
                    size="sm"
                    ghost
                    disabled={!downloadableBackupArchive || downloadingBackup}
                    prefix={
                      downloadingBackup ? (
                        <Spinner className="h-3.5 w-3.5" />
                      ) : (
                        <Download className="h-3.5 w-3.5" />
                      )
                    }
                    onClick={() => void downloadBackup()}
                  >
                    {SP.downloadBackup}
                  </Button>
                  <span
                    className="min-w-0 truncate text-xs text-muted-foreground"
                    title={pendingBackupArchive ?? SP.noBackupCreated}
                  >
                    {backupFileName(pendingBackupArchive, SP.noBackupCreated)}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex flex-col gap-3 border-t border-border pt-4 sm:flex-row sm:items-end">
              <div className="grid min-w-0 flex-1 gap-2">
                <Label>{SP.restoreFromUpload}</Label>
                <div className="flex min-w-0 flex-col gap-2 sm:flex-row sm:items-center">
                  <Button
                    type="button"
                    size="sm"
                    ghost
                    disabled={importingBackup}
                    prefix={<Upload className="h-3.5 w-3.5" />}
                    onClick={() => importUploadInputRef.current?.click()}
                  >
                    {SP.chooseRestoreZip}
                  </Button>
                  <span
                    className="min-w-0 truncate text-xs text-muted-foreground"
                    title={importFile?.name ?? SP.noBackupSelected}
                  >
                    {importFile?.name ?? SP.noBackupSelected}
                  </span>
                </div>
              </div>
              <Button
                size="sm"
                ghost
                disabled={!importFile || importingBackup}
                prefix={importingBackup ? <Spinner /> : undefined}
                onClick={() => {
                  if (!importFile) return;
                  setImportConfirmTarget({ kind: "upload", file: importFile });
                }}
              >
                {SP.restoreUpload}
              </Button>
            </div>

            <div className="flex flex-col gap-3 border-t border-border pt-4 sm:flex-row sm:items-end">
              <div className="grid min-w-0 flex-1 gap-2">
                <Label htmlFor="import-path">{SP.restoreFromPath}</Label>
                <Input
                  id="import-path"
                  value={importPath}
                  onChange={(e) => setImportPath(e.target.value)}
                  placeholder="$HERMES_HOME/backups/hermes-backup.zip"
                />
              </div>
              <Button
                size="sm"
                ghost
                disabled={!importPath.trim() || importingBackup}
                prefix={importingBackup ? <Spinner /> : undefined}
                onClick={() => {
                  const path = importPath.trim();
                  if (!path) return;
                  setImportConfirmTarget({ kind: "path", path });
                }}
              >
                {SP.restorePath}
              </Button>
            </div>
            <ConfirmDialog
              open={!!importConfirmTarget}
              title={SP.restoreConfirmTitle}
              description={SP.restoreConfirmBody.replace("{target}", backupImportLabel(importConfirmTarget))}
              destructive
              confirmLabel={SP.restore}
              cancelLabel={SP.cancel}
              onCancel={() => setImportConfirmTarget(null)}
              onConfirm={() => {
                const target = importConfirmTarget;
                setImportConfirmTarget(null);
                if (target) void runBackupImport(target);
              }}
            />
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
                  <span className="text-sm font-medium">{SP.shareDebugHeading}</span>
                  <span className="text-xs text-muted-foreground max-w-prose">
                    {SP.shareDebugBody}
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
                {sharing ? SP.uploading : SP.generateShareLink}
              </Button>
            </div>

            <div className="flex items-center gap-2.5">
              <Checkbox
                checked={shareRedact}
                disabled={sharing}
                id="share-redact"
                onCheckedChange={(checked) => setShareRedact(checked === true)}
              />

              <Label
                className="cursor-pointer select-none text-xs font-normal normal-case tracking-normal text-muted-foreground"
                htmlFor="share-redact"
              >
                {SP.redactLabel}
              </Label>
            </div>

            {shareResult && (
              <div className="flex flex-col gap-2 border-t border-border pt-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Badge tone="success">{SP.uploaded}</Badge>
                    {shareResult.redacted ? (
                      <Badge tone="outline">{SP.redacted}</Badge>
                    ) : (
                      <Badge tone="warning">{SP.notRedacted}</Badge>
                    )}
                    <span className="flex items-center gap-1 text-xs text-muted-foreground">
                      <Clock className="h-3 w-3" />
                      {SP.autoDeletesIn.replace(
                        "{hours}",
                        String(Math.round(shareResult.auto_delete_seconds / 3600)),
                      )}
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
                      {SP.copyAll}
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
                      aria-label={SP.copyLinkAria.replace("{label}", label)}
                      onClick={() => void copyToClipboard(url, label)}
                    >
                      {copiedLabel === label ? <Check /> : <Copy />}
                    </Button>
                  </div>
                ))}

                {shareResult.failures.length > 0 && (
                  <span className="text-xs text-destructive">
                    {SP.someLogsFailed.split("{errors}")[0]}
                    {shareResult.failures.join("; ")}
                  </span>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </section>

      {/* ── Checkpoints ───────────────────────────────────────────── */}
      <section className="flex flex-col gap-3">
        <H2 variant="sm" className="flex items-center gap-2 text-muted-foreground">
          <Database className="h-4 w-4" /> {SP.checkpointsHeading}
        </H2>
        <Card>
          <CardContent className="flex items-center justify-between py-4">
            <span className="text-sm text-muted-foreground">
              {SP.checkpointsSummary
                .replace("{sessions}", String(checkpoints?.sessions.length ?? 0))
                .replace("{size}", formatBytes(checkpoints?.total_bytes ?? 0))}
            </span>
            <Button size="sm" ghost className="text-destructive" disabled={!checkpoints?.sessions.length} prefix={<Trash2 className="h-3.5 w-3.5" />} onClick={() => checkpointsPrune.requestDelete("all")}>
              {SP.prune}
            </Button>
          </CardContent>
        </Card>
      </section>

      {/* ── Shell hooks ───────────────────────────────────────────── */}
      <section className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <H2 variant="sm" className="flex items-center gap-2 text-muted-foreground">
            <Terminal className="h-4 w-4" /> {SP.shellHooksHeading}
          </H2>
          <Button size="sm" className="uppercase" prefix={<Plus className="h-3.5 w-3.5" />} onClick={() => setHookModalOpen(true)}>
            {SP.newHook}
          </Button>
        </div>
        {(!hooks || hooks.hooks.length === 0) && (
          <Card>
            <CardContent className="py-6 text-center text-sm text-muted-foreground">
              {SP.noShellHooks}
            </CardContent>
          </Card>
        )}
        {hooks?.hooks.map((h: HookEntry, i) => (
          <Card key={`${h.event}-${i}`}>
            <CardContent className="flex items-center gap-3 py-3">
              <Badge tone="outline">{h.event}</Badge>
              {h.matcher && (
                <span className="text-xs text-muted-foreground">{SP.hookMatcher.replace("{matcher}", h.matcher)}</span>
              )}
              <span className="font-mono text-xs truncate flex-1">{h.command}</span>
              {h.executable === false && (
                <Badge tone="destructive">{SP.notExecutable}</Badge>
              )}
              <Badge tone={h.allowed ? "success" : "warning"}>
                {h.allowed ? SP.allowed : SP.notApproved}
              </Badge>
              <Button
                ghost
                size="icon"
                className="text-destructive"
                aria-label={SP.removeHook}
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
