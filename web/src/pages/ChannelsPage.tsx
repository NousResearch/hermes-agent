import { useCallback, useEffect, useLayoutEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Check,
  CheckCircle2,
  ExternalLink,
  Info,
  PlugZap,
  QrCode,
  Radio,
  RotateCw,
  Save,
  Settings2,
  WifiOff,
  X,
} from "lucide-react";
import * as QRCode from "qrcode";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Switch } from "@nous-research/ui/ui/components/switch";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { api } from "@/lib/api";
import type {
  MessagingPlatform,
  MessagingPlatformEnvVar,
  MessagingPlatformUpdate,
  TelegramOnboardingStartResponse,
} from "@/lib/api";
import { useModalBehavior } from "@/hooks/useModalBehavior";
import { usePageHeader } from "@/contexts/usePageHeader";
import { cn, themedBody } from "@/lib/utils";

// State → badge mapping. The backend emits a small, fixed vocabulary plus
// whatever the live gateway runtime reports (connected/disconnected/fatal).
const STATE_BADGE: Record<
  string,
  { tone: "success" | "warning" | "destructive" | "secondary" | "outline"; label: string }
> = {
  connected: { tone: "success", label: "Connected" },
  pending_restart: { tone: "warning", label: "Restart to apply" },
  gateway_stopped: { tone: "warning", label: "Gateway stopped" },
  startup_failed: { tone: "destructive", label: "Start failed" },
  disconnected: { tone: "warning", label: "Disconnected" },
  not_configured: { tone: "outline", label: "Not configured" },
  disabled: { tone: "secondary", label: "Disabled" },
  fatal: { tone: "destructive", label: "Error" },
};

function stateBadge(state: string) {
  return STATE_BADGE[state] ?? { tone: "outline" as const, label: state };
}

const TELEGRAM_USER_ID_RE = /^\d+$/;
const SLACK_MEMBER_ID_RE = /^[UW][A-Z0-9]{2,}$/;
const SLACK_TOKEN_PREFIXES: Record<string, string> = {
  SLACK_BOT_TOKEN: "xoxb-",
  SLACK_APP_TOKEN: "xapp-",
};

function validateMessagingEnvField(field: MessagingPlatformEnvVar, value: string): string | null {
  const trimmed = value.trim();
  if (!trimmed) return null;

  const expectedPrefix = SLACK_TOKEN_PREFIXES[field.key];
  if (expectedPrefix && !trimmed.startsWith(expectedPrefix)) {
    return `${field.prompt || field.key} must start with ${expectedPrefix}`;
  }

  if (field.key === "SLACK_ALLOWED_USERS") {
    // Mirror the gateway's parse (gateway/platforms/slack.py): drop empty
    // entries so a trailing/interior comma isn't rejected here. "*" is the
    // allow-all wildcard the gateway honors.
    const parts = trimmed
      .split(",")
      .map((part) => part.trim())
      .filter(Boolean);
    const invalid = parts.find((part) => part !== "*" && !SLACK_MEMBER_ID_RE.test(part));
    if (invalid) {
      return `${invalid} does not look like a Slack member ID. Use IDs like U01ABC2DEF3.`;
    }
  }

  return null;
}

function formatExpiry(expiresAt: string): string {
  const ms = Date.parse(expiresAt) - Date.now();
  if (!Number.isFinite(ms) || ms <= 0) return "expired";
  const seconds = Math.ceil(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return `${minutes}:${rest.toString().padStart(2, "0")}`;
}

function isTerminalTelegramOnboardingError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return /\b410\b/.test(message) && /\b(expired|claimed|gone)\b/i.test(message);
}

export default function ChannelsPage() {
  const [platforms, setPlatforms] = useState<MessagingPlatform[]>([]);
  const [envPath, setEnvPath] = useState("~/.hermes/.env");
  const [gatewayStartCommand, setGatewayStartCommand] = useState(
    "hermes gateway start",
  );
  const [loading, setLoading] = useState(true);
  const { toast, showToast } = useToast();
  const { setEnd } = usePageHeader();

  // Config modal state
  const [editing, setEditing] = useState<MessagingPlatform | null>(null);
  const [draftEnv, setDraftEnv] = useState<Record<string, string>>({});
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
  const closeEdit = useCallback(() => {
    setEditing(null);
    setFieldErrors({});
  }, []);
  const editModalRef = useModalBehavior({ open: editing !== null, onClose: closeEdit });

  // Per-card busy + restart-needed tracking
  const [togglingId, setTogglingId] = useState<string | null>(null);
  const [testingId, setTestingId] = useState<string | null>(null);
  const [restartNeeded, setRestartNeeded] = useState(false);
  const [restarting, setRestarting] = useState(false);

  const gatewayRunning = platforms.length > 0 && platforms[0].gateway_running;

  const load = useCallback(() => {
    return api
      .getMessagingPlatforms()
      .then((res) => {
        setPlatforms(res.platforms);
        setEnvPath(res.env_path || "~/.hermes/.env");
        setGatewayStartCommand(res.gateway_start_command || "hermes gateway start");
      })
      .catch((e) => showToast(`Error: ${e}`, "error"));
  }, [showToast]);

  useEffect(() => {
    load().finally(() => setLoading(false));
  }, [load]);

  const openConfig = (platform: MessagingPlatform) => {
    const initial: Record<string, string> = {};
    platform.env_vars.forEach((v) => {
      initial[v.key] = "";
    });
    setDraftEnv(initial);
    setFieldErrors({});
    setEditing(platform);
  };

  const handleSave = async () => {
    if (!editing) return;
    // Only send fields the user actually filled in — leaving a field blank
    // preserves the existing value rather than clobbering it.
    const env: Record<string, string> = {};
    Object.entries(draftEnv).forEach(([k, v]) => {
      if (v.trim()) env[k] = v.trim();
    });
    if (Object.keys(env).length === 0) {
      showToast("Nothing to save — fill in at least one field.", "error");
      return;
    }
    const missing = editing.env_vars.filter(
      (v) => v.required && !v.is_set && !env[v.key],
    );
    if (missing.length > 0) {
      showToast(`${missing[0].prompt || missing[0].key} is required`, "error");
      return;
    }
    const nextFieldErrors: Record<string, string> = {};
    editing.env_vars.forEach((field) => {
      const message = validateMessagingEnvField(field, draftEnv[field.key] || "");
      if (message) nextFieldErrors[field.key] = message;
    });
    if (Object.keys(nextFieldErrors).length > 0) {
      setFieldErrors(nextFieldErrors);
      showToast("Fix the highlighted fields before saving.", "error");
      return;
    }
    setSaving(true);
    try {
      const body: MessagingPlatformUpdate = { env, enabled: true };
      await api.updateMessagingPlatform(editing.id, body);
      showToast(`${editing.name} saved`, "success");
      setEditing(null);
      setRestartNeeded(true);
      await load();
    } catch (e) {
      showToast(`Failed to save: ${e}`, "error");
    } finally {
      setSaving(false);
    }
  };

  const handleToggle = async (platform: MessagingPlatform) => {
    const next = !platform.enabled;
    setTogglingId(platform.id);
    try {
      await api.updateMessagingPlatform(platform.id, { enabled: next });
      setPlatforms((prev) =>
        prev.map((p) =>
          p.id === platform.id
            ? { ...p, enabled: next, state: next ? "pending_restart" : "disabled" }
            : p,
        ),
      );
      setRestartNeeded(true);
    } catch (e) {
      showToast(`Error: ${e}`, "error");
    } finally {
      setTogglingId(null);
    }
  };

  const handleTest = async (platform: MessagingPlatform) => {
    setTestingId(platform.id);
    try {
      const res = await api.testMessagingPlatform(platform.id);
      showToast(`${platform.name}: ${res.message}`, res.ok ? "success" : "error");
    } catch (e) {
      showToast(`Error: ${e}`, "error");
    } finally {
      setTestingId(null);
    }
  };

  const handleRestart = async () => {
    setRestarting(true);
    try {
      await api.restartGateway();
      showToast("Gateway restarting…", "success");
      setRestartNeeded(false);
      // Give the gateway a moment to come up, then refresh status.
      setTimeout(() => void load(), 4000);
    } catch (e) {
      showToast(`Failed to restart: ${e}`, "error");
    } finally {
      setRestarting(false);
    }
  };

  useLayoutEffect(() => {
    setEnd(
      <Button
        className="uppercase"
        size="sm"
        onClick={handleRestart}
        disabled={restarting}
        prefix={restarting ? <Spinner /> : <RotateCw className="h-4 w-4" />}
      >
        {restarting ? "Restarting…" : "Restart gateway"}
      </Button>,
    );
    return () => setEnd(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setEnd, restarting]);

  const configured = useMemo(
    () => platforms.filter((p) => p.configured).length,
    [platforms],
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner className="text-2xl text-primary" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <Toast toast={toast} />

      {/* Restart banner */}
      {restartNeeded && (
        <Card className="border-warning/50">
          <CardContent className="flex flex-col gap-3 p-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-2 text-sm">
              <AlertTriangle className="h-4 w-4 shrink-0 text-warning" />
              <span>
                Changes are saved. Restart the gateway for them to take effect.
              </span>
            </div>
            <Button
              size="sm"
              className="uppercase shrink-0"
              onClick={handleRestart}
              disabled={restarting}
              prefix={restarting ? <Spinner /> : <RotateCw className="h-4 w-4" />}
            >
              {restarting ? "Restarting…" : "Restart now"}
            </Button>
          </CardContent>
        </Card>
      )}

      {!gatewayRunning && !restartNeeded && (
        <Card className="border-border">
          <CardContent className="flex items-center gap-2 p-4 text-sm text-muted-foreground">
            <WifiOff className="h-4 w-4 shrink-0" />
            <span>
              The gateway is not running. Configure channels here, then start the
              gateway with <code className="font-courier">{gatewayStartCommand}</code>{" "}
              (or the Restart button above).
            </span>
          </CardContent>
        </Card>
      )}

      <p className="text-xs text-muted-foreground">
        {configured} of {platforms.length} channels configured. Credentials are
        written to <code className="font-courier">{envPath}</code>; the
        gateway connects each enabled channel on its next restart.
      </p>

      {/* Config modal */}
      {editing && (
        <div
          ref={editModalRef}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-background/85 p-4"
          onClick={(e) => e.target === e.currentTarget && setEditing(null)}
          role="dialog"
          aria-modal="true"
          aria-labelledby="channel-config-title"
        >
          <div
            className={cn(
              themedBody,
              "relative w-full max-w-lg border border-border bg-card shadow-2xl flex flex-col max-h-[90vh]",
            )}
          >
            <Button
              ghost
              size="icon"
              onClick={() => setEditing(null)}
              className="absolute right-2 top-2 text-muted-foreground hover:text-foreground"
              aria-label="Close"
            >
              <X />
            </Button>

            <header className="p-5 pb-3 border-b border-border">
              <h2
                id="channel-config-title"
                className="font-mondwest text-display text-base tracking-wider"
              >
                Configure {editing.name}
              </h2>
              {editing.docs_url && (
                <a
                  href={editing.docs_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-1 inline-flex items-center gap-1 text-xs text-primary hover:underline"
                >
                  Setup guide <ExternalLink className="h-3 w-3" />
                </a>
              )}
            </header>

            <div className="p-5 grid gap-4 overflow-y-auto">
              <p className="text-xs text-muted-foreground">
                {editing.description}
              </p>
              {editing.env_vars.map((field: MessagingPlatformEnvVar) => (
                <div className="grid gap-1.5" key={field.key}>
                  <div className="flex items-center gap-1.5">
                    <Label htmlFor={`field-${field.key}`}>
                      {field.prompt || field.key}
                      {field.required ? " *" : ""}
                    </Label>
                    {field.help && (
                      <span
                        aria-label={field.help}
                        className="inline-flex text-muted-foreground hover:text-foreground"
                        role="img"
                        title={field.help}
                      >
                        <Info className="h-3.5 w-3.5" />
                      </span>
                    )}
                  </div>
                  {field.description && (
                    <span className="text-xs text-muted-foreground">
                      {field.description}
                    </span>
                  )}
                  <Input
                    id={`field-${field.key}`}
                    type={field.is_password ? "password" : "text"}
                    placeholder={
                      field.is_set
                        ? field.redacted_value || "•••••• (set — leave blank to keep)"
                        : field.key
                    }
                    value={draftEnv[field.key] ?? ""}
                    aria-invalid={Boolean(fieldErrors[field.key])}
                    onChange={(e) => {
                      const nextValue = e.target.value;
                      setDraftEnv((prev) => ({ ...prev, [field.key]: nextValue }));
                      setFieldErrors((prev) => {
                        if (!prev[field.key]) return prev;
                        const next = { ...prev };
                        delete next[field.key];
                        return next;
                      });
                    }}
                  />
                  {fieldErrors[field.key] && (
                    <span className="text-xs text-destructive">
                      {fieldErrors[field.key]}
                    </span>
                  )}
                </div>
              ))}

              <div className="flex justify-end gap-2 pt-1">
                <Button ghost size="sm" onClick={() => setEditing(null)}>
                  Cancel
                </Button>
                <Button
                  className="uppercase"
                  size="sm"
                  onClick={handleSave}
                  disabled={saving}
                  prefix={saving ? <Spinner /> : undefined}
                >
                  {saving ? "Saving…" : "Save & enable"}
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Platform list */}
      <div className="grid gap-3">
        {platforms.map((platform) => {
          const badge = stateBadge(platform.state);
          const busy = togglingId === platform.id;
          const StateIcon =
            platform.state === "connected"
              ? CheckCircle2
              : platform.state === "fatal" || platform.state === "startup_failed"
                ? AlertTriangle
                : Radio;
          return (
            <Card key={platform.id} className="border-border">
              <CardContent className="flex flex-col gap-4 p-4">
                <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <div className="flex items-start gap-3 min-w-0">
                    <StateIcon
                      className={cn(
                        "h-5 w-5 shrink-0 mt-0.5",
                        platform.state === "connected"
                          ? "text-success"
                          : platform.state === "fatal" ||
                              platform.state === "startup_failed"
                            ? "text-destructive"
                            : "text-muted-foreground",
                      )}
                    />
                    <div className="flex flex-col gap-0.5 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="font-mondwest normal-case text-sm font-medium">
                          {platform.name}
                        </span>
                        <Badge tone={badge.tone}>{badge.label}</Badge>
                      </div>
                      <span className="text-xs text-muted-foreground">
                        {platform.description}
                      </span>
                      {platform.error_message && (
                        <span className="text-xs text-destructive">
                          {platform.error_message}
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-2 shrink-0 self-start sm:self-center">
                    <div className="flex items-center gap-1.5">
                      {busy ? (
                        <Spinner className="text-sm" />
                      ) : (
                        <Switch
                          checked={platform.enabled}
                          onCheckedChange={() => void handleToggle(platform)}
                          aria-label={`Enable ${platform.name}`}
                        />
                      )}
                    </div>
                    <Button
                      ghost
                      size="sm"
                      onClick={() => handleTest(platform)}
                      disabled={testingId === platform.id}
                      prefix={
                        testingId === platform.id ? (
                          <Spinner />
                        ) : (
                          <PlugZap className="h-4 w-4" />
                        )
                      }
                    >
                      Test
                    </Button>
                    <Button
                      size="sm"
                      className="uppercase"
                      onClick={() => openConfig(platform)}
                      prefix={<Settings2 className="h-4 w-4" />}
                    >
                      Configure
                    </Button>
                  </div>
                </div>
                {platform.id === "telegram" && (
                  <TelegramOnboardingPanel
                    onChanged={load}
                    onRestartNeeded={() => setRestartNeeded(true)}
                    platform={platform}
                    setRestartNeeded={setRestartNeeded}
                    showToast={showToast}
                  />
                )}
                {platform.id === "weixin" && (
                  <WeixinOnboardingPanel
                    onChanged={load}
                    onRestartNeeded={() => setRestartNeeded(true)}
                    platform={platform}
                    setRestartNeeded={setRestartNeeded}
                    showToast={showToast}
                  />
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}

function TelegramOnboardingPanel({
  onChanged,
  onRestartNeeded,
  platform,
  setRestartNeeded,
  showToast,
}: {
  onChanged: () => Promise<void>;
  onRestartNeeded: () => void;
  platform: MessagingPlatform;
  setRestartNeeded: (needed: boolean) => void;
  showToast: (message: string, type: "success" | "error") => void;
}) {
  const [setup, setSetup] = useState<TelegramOnboardingStartResponse | null>(
    null,
  );
  const [qrDataUrl, setQrDataUrl] = useState("");
  const [phase, setPhase] = useState<
    "idle" | "starting" | "waiting" | "ready" | "applying"
  >("idle");
  const [botUsername, setBotUsername] = useState<string | null>(null);
  const [allowedIds, setAllowedIds] = useState<string[]>([]);
  const [detectedOwnerId, setDetectedOwnerId] = useState<string | null>(null);
  const [newAllowedId, setNewAllowedId] = useState("");
  const [error, setError] = useState("");
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (!setup || phase !== "waiting") return;
    let cancelled = false;
    let timeout: ReturnType<typeof setTimeout> | null = null;

    const poll = async () => {
      try {
        const status = await api.getTelegramOnboardingStatus(setup.pairing_id);
        if (cancelled) return;
        if (status.status === "ready") {
          setPhase("ready");
          setBotUsername(status.bot_username ?? null);
          setError("");
          if (
            status.owner_user_id &&
            TELEGRAM_USER_ID_RE.test(status.owner_user_id)
          ) {
            setDetectedOwnerId(status.owner_user_id);
            setAllowedIds([status.owner_user_id]);
          }
          return;
        }
        setError("");
        timeout = setTimeout(poll, 2000);
      } catch (pollError) {
        if (cancelled) return;

        const expiresAt = Date.parse(setup.expires_at);
        const expired =
          Number.isFinite(expiresAt) && Date.now() >= expiresAt;
        if (isTerminalTelegramOnboardingError(pollError) || expired) {
          setSetup(null);
          setQrDataUrl("");
          setPhase("idle");
          setError("Telegram pairing expired. Start a new QR setup to try again.");
          return;
        }

        setError(`Still waiting for Telegram. Retrying after: ${pollError}`);
        timeout = setTimeout(poll, 2000);
      }
    };

    timeout = setTimeout(poll, 1200);
    return () => {
      cancelled = true;
      if (timeout) clearTimeout(timeout);
    };
  }, [phase, setup]);

  useEffect(() => {
    if (!setup) return;
    const timer = setInterval(() => setTick((value) => value + 1), 1000);
    return () => clearInterval(timer);
  }, [setup]);

  const resetSetup = () => {
    setSetup(null);
    setQrDataUrl("");
    setPhase("idle");
    setBotUsername(null);
    setAllowedIds([]);
    setDetectedOwnerId(null);
    setNewAllowedId("");
    setError("");
  };

  const start = async () => {
    setPhase("starting");
    setError("");
    setBotUsername(null);
    setAllowedIds([]);
    setDetectedOwnerId(null);
    setNewAllowedId("");
    try {
      const res = await api.startTelegramOnboarding({ bot_name: "Hermes Agent" });
      const dataUrl = await QRCode.toDataURL(res.qr_payload, {
        errorCorrectionLevel: "M",
        margin: 1,
        width: 224,
      });
      setSetup(res);
      setQrDataUrl(dataUrl);
      setPhase("waiting");
    } catch (startError) {
      setPhase("idle");
      setError(String(startError));
    }
  };

  const cancel = async () => {
    if (setup) {
      try {
        await api.cancelTelegramOnboarding(setup.pairing_id);
      } catch {
        /* local cleanup still wins */
      }
    }
    resetSetup();
  };

  const addAllowedId = () => {
    const trimmed = newAllowedId.trim();
    if (!TELEGRAM_USER_ID_RE.test(trimmed)) {
      setError("Allowed Telegram user IDs must be numeric.");
      return;
    }
    setError("");
    setAllowedIds((ids) => (ids.includes(trimmed) ? ids : [...ids, trimmed]));
    setNewAllowedId("");
  };

  // restart_started only means the `hermes gateway restart` child spawned —
  // not that the restart will succeed (e.g. systemd linger missing, service
  // manager failure). Poll the action status briefly and surface a non-zero
  // exit via the manual-restart banner. Note: in no-service installs the
  // child becomes the foreground gateway and never exits, so "still running
  // when the window closes" counts as success.
  const watchRestartOutcome = async () => {
    for (let i = 0; i < 20; i++) {
      await new Promise((resolve) => setTimeout(resolve, 1500));
      try {
        const st = await api.getActionStatus("gateway-restart", 5);
        if (st.running) continue;
        if (st.exit_code !== 0 && st.exit_code !== null) {
          onRestartNeeded();
          showToast(
            `Gateway restart failed (exit ${st.exit_code}) — restart manually`,
            "error",
          );
        }
        return;
      } catch {
        // transient fetch error; keep polling
      }
    }
  };

  const apply = async () => {
    if (!setup) return;
    if (allowedIds.length === 0) {
      setError("Add at least one allowed Telegram user ID.");
      return;
    }
    setPhase("applying");
    setError("");
    try {
      const result = await api.applyTelegramOnboarding(setup.pairing_id, {
        allowed_user_ids: allowedIds,
      });
      resetSetup();
      if (result.restart_started) {
        showToast("Telegram saved; gateway restarting…", "success");
        setRestartNeeded(false);
        setTimeout(() => void onChanged(), 4000);
        void watchRestartOutcome();
      } else if (result.restart_started === undefined && result.needs_restart) {
        try {
          await api.restartGateway();
          showToast("Telegram saved; gateway restarting…", "success");
          setRestartNeeded(false);
          setTimeout(() => void onChanged(), 4000);
        } catch (restartError) {
          onRestartNeeded();
          showToast(`Telegram saved; gateway restart failed: ${restartError}`, "error");
        }
      } else {
        onRestartNeeded();
        const detail = result.restart_error ? `: ${result.restart_error}` : "";
        showToast(`Telegram saved; gateway restart failed${detail}`, "error");
      }
      await onChanged();
    } catch (applyError) {
      setPhase("ready");
      setError(String(applyError));
    }
  };

  const expiresIn = useMemo(
    () => (setup ? formatExpiry(setup.expires_at) : ""),
    // tick keeps the memo fresh without recalculating on every render branch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [setup, tick],
  );

  return (
    <div className="rounded-sm border border-border bg-background/35 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="sm"
          className="uppercase"
          onClick={() => void start()}
          disabled={phase === "starting" || phase === "waiting" || phase === "applying"}
          prefix={phase === "starting" ? <Spinner /> : <QrCode className="h-4 w-4" />}
        >
          {phase === "starting" ? "Starting…" : "Set up with QR"}
        </Button>
        {platform.configured && (
          <span className="text-xs text-muted-foreground">
            Existing Telegram credentials are configured.
          </span>
        )}
      </div>

      {error && (
        <div className="mt-3 border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {error}
        </div>
      )}

      {setup && qrDataUrl && (
        <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1fr)_260px]">
          <div className="grid gap-3">
            {(phase === "ready" || phase === "applying") && (
              <div className="grid gap-3">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge tone="success">Ready</Badge>
                  {botUsername && (
                    <span className="font-courier text-sm text-muted-foreground">
                      @{botUsername}
                    </span>
                  )}
                </div>

                <div className="grid gap-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-xs uppercase tracking-[0.12em] text-muted-foreground">
                      Allowed users
                    </span>
                    {detectedOwnerId && allowedIds.includes(detectedOwnerId) && (
                      <Badge tone="success">owner detected</Badge>
                    )}
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {allowedIds.map((id) => (
                      <button
                        key={id}
                        type="button"
                        className="inline-flex items-center gap-1 border border-border px-2 py-1 font-courier text-xs text-foreground hover:border-destructive/50"
                        onClick={() =>
                          setAllowedIds((ids) =>
                            ids.filter((existing) => existing !== id),
                          )
                        }
                      >
                        {id}
                        <X className="h-3 w-3" />
                      </button>
                    ))}
                    {allowedIds.length === 0 && (
                      <span className="text-sm text-muted-foreground">
                        Add at least one Telegram user ID.
                      </span>
                    )}
                  </div>
                </div>

                <div className="flex flex-col gap-2 sm:flex-row">
                  <Input
                    value={newAllowedId}
                    onChange={(event) => setNewAllowedId(event.target.value)}
                    placeholder="Telegram user ID"
                    className="font-courier"
                  />
                  <Button size="sm" outlined onClick={addAllowedId} prefix={<Check />}>
                    Add
                  </Button>
                </div>

                <div className="flex flex-wrap gap-2">
                  <Button
                    size="sm"
                    className="uppercase"
                    onClick={() => void apply()}
                    disabled={phase === "applying"}
                    prefix={phase === "applying" ? <Spinner /> : <Save className="h-4 w-4" />}
                  >
                    {phase === "applying" ? "Saving…" : "Save and restart"}
                  </Button>
                  <Button size="sm" ghost onClick={() => void cancel()}>
                    Cancel
                  </Button>
                </div>
              </div>
            )}
          </div>

          <div className="flex flex-col items-center justify-center gap-3">
            <img
              src={qrDataUrl}
              alt="Telegram setup QR code"
              className="h-56 w-56 bg-white p-2"
            />
            <div className="flex flex-wrap items-center justify-center gap-2 text-sm">
              <Badge tone={expiresIn === "expired" ? "destructive" : "outline"}>
                {expiresIn}
              </Badge>
              {phase === "waiting" && <Badge tone="warning">waiting</Badge>}
            </div>
            <div className="flex flex-wrap justify-center gap-2">
              <a
                href={setup.deep_link}
                target="_blank"
                rel="noreferrer"
                className="inline-flex h-8 items-center gap-1 border border-border px-3 text-xs uppercase text-foreground hover:border-foreground/40"
              >
                <ExternalLink className="h-4 w-4" />
                Open Telegram
              </a>
              <Button size="sm" ghost onClick={() => void cancel()}>
                Cancel
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function WeixinOnboardingPanel({
  onChanged,
  onRestartNeeded,
  platform,
  setRestartNeeded,
  showToast,
}: {
  onChanged: () => Promise<void>;
  onRestartNeeded: () => void;
  platform: MessagingPlatform;
  setRestartNeeded: (needed: boolean) => void;
  showToast: (message: string, type: "success" | "error") => void;
}) {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [qrImageSrc, setQrImageSrc] = useState("");
  const [phase, setPhase] = useState<
    "idle" | "starting" | "waiting" | "scanned" | "applying"
  >("idle");
  const [error, setError] = useState("");
  const [tick, setTick] = useState(0);
  const [expiresAt, setExpiresAt] = useState<string>("");

  // Poll for QR scan status while waiting.
  useEffect(() => {
    if (!sessionId || (phase !== "waiting" && phase !== "scanned")) return;
    let cancelled = false;
    let timeout: ReturnType<typeof setTimeout> | null = null;

    const poll = async () => {
      try {
        const status = await api.getWeixinOnboardingStatus(sessionId);
        if (cancelled) return;

        if (status.qr_image_base64) {
          setQrImageSrc(`data:image/png;base64,${status.qr_image_base64}`);
        }

        if (status.state === "waiting") {
          setPhase("waiting");
          setError("");
          timeout = setTimeout(poll, 2000);
          return;
        }
        if (status.state === "scanned") {
          setPhase("scanned");
          setError("");
          timeout = setTimeout(poll, 2000);
          return;
        }
        if (status.state === "confirmed") {
          // Auto-apply: save credentials and restart gateway.
          void apply();
          return;
        }
        if (status.state === "failed" || status.state === "expired" || status.state === "cancelled") {
          resetSetup();
          setError(
            status.error ||
              "WeChat setup expired or failed. Start a new QR setup to try again.",
          );
          return;
        }
        // starting or unknown — keep polling.
        timeout = setTimeout(poll, 2000);
      } catch (pollError) {
        if (cancelled) return;
        const parsedExpiresAt = Date.parse(expiresAt);
        const expired =
          Number.isFinite(parsedExpiresAt) && Date.now() >= parsedExpiresAt;
        if (expired) {
          resetSetup();
          setError("WeChat setup expired. Start a new QR setup to try again.");
          return;
        }
        setError(`Still waiting for WeChat. Retrying after: ${pollError}`);
        timeout = setTimeout(poll, 2000);
      }
    };

    timeout = setTimeout(poll, 1500);
    return () => {
      cancelled = true;
      if (timeout) clearTimeout(timeout);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, sessionId, expiresAt]);

  // Expiry countdown ticker.
  useEffect(() => {
    if (!sessionId) return;
    const timer = setInterval(() => setTick((value) => value + 1), 1000);
    return () => clearInterval(timer);
  }, [sessionId]);

  const resetSetup = () => {
    setSessionId(null);
    setQrImageSrc("");
    setPhase("idle");
    setExpiresAt("");
    setError("");
  };

  const start = async () => {
    setPhase("starting");
    setError("");
    setQrImageSrc("");
    try {
      const res = await api.startWeixinOnboarding({});
      setSessionId(res.session_id);
      setExpiresAt(res.expires_at);
      setPhase("waiting");
    } catch (startError) {
      setPhase("idle");
      setError(String(startError));
    }
  };

  const cancel = async () => {
    if (sessionId) {
      try {
        await api.cancelWeixinOnboarding(sessionId);
      } catch {
        /* local cleanup still wins */
      }
    }
    resetSetup();
  };

  const watchRestartOutcome = async () => {
    for (let i = 0; i < 20; i++) {
      await new Promise((resolve) => setTimeout(resolve, 1500));
      try {
        const st = await api.getActionStatus("gateway-restart", 5);
        if (st.running) continue;
        if (st.exit_code !== 0 && st.exit_code !== null) {
          onRestartNeeded();
          showToast(
            `Gateway restart failed (exit ${st.exit_code}) — restart manually`,
            "error",
          );
        }
        return;
      } catch {
        // transient fetch error; keep polling
      }
    }
  };

  const apply = async () => {
    if (!sessionId) return;
    setPhase("applying");
    setError("");
    try {
      const result = await api.applyWeixinOnboarding(sessionId, {});
      resetSetup();
      if (result.restart_started) {
        showToast("WeChat saved; gateway restarting…", "success");
        setRestartNeeded(false);
        setTimeout(() => void onChanged(), 4000);
        void watchRestartOutcome();
      } else if (result.restart_started === undefined && result.needs_restart) {
        try {
          await api.restartGateway();
          showToast("WeChat saved; gateway restarting…", "success");
          setRestartNeeded(false);
          setTimeout(() => void onChanged(), 4000);
        } catch (restartError) {
          onRestartNeeded();
          showToast(`WeChat saved; gateway restart failed: ${restartError}`, "error");
        }
      } else {
        onRestartNeeded();
        const detail = result.restart_error ? `: ${result.restart_error}` : "";
        showToast(`WeChat saved; gateway restart failed${detail}`, "error");
      }
      await onChanged();
    } catch (applyError) {
      // Don't go back to "waiting" — that would trigger the polling effect
      // to auto-call apply() again on the next "confirmed" status, creating
      // a retry loop with repeated writes/restarts. Reset to idle so the
      // user can manually retry by clicking "Set up with QR" again.
      resetSetup();
      setError(String(applyError));
    }
  };

  const expiresIn = useMemo(
    () => (expiresAt ? formatExpiry(expiresAt) : ""),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [expiresAt, tick],
  );

  return (
    <div className="rounded-sm border border-border bg-background/35 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="sm"
          className="uppercase"
          onClick={() => void start()}
          disabled={phase === "starting" || phase === "waiting" || phase === "scanned" || phase === "applying"}
          prefix={phase === "starting" ? <Spinner /> : <QrCode className="h-4 w-4" />}
        >
          {phase === "starting" ? "Starting…" : "Set up with QR"}
        </Button>
        {platform.configured && (
          <span className="text-xs text-muted-foreground">
            Existing WeChat credentials are configured.
          </span>
        )}
      </div>

      {error && (
        <div className="mt-3 border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {error}
        </div>
      )}

      {sessionId && qrImageSrc && (
        <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1fr)_260px]">
          <div className="grid gap-3">
            <div className="flex flex-wrap items-center gap-2">
              {phase === "waiting" && (
                <>
                  <Badge tone="warning">waiting</Badge>
                  <span className="text-sm text-muted-foreground">
                    Scan the QR code with WeChat on your phone.
                  </span>
                </>
              )}
              {phase === "scanned" && (
                <>
                  <Badge tone="success">scanned</Badge>
                  <span className="text-sm text-muted-foreground">
                    Confirm the login in WeChat to finish.
                  </span>
                </>
              )}
              {phase === "applying" && (
                <>
                  <Badge tone="success">confirmed</Badge>
                  <span className="text-sm text-muted-foreground">
                    Saving credentials and restarting gateway…
                  </span>
                </>
              )}
            </div>

            <div className="flex flex-wrap gap-2">
              <Button size="sm" ghost onClick={() => void cancel()}>
                Cancel
              </Button>
            </div>
          </div>

          <div className="flex flex-col items-center justify-center gap-3">
            <img
              src={qrImageSrc}
              alt="WeChat setup QR code"
              className="h-56 w-56 bg-white p-2"
            />
            <div className="flex flex-wrap items-center justify-center gap-2 text-sm">
              <Badge tone={expiresIn === "expired" ? "destructive" : "outline"}>
                {expiresIn}
              </Badge>
            </div>
            <Button size="sm" ghost onClick={() => void cancel()}>
              Cancel
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
